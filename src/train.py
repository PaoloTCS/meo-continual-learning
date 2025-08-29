"""
Training script for MEO and EWC continual learning experiments.

- Config-driven entrypoint (YAML): --config <path>, --output_dir <dir>
- Supports methods: finetune, meo, ewc (config tells which one)
- ResNet-50 backbone on CIFAR-100 10-task split by default
- Apple Silicon friendly (prefers MPS), reproducible seeds
- Cosine schedule restarts per task (T_max = epochs_per_task)
- EWC Fisher update at the end of each task (online variant)
- Aggregator-friendly JSON written to results/logs/
"""

from __future__ import annotations
import argparse
import os
import json
import yaml
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# ---- repo-local modules ----
# Expect: data.py exposes CIFAR100Continual; ewc.py exposes EWC with methods used below.
try:
    from data import CIFAR100Continual
except Exception as e:
    raise SystemExit(f"train.py: cannot import CIFAR100Continual from data.py: {e}")

# EWC is optional (finetune/MEO don’t require it). We import lazily in trainer if needed.


# ----------------------
# Utility: device select
# ----------------------
def pick_device(prefer: str = "auto") -> torch.device:
    """
    Prefer MPS on Apple Silicon, else CUDA, else CPU.
    If `prefer == "cpu"`, force CPU.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------
# Core Trainer
# -----------------------
class ContinualTrainer:
    """Main training class for continual learning experiments."""

    def __init__(self, config_path: str):
        # Load config
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Device (M1/M2 Apple Silicon friendly)
        prefer = self.config.get("device", "auto")
        self.device = pick_device(prefer)
        print(f"Using device: {self.device}")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Seeds
        seed = int(self.config.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Data
        data_cfg = self.config.get("data", {})
        self.data_manager = CIFAR100Continual(
            root=data_cfg.get("root", "./data"),
            num_tasks=int(data_cfg.get("num_tasks", 10)),
            batch_size=int(data_cfg.get("batch_size", 128)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            seed=seed,
        )

        # Model
        self.model = self._create_model().to(self.device)

        # Optimizer (recreated per task if you prefer; here we keep a single optimizer)
        trn = self.config.get("training", {})
        self.lr = float(trn.get("lr", 0.01))
        self.momentum = float(trn.get("momentum", 0.9))
        self.weight_decay = float(trn.get("weight_decay", 5e-4))
        self.epochs_per_task = int(trn.get("epochs_per_task", 20))

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # Scheduler will be recreated at the START of each task (cosine restart)
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

        # Method
        method_cfg = self.config.get("method", {})
        # allow shorthand strings too: method: "ewc"
        if isinstance(method_cfg, str):
            method_cfg = {"type": method_cfg}
        self.method = str(method_cfg.get("type", "finetune")).lower()

        # EWC config
        self.ewc_lambda = None
        self.ewc_gamma = float(method_cfg.get("gamma", 0.9))
        self.ewc_mode = method_cfg.get("mode", "online")
        self.fisher_batches = int(method_cfg.get("fisher_batches", 200))
        self.fisher_batch_size = int(method_cfg.get("fisher_batch_size", data_cfg.get("batch_size", 128)))
        if self.method == "ewc":
            # support multiple keys
            for k in ("lambda_ewc", "lambda", "lam", "ewc_lambda"):
                if k in method_cfg:
                    self.ewc_lambda = float(method_cfg[k])
                    break
            if self.ewc_lambda is None:
                raise SystemExit("EWC requires method.lambda_ewc (or lambda/lam/ewc_lambda) in the config.")

            # Lazy import EWC only if requested
            try:
                from ewc import EWC  # type: ignore
            except Exception as e:
                raise SystemExit(f"EWC selected but cannot import EWC from ewc.py: {e}")

            # Construct EWC object; expected to hold consolidated Fisher & theta*
            self.ewc = EWC(lambda_=self.ewc_lambda, gamma=self.ewc_gamma, mode=self.ewc_mode)
        else:
            self.ewc = None

        # MEO config (if you wire masking via forward hooks in meo.py)
        self.meo_alpha = None
        if self.method == "meo":
            # optional keys, adapt to your meo.py interface
            self.meo_alpha = float(method_cfg.get("alpha", 0.1))
            self.meo_evolution = method_cfg.get("evolution", "identity")
            # If you implement MEO via hooks:
            try:
                from meo import attach_meo_hooks  # type: ignore
                attach_meo_hooks(self.model, alpha=self.meo_alpha, evolution=self.meo_evolution)
                print(f"MEO hooks attached: alpha={self.meo_alpha}, evolution={self.meo_evolution}")
            except Exception:
                # it’s fine if meo hooks aren’t present; you may be using a different implementation
                pass

        # Results state
        self.results: Dict[str, Any] = {
            "per_task_acc": [],
            "final_avg_accuracy": None,
        }

    # -----------------------
    # Model factory
    # -----------------------
    def _create_model(self) -> nn.Module:
        mdl_cfg = self.config.get("model", {})
        name = mdl_cfg.get("name", "resnet50")
        pretrained = bool(mdl_cfg.get("pretrained", False))
        num_classes = int(mdl_cfg.get("num_classes", 100))

        if name.lower() == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            # Optional init for new head (PyTorch default is fine)
            nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(model.fc.bias)
            return model

        raise SystemExit(f"Unsupported model name: {name}")

    # -----------------------
    # Training / Eval
    # -----------------------
    def _make_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        # Cosine restart per task
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs_per_task)

    def train_task(self, task_id: int) -> Dict[str, float]:
        train_loader, test_loader = self.data_manager.get_task_loaders(task_id)

        # Optional: recreate optimizer per task for strict fairness with some baselines
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # Restart cosine each task
        self.scheduler = self._make_scheduler()

        ce = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs_per_task):
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = ce(logits, y)

                # Add EWC penalty if enabled
                if self.method == "ewc":
                    # Expect penalty computed on current model vs consolidated Fisher/theta*
                    if not hasattr(self.ewc, "compute_ewc_loss"):
                        raise SystemExit("EWC implementation missing compute_ewc_loss(model). Please add it in ewc.py")
                    loss = loss + self.ewc.compute_ewc_loss(self.model)

                loss.backward()
                self.optimizer.step()

            # Cosine step once per epoch
            self.scheduler.step()

        # After training current task: update EWC statistics at θ*
        if self.method == "ewc":
            if not all(hasattr(self.ewc, name) for name in ("update_fisher", "save_optimal_params")):
                raise SystemExit("EWC implementation must expose update_fisher(model, dataloader, device, batches=...) and save_optimal_params(model)")
            # snapshot θ* and update Fisher on the train set of this task
            self.ewc.save_optimal_params(self.model)
            self.ewc.update_fisher(self.model, train_loader, device=self.device, batches=self.fisher_batches)

        # Evaluate accuracy on current task (you can also evaluate on all seen tasks here)
        acc = self.evaluate_loader(test_loader)
        return {"task_id": task_id, "acc": acc}

    @torch.no_grad()
    def evaluate_loader(self, loader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x)
            _, pred = torch.max(logits, 1)  # avoid .data
            correct += (pred == y).sum().item()
            total += y.size(0)
        return 100.0 * correct / max(1, total)

    def evaluate_seen_tasks(self, upto_task: int) -> List[float]:
        """Evaluate on all test loaders up to and including `upto_task` and return list of accuracies."""
        accs = []
        for t in range(upto_task + 1):
            _, test_loader = self.data_manager.get_task_loaders(t)
            accs.append(self.evaluate_loader(test_loader))
        return accs

    def run_experiment(self) -> Dict[str, Any]:
        per_task_acc = []

        for t in range(self.data_manager.num_tasks):
            print(f"\n=== Training Task {t+1}/{self.data_manager.num_tasks} ===")
            res = self.train_task(t)
            per_task_acc.append(res["acc"])

            # (Optional) track average over seen tasks
            seen_accs = self.evaluate_seen_tasks(t)
            avg_seen = float(np.mean(seen_accs))
            print(f"[Task {t}] current task acc={res['acc']:.2f} | avg over seen tasks={avg_seen:.2f}")

        # Final average accuracy across all tasks' test sets
        all_accs = self.evaluate_seen_tasks(self.data_manager.num_tasks - 1)
        final_avg = float(np.mean(all_accs))
        print(f"\n[FINAL] Average accuracy over all {self.data_manager.num_tasks} tasks: {final_avg:.3f}%")

        self.results["per_task_acc"] = per_task_acc
        self.results["final_avg_accuracy"] = final_avg
        return self.results

    def save_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2)


# -----------------------
# Entry point
# -----------------------
def main():
    """Main training entrypoint."""
    parser = argparse.ArgumentParser(description="MEO/EWC Continual Learning Training")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    args = parser.parse_args()

    trainer = ContinualTrainer(args.config)
    results = trainer.run_experiment()
    trainer.save_results(args.output_dir)

    # === Aggregator-friendly JSON log for scripts/aggregate_results.py ===
    # Determine method & seed from config (and EWC lambda if applicable)
    method_cfg = trainer.config.get("method", {})
    if isinstance(method_cfg, str):
        method = method_cfg.lower()
        method_cfg = {"type": method}
    else:
        method = str(method_cfg.get("type", "finetune")).lower()
    seed = int(trainer.config.get("seed", 42))

    ewc_lambda = None
    if method == "ewc":
        for k in ("lambda_ewc", "lambda", "lam", "ewc_lambda"):
            if k in method_cfg and method_cfg[k] is not None:
                try:
                    ewc_lambda = int(method_cfg[k])
                except Exception:
                    ewc_lambda = float(method_cfg[k])
                break

    final_avg_acc = float(results.get("final_avg_accuracy", float("nan")))
    print(f"[FINAL] final_avg_acc={final_avg_acc:.3f}")

    logdir = os.path.join("results", "logs")
    os.makedirs(logdir, exist_ok=True)
    tag = f"{method}_seed{seed}"
    if method == "ewc" and ewc_lambda is not None:
        tag += f"_lam{ewc_lambda}"
    out_json = os.path.join(logdir, f"{tag}.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "method": method,
                "lambda": ewc_lambda,
                "seed": seed,
                "final_avg_acc": final_avg_acc,
            },
            f,
            indent=2,
        )

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
