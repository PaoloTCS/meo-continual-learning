# src/train.py
"""
Training script for MEO / EWC / Finetune continual learning on CIFAR-100.

- Config-driven: --config <yaml>, --output_dir <dir>
- Apple Silicon friendly (prefers MPS), falls back to CUDA->CPU
- Cosine LR restarts **per task** (fixes zero-LR-on-task-2)
- EWC Fisher update after each task when method == "ewc"
- Aggregator-friendly JSON logs in results/logs/
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Suppress harmless runpy warnings when using -m with multiprocessing
import warnings as _warnings
_warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")

# ---- repo-local ----
try:
    from src.data import CIFAR100Continual
except Exception as e:
    raise SystemExit(f"train.py: cannot import CIFAR100Continual from src.data: {e}")

# ----------------------
# Device selection
# ----------------------
def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ContinualTrainer:
    """Main training class for continual learning experiments."""

    def __init__(self, config_path: str):
        import yaml
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Device
        prefer = self.config.get("device", "auto")
        self.device = pick_device(prefer)
        print(f"Using device: {self.device}")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Seed
        seed = int(self.config.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Data
        data_cfg = self.config.get("data", {})
        self.data_manager = CIFAR100Continual(
            root=data_cfg.get("root", "./data"),
            num_tasks=int(data_cfg.get("num_tasks", 10)),
            batch_size=int(data_cfg.get("batch_size", 128)),
            num_workers=int(data_cfg.get("num_workers", 2)),
            seed=seed,
        )

        # Model
        self.model = self._create_model().to(self.device)

        # Train hyperparams
        trn = self.config.get("training", {})
        self.lr = float(trn.get("lr", 0.01))
        self.momentum = float(trn.get("momentum", 0.9))
        self.weight_decay = float(trn.get("weight_decay", 5e-4))
        self.epochs_per_task = int(trn.get("epochs_per_task", 20))

        # Created per-task (don’t persist across tasks)
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None

        # Method
        method_cfg = self.config.get("method", {})
        if isinstance(method_cfg, str):
            method_cfg = {"type": method_cfg}
        self.method = str(method_cfg.get("type", "finetune")).lower()

        # EWC config
        self.ewc_lambda = None
        self.ewc_gamma = float(method_cfg.get("gamma", 0.9))
        self.ewc_mode = method_cfg.get("mode", "online")
        self.fisher_batches = int(method_cfg.get("fisher_batches", 200))
        self.fisher_batch_size = int(method_cfg.get("fisher_batch_size", self.config.get("data", {}).get("batch_size", 128)))
        if self.method == "ewc":
            for k in ("lambda_ewc", "lambda", "lam", "ewc_lambda"):
                if k in method_cfg:
                    self.ewc_lambda = float(method_cfg[k])
                    break
            if self.ewc_lambda is None:
                raise SystemExit("EWC requires method.lambda_ewc (or lambda/lam/ewc_lambda) in the config.")
            try:
                from src.ewc import EWC  # your implementation
            except Exception as e:
                raise SystemExit(f"EWC selected but cannot import EWC from src.ewc: {e}")
            self.ewc = EWC(lambda_=self.ewc_lambda, gamma=self.ewc_gamma, mode=self.ewc_mode)
        else:
            self.ewc = None

        # Optional MEO hooks (if implemented in src/meo.py)
        self._meo_obj = None
        try:
            if self.method == "meo":
                self.meo_alpha = float(method_cfg.get("alpha", 0.1))
                self.meo_evolution = method_cfg.get("evolution", "identity")
                from src.meo import attach_meo_hooks  # optional
                attach = attach_meo_hooks(self.model, alpha=self.meo_alpha, evolution=self.meo_evolution)
                self._meo_obj = attach.get("meo")
                self._meo_layers = attach.get("layer_names", [])
                print(f"MEO hooks attached: alpha={self.meo_alpha}, evolution={self.meo_evolution}")
        except Exception:
            pass

        # Results
        self.results: Dict[str, Any] = {
            "per_task_acc": [],
            "final_avg_accuracy": None,
            "drift": {"per_epoch": [], "per_task": []},
        }

    # -----------------------
    # Model factory
    # -----------------------
    def _create_model(self) -> nn.Module:
        mdl = self.config.get("model", {})
        name = str(mdl.get("name", "resnet50")).lower()
        pretrained = bool(mdl.get("pretrained", False))
        num_classes = int(mdl.get("num_classes", 100))

        if name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(model.fc.bias)
            return model

        raise SystemExit(f"Unsupported model: {name}")

    # -----------------------
    # Optim/Sched builders
    # -----------------------
    def _make_optimizer(self) -> optim.Optimizer:
        for p in self.model.parameters():
            p.requires_grad = True
        return optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def _make_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        # Restart cosine each task
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs_per_task)

    # -----------------------
    # Train / Eval
    # -----------------------
    def train_task(self, task_id: int) -> Dict[str, float]:
        train_loader, test_loader = self.data_manager.get_task_loaders(task_id)

        # (Re)create optimizer & cosine scheduler **per task**
        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()

        ce = nn.CrossEntropyLoss()

        for epoch in range(self.epochs_per_task):
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = ce(logits, y)

                if self.method == "ewc":
                    if not hasattr(self.ewc, "compute_ewc_loss"):
                        raise SystemExit("EWC implementation missing compute_ewc_loss(model).")
                    loss = loss + self.ewc.compute_ewc_loss(self.model)

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            # Optional: drift metric per epoch for MEO
            if self.method == "meo" and self._meo_obj is not None:
                try:
                    drift_value = float(self._meo_obj.get_drift_metric(getattr(self, "_meo_layers", [])))
                    self.results["drift"]["per_epoch"].append(
                        {"task": task_id, "epoch": epoch, "drift": drift_value}
                    )
                except Exception:
                    pass

        # EWC consolidation at end of task
        if self.method == "ewc":
            if not all(hasattr(self.ewc, n) for n in ("save_optimal_params", "update_fisher")):
                raise SystemExit("EWC must expose save_optimal_params() and update_fisher().")
            self.ewc.save_optimal_params(self.model)
            self.ewc.update_fisher(self.model, train_loader, device=self.device, batches=self.fisher_batches)

        # Evaluate this task’s test set
        acc = self.evaluate_loader(test_loader)

        # Optional: drift per task
        if self.method == "meo" and self._meo_obj is not None:
            try:
                drift_value = float(self._meo_obj.get_drift_metric(getattr(self, "_meo_layers", [])))
                self.results["drift"]["per_task"].append({"task": task_id, "drift": drift_value})
            except Exception:
                pass

        return {"task_id": task_id, "acc": acc}

    @torch.no_grad()
    def evaluate_loader(self, loader) -> float:
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return 100.0 * correct / max(1, total)

    def evaluate_seen_tasks(self, upto_task: int) -> List[float]:
        accs: List[float] = []
        for t in range(upto_task + 1):
            _, test_loader = self.data_manager.get_task_loaders(t)
            accs.append(self.evaluate_loader(test_loader))
        return accs

    def run_experiment(self) -> Dict[str, Any]:
        per_task_acc: List[float] = []
        for t in range(self.data_manager.num_tasks):
            print(f"\n=== Training Task {t+1}/{self.data_manager.num_tasks} ===")
            res = self.train_task(t)
            per_task_acc.append(res["acc"])

            seen = self.evaluate_seen_tasks(t)
            avg_seen = float(np.mean(seen))
            print(f"[Task {t}] current acc={res['acc']:.2f} | avg over seen={avg_seen:.2f}")

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
        if "drift" in self.results:
            with open(os.path.join(output_dir, "drift.json"), "w") as f:
                json.dump(self.results["drift"], f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MEO/EWC Continual Learning Training")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    args = parser.parse_args()

    trainer = ContinualTrainer(args.config)
    results = trainer.run_experiment()
    trainer.save_results(args.output_dir)

    # Aggregator-friendly JSON
    method_cfg = trainer.config.get("method", {})
    method = method_cfg if isinstance(method_cfg, str) else str(method_cfg.get("type", "finetune")).lower()
    seed = int(trainer.config.get("seed", 42))

    ewc_lambda = None
    if method == "ewc" and isinstance(method_cfg, dict):
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
            {"method": method, "lambda": ewc_lambda, "seed": seed, "final_avg_acc": final_avg_acc},
            f,
            indent=2,
        )

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
