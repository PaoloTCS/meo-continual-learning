from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

class EWC:
    """
    Online EWC with normalized Fisher:
      F_t = gamma * F_{t-1} + F_batch_avg
      penalty = 0.5 * lambda * sum_n F[n] * (theta_n - theta*_n)^2
    """
    def __init__(self, lambda_: float, gamma: float = 0.9, mode: str = "online"):
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.mode = mode
        self.means: Dict[str, torch.Tensor] = {}
        self.F: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def save_optimal_params(self, model: nn.Module) -> None:
        self.means = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    # NOTE: DO NOT decorate with @torch.no_grad() — we need autograd here.
    def update_fisher(self, model: nn.Module, dataloader, device: torch.device,
                      batches: int = 200) -> None:
        model.eval()
        ce = nn.CrossEntropyLoss()
        F_accum: Dict[str, torch.Tensor] = {}
        n_seen = 0

        it = iter(dataloader)
        # ensure grads are enabled even if outer context disabled them
        with torch.enable_grad():
            for _ in range(batches):
                try:
                    x, y = next(it)
                except StopIteration:
                    break
                x, y = x.to(device), y.to(device)

                model.zero_grad(set_to_none=True)
                logits = model(x)
                loss = ce(logits, y)
                loss.backward()

                for n, p in model.named_parameters():
                    if not p.requires_grad or (p.grad is None):
                        continue
                    g2 = p.grad.detach() ** 2
                    if n not in F_accum:
                        F_accum[n] = g2.clone()
                    else:
                        F_accum[n] += g2
                n_seen += y.numel()

        if n_seen == 0:
            return

        # normalize by number of samples to stabilize scale
        for n in F_accum:
            F_accum[n] /= float(n_seen)

        # online update
        for n, Fi in F_accum.items():
            if n in self.F:
                self.F[n] = self.gamma * self.F[n] + Fi
            else:
                self.F[n] = Fi.clone()

    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        # no penalty before first consolidation
        if not self.means or not self.F:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        pen = torch.zeros((), device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.means and n in self.F:
                pen = pen + (self.F[n] * (p - self.means[n]).pow(2)).sum()
        return 0.5 * self.lambda_ * pen
