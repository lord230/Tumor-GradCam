"""
training/scheduler.py
======================
Learning rate scheduler utilities.

Provides:
    WarmupCosineScheduler — linear warmup followed by CosineAnnealingLR
    build_scheduler       — factory that reads config dict
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warm-up for `warmup_epochs`, then CosineAnnealing to `eta_min`.

    Works by computing a LR multiplier at each epoch:

        epoch < warmup_epochs  : lr = base_lr * (epoch / warmup_epochs)
        epoch >= warmup_epochs : cosine decay from base_lr → eta_min

    Args:
        optimizer:      PyTorch optimizer.
        warmup_epochs:  Number of linear warm-up epochs.
        total_epochs:   Total training epochs.
        eta_min:        Minimum LR at end of cosine decay.
        last_epoch:     Epoch to resume from (-1 = start fresh).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:

            warmup_factor = (epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        cosine_epochs = self.total_epochs - self.warmup_epochs
        progress      = (epoch - self.warmup_epochs) / max(cosine_epochs, 1)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]

def build_scheduler(
    optimizer: Optimizer,
    cfg: dict,
) -> _LRScheduler:
    """
    Build a scheduler from config dict.

    Currently supports:
        scheduler: "cosine"   → WarmupCosineScheduler

    Args:
        optimizer: The optimizer to schedule.
        cfg:       Parsed YAML config dict.

    Returns:
        A PyTorch _LRScheduler instance.
    """
    t = cfg["training"]
    scheduler_type = t.get("scheduler", "cosine")

    if scheduler_type == "cosine":
        return WarmupCosineScheduler(
            optimizer     = optimizer,
            warmup_epochs = t.get("warmup_epochs", 5),
            total_epochs  = t.get("epochs", 100),
            eta_min       = t.get("eta_min", 1e-6),
        )

    raise ValueError(f"Unknown scheduler type: '{scheduler_type}'")
