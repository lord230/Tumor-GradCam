"""
training/losses.py
==================
Loss functions for HybridMemoryNet training.

Available losses:
    - FocalLoss              : Class-imbalance robust classification loss
    - PrototypeCompactnessLoss: Pulls embeddings toward the correct prototype
    - HybridLoss              : Combines CE (or Focal) + PrototypeCompactness
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for dense classification.

    L_focal = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0).
        alpha: Scalar weighting factor (default 1.0).
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw logits
            targets: (B,)   class indices
        """
        log_p = F.log_softmax(logits, dim=1)
        p     = log_p.exp()

        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -self.alpha * (1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class PrototypeCompactnessLoss(nn.Module):
    """
    Prototype Compactness Loss.

    Minimises the squared L2 distance between each embedding and the
    prototype of its ground-truth class.  Encourages tight intra-class
    clustering in embedding space.

    L_proto = mean_over_batch( ||embedding - prototype[y]||^2 )
    """

    def forward(
        self,
        embedding: torch.Tensor,
        targets:   torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embedding:  (B, D)  batch embeddings
            targets:    (B,)    class indices
            prototypes: (C, D)  class prototypes from PrototypeMemoryModule
        Returns:
            scalar loss
        """

        target_protos = prototypes[targets]

        loss = (embedding - target_protos).pow(2).sum(dim=1).mean()
        return loss

class HybridLoss(nn.Module):
    """
    Combined training loss:

        L = L_cls + lambda_prototype * L_prototype

    where L_cls is either CrossEntropyLoss or FocalLoss.

    Args:
        lambda_prototype: Weight for PrototypeCompactnessLoss.
        use_focal:        Use FocalLoss instead of CrossEntropyLoss.
        focal_gamma:      Gamma for FocalLoss.
        focal_alpha:      Alpha for FocalLoss.
        label_smoothing:  Label smoothing for CE (0 = disabled).
    """

    def __init__(
        self,
        lambda_prototype: float = 0.5,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 1.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        self.lambda_prototype = lambda_prototype

        if use_focal:
            self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.proto_loss = PrototypeCompactnessLoss()

    def forward(
        self,
        logits:     torch.Tensor,
        embedding:  torch.Tensor,
        targets:    torch.Tensor,
        prototypes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            logits:     (B, C)
            embedding:  (B, D)
            targets:    (B,)
            prototypes: (C, D)

        Returns:
            dict with keys: 'total', 'cls', 'prototype'
        """
        l_cls   = self.cls_loss(logits, targets)
        l_proto = self.proto_loss(embedding, targets, prototypes)
        total   = l_cls + self.lambda_prototype * l_proto

        return {
            "total":     total,
            "cls":       l_cls,
            "prototype": l_proto,
        }

    @staticmethod
    def from_config(cfg: dict) -> "HybridLoss":
        t = cfg["training"]
        return HybridLoss(
            lambda_prototype=t["lambda_prototype"],
            use_focal=t["use_focal_loss"],
            focal_gamma=t["focal_gamma"],
            focal_alpha=t["focal_alpha"],
        )
