"""
models/backbone_cnn.py
======================
EfficientNet-B0 CNN backbone.

Loads a pretrained EfficientNet-B0, strips the classifier head,
and projects the 1280-dim pooled feature vector to `out_dim` (default 512).

Optionally supports gradient checkpointing (unused here – the transformer
backbone is the memory-intensive one, but left as an interface).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class CNNBackbone(nn.Module):
    """
    EfficientNet-B0 feature extractor.

    Args:
        out_dim:  Projected output dimension (default 512).
        freeze:   If True, freeze all EfficientNet weights.
        pretrained: Load ImageNet weights (default True).
    """

    def __init__(
        self,
        out_dim: int = 512,
        freeze: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.pool     = backbone.avgpool

        self._cnn_dim = backbone.classifier[1].in_features

        self.proj = nn.Sequential(
            nn.Linear(self._cnn_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        if freeze:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone weights (call after warm-up)."""
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feat: (B, out_dim)
        """
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x

if __name__ == "__main__":
    model = CNNBackbone(out_dim=512)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"CNNBackbone output shape: {out.shape}")
