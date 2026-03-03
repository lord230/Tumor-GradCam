"""
models/backbone_transformer.py
================================
Swin-Tiny Transformer backbone.

Loads pretrained Swin-Tiny via `timm`, removes the classification head,
and projects the 768-dim mean-pooled feature to `out_dim` (default 512).

Gradient checkpointing can be enabled to reduce VRAM usage on RTX 3060.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for the Transformer backbone. "
        "Install via: pip install timm"
    ) from e

class TransformerBackbone(nn.Module):
    """
    Swin-Tiny Transformer feature extractor.

    Args:
        out_dim:               Projected output dimension (default 512).
        freeze:                If True, freeze all Swin weights.
        pretrained:            Load ImageNet-1k weights (default True).
        gradient_checkpointing: Enable gradient checkpointing in Swin blocks.
    """

    _SWIN_TINY_DIM: int = 768

    def __init__(
        self,
        out_dim: int = 512,
        freeze: bool = False,
        pretrained: bool = True,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        self.proj = nn.Sequential(
            nn.Linear(self._SWIN_TINY_DIM, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        if freeze:
            self._freeze_backbone()

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on all Swin BasicLayer blocks."""
        for module in self.backbone.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone weights."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feat: (B, out_dim)
        """
        x = self.backbone(x)
        x = self.proj(x)
        return x

if __name__ == "__main__":
    model = TransformerBackbone(out_dim=512, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"TransformerBackbone output shape: {out.shape}")
