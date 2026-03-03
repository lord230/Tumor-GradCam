"""
models/hybrid_model.py
=======================
HybridMemoryNet — the full research-level classification model.

Architecture:
    Input (B, 3, 224, 224)
        │
        ├── CNNBackbone (EfficientNet-B0) ──────► (B, 512)
        │
        └── TransformerBackbone (Swin-Tiny) ───► (B, 512)
                │
        ChannelAttentionFusion ─────────────────► (B, 512)  [embedding]
                │
        PrototypeMemoryModule ──────────────────► (B, 4)   [logits]

Forward output:
    {
        "logits"    : (B, num_classes),
        "embedding" : (B, embed_dim),
        "cnn_feat"  : (B, cnn_out_dim),
        "tr_feat"   : (B, transformer_out_dim),
    }
    prototypes are accessible via model.memory.prototypes
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.backbone_cnn        import CNNBackbone
from models.backbone_transformer import TransformerBackbone
from models.fusion_module        import ChannelAttentionFusion
from models.memory_module        import PrototypeMemoryModule

class HybridMemoryNet(nn.Module):
    """
    Hybrid CNN + Transformer network with Prototype Memory classification head.

    Args:
        num_classes:            Number of output classes (default 4).
        cnn_out_dim:            CNN backbone projected dim (default 512).
        transformer_out_dim:    Transformer backbone projected dim (default 512).
        fusion_embed_dim:       Fused embedding dim (default 512).
        memory_scale:           Cosine similarity temperature (default 10).
        freeze_cnn:             Freeze CNN backbone weights (default False).
        freeze_transformer:     Freeze Transformer backbone weights (default False).
        gradient_checkpointing: Enable gradient checkpointing in Swin blocks.
        pretrained:             Use ImageNet pretrained weights.
        dropout:                Dropout in fusion projection.
    """

    def __init__(
        self,
        num_classes: int = 4,
        cnn_out_dim: int = 512,
        transformer_out_dim: int = 512,
        fusion_embed_dim: int = 512,
        memory_scale: float = 10.0,
        freeze_cnn: bool = False,
        freeze_transformer: bool = False,
        gradient_checkpointing: bool = True,
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cnn = CNNBackbone(
            out_dim=cnn_out_dim,
            freeze=freeze_cnn,
            pretrained=pretrained,
        )

        self.transformer = TransformerBackbone(
            out_dim=transformer_out_dim,
            freeze=freeze_transformer,
            pretrained=pretrained,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.fusion = ChannelAttentionFusion(
            branch_dim=cnn_out_dim,
            embed_dim=fusion_embed_dim,
            dropout=dropout,
        )

        self.memory = PrototypeMemoryModule(
            num_classes=num_classes,
            embed_dim=fusion_embed_dim,
            scale=memory_scale,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            dict with keys:
                'logits'    : (B, num_classes)
                'embedding' : (B, fusion_embed_dim)
                'cnn_feat'  : (B, cnn_out_dim)
                'tr_feat'   : (B, transformer_out_dim)
        """
        cnn_feat = self.cnn(x)
        tr_feat  = self.transformer(x)
        embedding = self.fusion(cnn_feat, tr_feat)
        logits    = self.memory(embedding)

        return {
            "logits":    logits,
            "embedding": embedding,
            "cnn_feat":  cnn_feat,
            "tr_feat":   tr_feat,
        }

    def get_num_params(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def from_config(cfg: dict) -> "HybridMemoryNet":
        """
        Instantiate HybridMemoryNet from a parsed YAML config dict.

        Example:
            cfg = yaml.safe_load(open('configs/config.yaml'))
            model = HybridMemoryNet.from_config(cfg)
        """
        m = cfg["model"]
        t = cfg["training"]
        return HybridMemoryNet(
            num_classes=m["num_classes"],
            cnn_out_dim=m["cnn_out_dim"],
            transformer_out_dim=m["transformer_out_dim"],
            fusion_embed_dim=m["fusion_embed_dim"],
            memory_scale=m["memory_scale"],
            freeze_cnn=m["freeze_cnn"],
            freeze_transformer=m["freeze_transformer"],
            gradient_checkpointing=m["gradient_checkpointing"],
        )

if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("configs/config.yaml"))
    model = HybridMemoryNet.from_config(cfg)
    print(model)
    print(f"\nTotal trainable parameters: {model.get_num_params():,}")
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
