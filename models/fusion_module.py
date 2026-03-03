"""
models/fusion_module.py
========================
Channel Attention Fusion module.

Concatenates CNN and Transformer feature vectors (each `branch_dim` wide),
applies a Squeeze-and-Excitation style channel attention block,
then projects down to `embed_dim`.

Input : two tensors of shape (B, branch_dim)
Output: fused tensor of shape (B, embed_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn

class ChannelAttentionFusion(nn.Module):
    """
    SE-style Channel Attention over concatenated CNN + Transformer features.

    Architecture:
        [concat] → (B, 2*branch_dim)
        [SE gate] → element-wise reweighting
        [linear]  → (B, embed_dim)
        [LN + GELU]

    Args:
        branch_dim: Dimension of each backbone's output (e.g. 512).
        embed_dim:  Output embedding dimension (e.g. 512).
        reduction:  SE reduction ratio (default 16).
        dropout:    Dropout probability on the final embedding.
    """

    def __init__(
        self,
        branch_dim: int = 512,
        embed_dim: int = 512,
        reduction: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        concat_dim = branch_dim * 2

        bottleneck = max(concat_dim // reduction, 8)
        self.se = nn.Sequential(
            nn.Linear(concat_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, concat_dim),
            nn.Sigmoid(),
        )

        self.proj = nn.Sequential(
            nn.Linear(concat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        cnn_feat: torch.Tensor,
        tr_feat:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cnn_feat: (B, branch_dim) from CNNBackbone
            tr_feat:  (B, branch_dim) from TransformerBackbone
        Returns:
            embedding: (B, embed_dim)
        """
        x = torch.cat([cnn_feat, tr_feat], dim=1)
        gate = self.se(x)
        x = x * gate
        embedding = self.proj(x)
        return embedding

if __name__ == "__main__":
    fusion = ChannelAttentionFusion(branch_dim=512, embed_dim=512)
    cnn_feat = torch.randn(4, 512)
    tr_feat  = torch.randn(4, 512)
    out = fusion(cnn_feat, tr_feat)
    print(f"FusionModule output shape: {out.shape}")
