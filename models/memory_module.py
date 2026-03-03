"""
models/memory_module.py
========================
Prototype Memory Module.

Maintains a set of learnable class prototypes (one per class).
At inference, computes cosine similarity between the input embedding
and all prototypes, then scales by a temperature to produce logits.

This encourages the model to cluster embeddings near their class prototype,
which is regularized further by PrototypeCompactnessLoss in losses.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeMemoryModule(nn.Module):
    """
    Learnable class prototypes with cosine similarity classification head.

    Args:
        num_classes: Number of output classes.
        embed_dim:   Dimension of input embeddings.
        scale:       Temperature scale applied to cosine similarities (default 10).
    """

    def __init__(
        self,
        num_classes: int = 4,
        embed_dim: int = 512,
        scale: float = 10.0,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.scale       = scale

        self.prototypes = nn.Parameter(
            torch.randn(num_classes, embed_dim)
        )

        nn.init.xavier_uniform_(self.prototypes)

    @property
    def normalised_prototypes(self) -> torch.Tensor:
        """L2-normalised prototypes; shape (num_classes, embed_dim)."""
        return F.normalize(self.prototypes, p=2, dim=1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: (B, embed_dim)  — L2-normalised inside this function
        Returns:
            logits: (B, num_classes)   — scaled cosine similarities
        """

        emb_norm = F.normalize(embedding, p=2, dim=1)
        proto_norm = self.normalised_prototypes

        cosine_sim = emb_norm @ proto_norm.t()
        logits = self.scale * cosine_sim
        return logits

    def get_prototype_distances(
        self, embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns L2 distance from each embedding to each prototype.
        Used by PrototypeCompactnessLoss.

        Args:
            embedding: (B, embed_dim)
        Returns:
            distances: (B, num_classes)
        """

        diffs = embedding.unsqueeze(1) - self.prototypes.unsqueeze(0)
        distances = diffs.pow(2).sum(dim=2)
        return distances

if __name__ == "__main__":
    mem = PrototypeMemoryModule(num_classes=4, embed_dim=512, scale=10.0)
    emb = torch.randn(8, 512)
    logits = mem(emb)
    print(f"MemoryModule logits shape : {logits.shape}")
    dist   = mem.get_prototype_distances(emb)
    print(f"MemoryModule dist   shape : {dist.shape}")
