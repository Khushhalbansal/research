"""Embedding Normalizer (diagram box: "Embedding Normalizer").

z = normalize(z_tilde) so every embedding lives on the unit hypersphere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class EmbeddingNormalizer(torch.nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, z_tilde: torch.Tensor) -> torch.Tensor:
        return F.normalize(z_tilde, p=2, dim=-1, eps=self.eps)
