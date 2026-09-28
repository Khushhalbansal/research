"""Projection Head (diagram box: "Projection Head"): 2-layer MLP h -> z-tilde."""

from __future__ import annotations

import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h):
        return self.net(h)
