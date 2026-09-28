"""Learnable Radii Parameters (diagram box: "Learnable Radii Parameters").

r_c = softplus(rho_c), rho_c a learnable nn.Parameter per known class,
initialized so r_c = r_init. Meant to live in its own optimizer param group
with its own (typically smaller) learning rate and zero weight decay -- see
``engine/train.py`` and docs/DECISIONS.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(r: float) -> float:
    # softplus(rho) = log(1 + exp(rho)) = r  =>  rho = log(exp(r) - 1)
    return math.log(math.expm1(r))


class LearnableRadiiParameters(nn.Module):
    def __init__(self, num_classes: int, r_init: float = 0.5):
        super().__init__()
        init_val = _inverse_softplus(r_init)
        self.rho = nn.Parameter(torch.full((num_classes,), float(init_val)))

    @property
    def radii(self) -> torch.Tensor:
        return F.softplus(self.rho)

    def forward(self, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        r = self.radii
        if class_ids is None:
            return r
        return r[class_ids]
