"""Class Prototype Estimator (diagram box: "Class Prototype Estimator").

p_c = mean of normalized class embeddings, re-normalized. Two update modes:

* ``ema`` (default): exponential moving average across batches with momentum
  ``m``, kept as a stop-gradient buffer (no grad flows into the prototype
  itself -- only through the per-sample distance terms in the loss).
* ``batch_mean``: prototype = this batch's class mean (no memory across
  batches).

A class is initialized lazily from the first batch that contains it; a class
absent from the current batch is left untouched (neither mode "forgets" a
class just because it didn't appear this step).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassPrototypeEstimator(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, mode: str = "ema", momentum: float = 0.9):
        super().__init__()
        if mode not in ("ema", "batch_mean"):
            raise ValueError(f"unknown prototype mode: {mode}")
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.mode = mode
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, embed_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """z: (B, D) normalized embeddings. y: (B,) int labels; entries < 0 (unknown
        sentinel) are ignored -- prototypes are only ever fit on known-class data."""
        z = z.detach()
        for c in torch.unique(y).tolist():
            if c < 0:
                continue
            mask = y == c
            if mask.sum() == 0:
                continue
            class_mean = F.normalize(z[mask].mean(dim=0), dim=0, eps=1e-12)
            if not bool(self.initialized[c]):
                self.prototypes[c] = class_mean
                self.initialized[c] = True
            elif self.mode == "ema":
                updated = self.momentum * self.prototypes[c] + (1 - self.momentum) * class_mean
                self.prototypes[c] = F.normalize(updated, dim=0, eps=1e-12)
            else:  # batch_mean
                self.prototypes[c] = class_mean

    def get(self, class_ids: torch.Tensor | None = None) -> torch.Tensor:
        if class_ids is None:
            return self.prototypes
        return self.prototypes[class_ids]

    def as_dict(self) -> dict[int, torch.Tensor]:
        return {
            c: self.prototypes[c].clone()
            for c in range(self.num_classes)
            if bool(self.initialized[c])
        }
