"""Plain prototype cosine threshold baseline: same embedding space as LRMC
(nearest-prototype assignment) but ONE global distance threshold instead of
learnable per-class radii -- the most direct "what does the radius buy you"
comparison."""

from __future__ import annotations

import numpy as np

from lrmc.baselines.base import PerClassStatBaseline


class PrototypeCosineBaseline(PerClassStatBaseline):
    def _fit_embeddings(self, z: np.ndarray, y: np.ndarray) -> None:
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        num_classes = len(self.families)
        dim = z.shape[1]
        prototypes = np.zeros((num_classes, dim))
        for c in range(num_classes):
            mask = y == c
            if mask.sum() == 0:
                continue
            m = z[mask].mean(axis=0)
            prototypes[c] = m / np.linalg.norm(m)
        self.prototypes = prototypes

    def _per_class_stat(self, z: np.ndarray) -> np.ndarray:
        zn = z / np.linalg.norm(z)
        return 1.0 - self.prototypes @ zn
