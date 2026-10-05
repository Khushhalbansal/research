"""SupCon + prototype + FIXED radius baseline (percentile-based): the key
ablation for learnable radii. Same prototype geometry as LRMC, but each
class's radius is a fixed per-class quantile of KNOWN-validation... here,
fit purely from the KNOWN-training distances (no learning, no gradient), so
it isolates the effect of making radii *learnable* rather than *fixed*.
"""

from __future__ import annotations

import numpy as np

from lrmc.baselines.base import UNKNOWN_LABEL, PerClassStatBaseline


class FixedRadiusPrototypeBaseline(PerClassStatBaseline):
    def __init__(self, encoder, label_map, quantile: float = 0.9):
        super().__init__(encoder, label_map)
        self.quantile = quantile

    def _fit_embeddings(self, z: np.ndarray, y: np.ndarray) -> None:
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        num_classes = len(self.families)
        dim = z.shape[1]
        prototypes = np.zeros((num_classes, dim))
        radii = np.full(num_classes, np.nan)
        for c in range(num_classes):
            mask = y == c
            if mask.sum() == 0:
                continue
            m = z[mask].mean(axis=0)
            m = m / np.linalg.norm(m)
            prototypes[c] = m
            d = 1.0 - z[mask] @ m
            # Near-duplicate classes give a ~0 quantile; floor it so d / r stays finite.
            radii[c] = max(float(np.quantile(d, self.quantile)), 1e-6)
        self.prototypes = prototypes
        self.radii = radii

    def _per_class_stat(self, z: np.ndarray) -> np.ndarray:
        zn = z / np.linalg.norm(z)
        d = 1.0 - self.prototypes @ zn
        return d / self.radii

    def _predict_embedding(self, z: np.ndarray) -> str:
        # Fixed radii already define a per-class acceptance boundary (ratio <=
        # 1); kappa (if calibrated) overrides that default 1.0 operating point
        # for a fair comparison against LRMC's own tuned kappa.
        stat = self._per_class_stat(z)
        best = int(np.nanargmin(stat))
        threshold = self.kappa if self.kappa is not None else 1.0
        if stat[best] > threshold:
            return UNKNOWN_LABEL
        return self.families[best]
