"""Mahalanobis-distance baseline (Lee et al., 2018 style): per-class mean,
tied (shared) covariance estimated from within-class residuals pooled across
all known classes."""

from __future__ import annotations

import numpy as np

from lrmc.baselines.base import PerClassStatBaseline


class MahalanobisBaseline(PerClassStatBaseline):
    def _fit_embeddings(self, z: np.ndarray, y: np.ndarray) -> None:
        num_classes = len(self.families)
        dim = z.shape[1]
        means = np.zeros((num_classes, dim))
        residuals = []
        for c in range(num_classes):
            mask = y == c
            if mask.sum() == 0:
                continue
            means[c] = z[mask].mean(axis=0)
            residuals.append(z[mask] - means[c])
        residuals = np.concatenate(residuals, axis=0)
        cov = np.cov(residuals, rowvar=False) + 1e-6 * np.eye(dim)
        self.means = means
        self.precision = np.linalg.inv(cov)

    def _per_class_stat(self, z: np.ndarray) -> np.ndarray:
        diff = self.means - z  # (K, D)
        return np.einsum("kd,de,ke->k", diff, self.precision, diff)
