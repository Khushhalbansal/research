"""One-Class SVM baseline (Tax & Duin, 2004 lineage) on embeddings: a single
global boundary around all known-class embeddings pooled together, with
family attribution via nearest prototype among knowns when accepted."""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from lrmc.baselines.base import UNKNOWN_LABEL, EmbeddingBaseline


class OCSVMBaseline(EmbeddingBaseline):
    def __init__(self, encoder, label_map: dict[str, int], nu: float = 0.05, kernel: str = "rbf", gamma="scale"):
        super().__init__(encoder, label_map)
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def _fit_embeddings(self, z: np.ndarray, y: np.ndarray) -> None:
        zn = z / np.linalg.norm(z, axis=1, keepdims=True)
        self.model.fit(zn)
        num_classes = len(self.families)
        prototypes = np.zeros((num_classes, z.shape[1]))
        for c in range(num_classes):
            mask = y == c
            if mask.sum() > 0:
                m = zn[mask].mean(axis=0)
                prototypes[c] = m / np.linalg.norm(m)
        self.prototypes = prototypes

    def _score_embedding(self, z: np.ndarray) -> float:
        zn = z / np.linalg.norm(z)
        return float(-self.model.decision_function(zn.reshape(1, -1))[0])

    def _predict_embedding(self, z: np.ndarray) -> str:
        zn = z / np.linalg.norm(z)
        is_inlier = self.model.predict(zn.reshape(1, -1))[0] == 1
        if not is_inlier:
            return UNKNOWN_LABEL
        d = 1.0 - self.prototypes @ zn
        return self.families[int(d.argmin())]

    def calibrate(self, val_loader, target_known_tpr: float = 0.95) -> None:
        # OCSVM's boundary is already fixed by nu at fit() time; nothing to
        # calibrate on val data, kept as a documented no-op for interface parity.
        return None
