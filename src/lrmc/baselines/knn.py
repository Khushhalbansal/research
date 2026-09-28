"""kNN-distance baseline (Sun et al., 2022 style): unknown-score is the cosine
distance to the k-th nearest KNOWN-training embedding (pooled across all
classes); family attribution is majority vote among the k nearest neighbors.
"""

from __future__ import annotations

import numpy as np

from lrmc.baselines.base import UNKNOWN_LABEL, EmbeddingBaseline, majority_vote


class KNNBaseline(EmbeddingBaseline):
    def __init__(self, encoder, label_map: dict[str, int], k: int = 5):
        super().__init__(encoder, label_map)
        self.k = k

    def _fit_embeddings(self, z: np.ndarray, y: np.ndarray) -> None:
        self.train_z = z / np.linalg.norm(z, axis=1, keepdims=True)
        self.train_y = y

    def _neighbors(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zn = z / np.linalg.norm(z)
        d = 1.0 - self.train_z @ zn
        k = min(self.k, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        idx = idx[np.argsort(d[idx])]
        return d, idx

    def _score_embedding(self, z: np.ndarray) -> float:
        d, idx = self._neighbors(z)
        return float(d[idx[-1]])

    def _predict_embedding(self, z: np.ndarray) -> str:
        d, idx = self._neighbors(z)
        score = float(d[idx[-1]])
        if self.kappa is not None and score > self.kappa:
            return UNKNOWN_LABEL
        best_label = majority_vote(self.train_y[idx])
        return self.families[best_label]

    def calibrate(self, val_loader, target_known_tpr: float = 0.95) -> None:
        z, _y = self._embed_loader(val_loader)
        scores = np.array([self._score_embedding(zi) for zi in z.numpy()])
        self.kappa = float(np.quantile(scores, target_known_tpr))

    def nearest_family(self, x) -> str:
        z = self._embed_one(x)
        _d, idx = self._neighbors(z)
        return self.families[majority_vote(self.train_y[idx])]
