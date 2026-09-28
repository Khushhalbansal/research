"""Shared baseline interface: ``fit(train_loader)``, ``score(x) -> unknown-score``
(higher = more anomalous/unknown), ``predict(x) -> family name | "UNKNOWN"``.

``train_loader`` yields ``(images, labels, sample_ids)`` batches from an
:class:`~lrmc.data.datasets.EvalMalwareDataset` over the KNOWN training split
only (no test-set or unknown-family leakage). Every baseline here shares the
same data pipeline and metrics as LRMC for a fair comparison.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn

UNKNOWN_LABEL = "UNKNOWN"


class BaselineDetector:
    families: list[str]
    kappa: float | None = None

    def fit(self, train_loader) -> None:
        raise NotImplementedError

    def score(self, x: torch.Tensor) -> float:
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> str:
        raise NotImplementedError

    def calibrate(self, val_loader, target_known_tpr: float = 0.95) -> None:
        """Default: threshold = the target_known_tpr quantile of KNOWN-val
        scores (so that quantile-fraction of known val samples are accepted).
        A no-op for baselines whose boundary is fixed by their own fit()
        (e.g. OCSVM's nu)."""
        scores = []
        for images, _labels, _ids in val_loader:
            for i in range(images.shape[0]):
                scores.append(self.score(images[i]))
        if scores:
            self.kappa = float(np.quantile(scores, target_known_tpr))


class EmbeddingBaseline(BaselineDetector):
    """Base for baselines that operate on embeddings from a shared, frozen
    encoder (fair comparison: same backbone/preprocessing as LRMC)."""

    def __init__(self, encoder: nn.Module, label_map: dict[str, int]):
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.label_map = label_map
        self.families = [f for f, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    @torch.no_grad()
    def _embed_loader(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
        zs, ys = [], []
        for images, labels, _ids in loader:
            zs.append(self.encoder(images))
            ys.append(labels)
        return torch.cat(zs), torch.cat(ys)

    @torch.no_grad()
    def _embed_one(self, x: torch.Tensor) -> np.ndarray:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.encoder(x).squeeze(0).numpy()

    def fit(self, train_loader) -> None:
        z, y = self._embed_loader(train_loader)
        self._fit_embeddings(z.numpy(), y.numpy())

    def score(self, x: torch.Tensor) -> float:
        return self._score_embedding(self._embed_one(x))

    def predict(self, x: torch.Tensor) -> str:
        return self._predict_embedding(self._embed_one(x))

    def _fit_embeddings(self, z: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def _score_embedding(self, z: np.ndarray) -> float:
        raise NotImplementedError

    def _predict_embedding(self, z: np.ndarray) -> str:
        raise NotImplementedError


class PerClassStatBaseline(EmbeddingBaseline):
    """Baselines whose score is min_c stat_c(z) and whose family prediction is
    argmin_c stat_c(z), rejected as UNKNOWN if that minimum exceeds kappa."""

    def _per_class_stat(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _score_embedding(self, z: np.ndarray) -> float:
        return float(np.nanmin(self._per_class_stat(z)))

    def _predict_embedding(self, z: np.ndarray) -> str:
        stat = self._per_class_stat(z)
        best = int(np.nanargmin(stat))
        if self.kappa is not None and stat[best] > self.kappa:
            return UNKNOWN_LABEL
        return self.families[best]

    def calibrate(self, val_loader, target_known_tpr: float = 0.95) -> None:
        z, _y = self._embed_loader(val_loader)
        scores = np.array([self._score_embedding(zi) for zi in z.numpy()])
        self.kappa = float(np.quantile(scores, target_known_tpr))


def majority_vote(labels: np.ndarray) -> int:
    return Counter(labels.tolist()).most_common(1)[0][0]
