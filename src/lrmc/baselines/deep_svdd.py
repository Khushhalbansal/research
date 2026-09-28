"""Deep SVDD baseline (Ruff et al., 2018): jointly learns a feature map and a
SINGLE global hypersphere (center c, no per-class structure) by minimizing
mean ||phi(x) - c||^2 over all known-class training samples pooled together.

Simplification vs. the original paper (documented, not hidden): the original
architecture removes bias terms and bounded activations to block the trivial
all-zero collapse; this implementation uses the same ViTBackbone as everyone
else (for a fair, shared-preprocessing comparison) and instead relies on a
short warm-up (center fixed from an untrained forward pass, network then
trained from there) plus weight decay to discourage collapse. If a run's
radii/embeddings look degenerate (near-zero variance), that is a known risk
of this simplification -- see HANDOFF.md.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from lrmc.baselines.base import UNKNOWN_LABEL, BaselineDetector
from lrmc.config import BackboneConfig
from lrmc.models.backbone import ViTBackbone


class DeepSVDDBaseline(BaselineDetector):
    def __init__(
        self,
        backbone_cfg: BackboneConfig,
        label_map: dict[str, int],
        feature_dim: int = 32,
        device: str = "cpu",
        epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.label_map = label_map
        self.families = [f for f, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        self.backbone = ViTBackbone(backbone_cfg)
        self.proj = nn.Linear(self.backbone.feature_dim, feature_dim)
        self.device = torch.device(device)
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.center: torch.Tensor | None = None
        self.family_centroids: np.ndarray | None = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))

    def fit(self, train_loader) -> None:
        self.backbone.to(self.device)
        self.proj.to(self.device)
        params = list(self.backbone.parameters()) + list(self.proj.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        self.backbone.eval()
        self.proj.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for images, labels, _ids in train_loader:
                all_z.append(self._forward(images.to(self.device)))
                all_y.append(labels)
        self.center = torch.cat(all_z).mean(dim=0)

        self.backbone.train()
        self.proj.train()
        for _epoch in range(self.epochs):
            for images, labels, _ids in train_loader:
                optimizer.zero_grad()
                z = self._forward(images.to(self.device))
                loss = ((z - self.center) ** 2).sum(dim=1).mean()
                loss.backward()
                optimizer.step()

        self.backbone.eval()
        self.proj.eval()
        with torch.no_grad():
            all_z2, all_y2 = [], []
            for images, labels, _ids in train_loader:
                all_z2.append(self._forward(images.to(self.device)))
                all_y2.append(labels)
            z_final = torch.cat(all_z2).numpy()
            y_final = torch.cat(all_y2).numpy()
        num_classes = len(self.families)
        centroids = np.zeros((num_classes, z_final.shape[1]))
        for c in range(num_classes):
            mask = y_final == c
            if mask.sum() > 0:
                centroids[c] = z_final[mask].mean(axis=0)
        self.family_centroids = centroids

    @torch.no_grad()
    def _distance(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        z = self._forward(x.to(self.device))[0]
        d = ((z - self.center) ** 2).sum().item()
        return d, z.numpy()

    def score(self, x: torch.Tensor) -> float:
        d, _z = self._distance(x)
        return d

    def predict(self, x: torch.Tensor) -> str:
        d, z = self._distance(x)
        if self.kappa is not None and d > self.kappa:
            return UNKNOWN_LABEL
        dists = np.linalg.norm(self.family_centroids - z, axis=1)
        return self.families[int(dists.argmin())]
