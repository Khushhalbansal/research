"""Softmax cross-entropy classifier with Maximum Softmax Probability (MSP)
thresholding (Hendrycks & Gimpel, 2017 lineage): the simplest closed-set
baseline, unknown-score = 1 - max_c softmax(logits)_c."""

from __future__ import annotations

import torch

from lrmc.baselines._softmax_net import train_softmax_net
from lrmc.baselines.base import UNKNOWN_LABEL, BaselineDetector
from lrmc.config import BackboneConfig


class SoftmaxMSPBaseline(BaselineDetector):
    def __init__(
        self,
        backbone_cfg: BackboneConfig,
        label_map: dict[str, int],
        device: str = "cpu",
        epochs: int = 5,
        lr: float = 1e-3,
    ):
        self.backbone_cfg = backbone_cfg
        self.label_map = label_map
        self.families = [f for f, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        self.device = torch.device(device)
        self.epochs = epochs
        self.lr = lr
        self.net = None

    def fit(self, train_loader) -> None:
        self.net = train_softmax_net(
            self.backbone_cfg, len(self.families), train_loader, self.device, self.epochs, self.lr
        )

    @torch.no_grad()
    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        logits, _feat = self.net(x.to(self.device))
        return torch.softmax(logits[0], dim=0)

    def score(self, x: torch.Tensor) -> float:
        probs = self._probs(x)
        return float(1.0 - probs.max().item())

    def predict(self, x: torch.Tensor) -> str:
        probs = self._probs(x)
        score = float(1.0 - probs.max().item())
        best = int(probs.argmax().item())
        if self.kappa is not None and score > self.kappa:
            return UNKNOWN_LABEL
        return self.families[best]
