"""Energy-score baseline (Liu et al., 2020): unknown-score = -logsumexp(logits
/ T), i.e. the negative free energy of the softmax distribution. Chosen over
literal ODIN (which additionally perturbs the INPUT via its own gradient,
adding a second forward+backward pass and a temperature/epsilon grid search)
as the more robust and cheaper of the two "ODIN-style or energy score"
options the mission allows; temperature T is still exposed for an ODIN-style
temperature-scaling ablation.
"""

from __future__ import annotations

import torch

from lrmc.baselines._softmax_net import train_softmax_net
from lrmc.baselines.base import UNKNOWN_LABEL, BaselineDetector
from lrmc.config import BackboneConfig


class EnergyBaseline(BaselineDetector):
    def __init__(
        self,
        backbone_cfg: BackboneConfig,
        label_map: dict[str, int],
        device: str = "cpu",
        epochs: int = 5,
        lr: float = 1e-3,
        temperature: float = 1.0,
    ):
        self.backbone_cfg = backbone_cfg
        self.label_map = label_map
        self.families = [f for f, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        self.device = torch.device(device)
        self.epochs = epochs
        self.lr = lr
        self.temperature = temperature
        self.net = None

    def fit(self, train_loader) -> None:
        self.net = train_softmax_net(
            self.backbone_cfg, len(self.families), train_loader, self.device, self.epochs, self.lr
        )

    @torch.no_grad()
    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        logits, _feat = self.net(x.to(self.device))
        return logits[0]

    def score(self, x: torch.Tensor) -> float:
        logits = self._logits(x)
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=0)
        return float(energy.item())  # higher energy = more OOD

    def predict(self, x: torch.Tensor) -> str:
        logits = self._logits(x)
        energy = float(
            (-self.temperature * torch.logsumexp(logits / self.temperature, dim=0)).item()
        )
        best = int(logits.argmax().item())
        if self.kappa is not None and energy > self.kappa:
            return UNKNOWN_LABEL
        return self.families[best]
