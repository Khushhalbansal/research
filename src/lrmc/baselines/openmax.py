"""OpenMax (Bendale & Boult, 2016), implemented from the original algorithm:

1. Train a softmax classifier (own copy, via ``_softmax_net``).
2. For each known class, compute the Mean Activation Vector (MAV) -- the mean
   penultimate-layer feature over CORRECTLY-classified training samples of
   that class.
3. Take that class's ``tail_size`` largest distances-to-MAV among its correctly
   classified training samples and fit a Weibull distribution to the tail via
   ``scipy.stats.weibull_min.fit`` (a per-class "meta-recognition" model of how
   far a genuine member of the class can plausibly sit from its MAV).
4. At inference, for the ``alpha`` highest-logit classes, revise each logit
   down by its Weibull CDF value (how unusually far this sample's activation
   is from that class's MAV) and route the removed mass into a synthetic
   "unknown" logit; softmax over [revised logits, unknown logit] gives OpenMax
   probabilities.

Simplification vs. the original paper (documented, not hidden): distance-to-MAV
uses cosine distance only (the paper uses a Euclidean+cosine hybrid) and omits
the rank-based alpha-weighting of the Weibull scores -- both are reasonable,
commonly-used simplifications that preserve the core recalibration mechanism.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import weibull_min

from lrmc.baselines._softmax_net import train_softmax_net
from lrmc.baselines.base import UNKNOWN_LABEL, BaselineDetector
from lrmc.config import BackboneConfig


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class OpenMaxBaseline(BaselineDetector):
    def __init__(
        self,
        backbone_cfg: BackboneConfig,
        label_map: dict[str, int],
        device: str = "cpu",
        epochs: int = 5,
        lr: float = 1e-3,
        tail_size: int = 10,
        alpha: int | None = None,
    ):
        self.backbone_cfg = backbone_cfg
        self.label_map = label_map
        self.families = [f for f, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        self.device = torch.device(device)
        self.epochs = epochs
        self.lr = lr
        self.tail_size = tail_size
        self.alpha = alpha or min(3, len(self.families))
        self.net = None
        self.mavs: np.ndarray | None = None
        self.weibull_params: list[tuple[float, float, float] | None] = []

    def fit(self, train_loader) -> None:
        num_classes = len(self.families)
        self.net = train_softmax_net(
            self.backbone_cfg, num_classes, train_loader, self.device, self.epochs, self.lr
        )
        feats_by_class: list[list[np.ndarray]] = [[] for _ in range(num_classes)]
        with torch.no_grad():
            for images, labels, _ids in train_loader:
                logits, feat = self.net(images.to(self.device))
                preds = logits.argmax(dim=1)
                for i in range(images.shape[0]):
                    if int(preds[i]) == int(labels[i]):
                        feats_by_class[int(labels[i])].append(feat[i].numpy())

        dim = self.net.backbone.feature_dim
        mavs = np.zeros((num_classes, dim))
        self.weibull_params = [None] * num_classes
        for c in range(num_classes):
            feats = feats_by_class[c]
            if not feats:
                continue
            feats_arr = np.stack(feats)
            mav = feats_arr.mean(axis=0)
            mavs[c] = mav
            distances = np.array([_cosine_distance(f, mav) for f in feats_arr])
            tail = np.sort(distances)[-min(self.tail_size, len(distances)) :]
            if len(tail) >= 3 and tail.std() > 1e-9:
                try:
                    shape, loc, scale = weibull_min.fit(tail, floc=0)
                    self.weibull_params[c] = (shape, loc, scale)
                except Exception:  # noqa: BLE001 - fail-soft: no revision for this class
                    self.weibull_params[c] = None
        self.mavs = mavs

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        logits, feat = self.net(x.to(self.device))
        return logits[0].numpy(), feat[0].numpy()

    def _openmax_probs(self, x: torch.Tensor) -> np.ndarray:
        logits, feat = self._forward(x)
        ranked = np.argsort(-logits)[: self.alpha]
        revised = logits.copy()
        unknown_mass = 0.0
        for c in ranked:
            params = self.weibull_params[c]
            if params is None:
                continue
            dist = _cosine_distance(feat, self.mavs[c])
            shape, loc, scale = params
            w_score = float(weibull_min.cdf(dist, shape, loc=loc, scale=scale))
            w_score = min(max(w_score, 0.0), 1.0)
            revised[c] = logits[c] * (1.0 - w_score)
            unknown_mass += logits[c] * w_score
        all_logits = np.concatenate([revised, [unknown_mass]])
        exp = np.exp(all_logits - all_logits.max())
        probs = exp / exp.sum()
        return probs

    def score(self, x: torch.Tensor) -> float:
        probs = self._openmax_probs(x)
        return float(probs[-1])  # P(unknown)

    def predict(self, x: torch.Tensor) -> str:
        probs = self._openmax_probs(x)
        best = int(probs.argmax())
        threshold = self.kappa
        if best == len(self.families):
            return UNKNOWN_LABEL
        if threshold is not None and probs[-1] > threshold:
            return UNKNOWN_LABEL
        return self.families[best]

    def nearest_family(self, x: torch.Tensor) -> str:
        logits, _feat = self._forward(x)
        return self.families[int(np.argmax(logits))]
