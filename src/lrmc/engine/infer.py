"""Frozen inference: Distance Calculator + Zero Day Classifier (diagram boxes
inside the "Radius Based Decision Engine"), built on the same Fixed Feature
Extraction Network used at training time, now frozen (``eval()``, ``no_grad``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from lrmc.config import Config
from lrmc.engine.checkpoint import load_checkpoint
from lrmc.models.network import FeatureExtractionNetwork


class DistanceCalculator:
    """Computes d(z, p_c) for every known family, given Stored Class Prototypes."""

    def __init__(self, prototypes: torch.Tensor, families: list[str], metric: str = "cosine"):
        self.prototypes = prototypes  # (K, D)
        self.families = families  # index-aligned with prototypes rows
        self.metric = metric

    def compute(self, z: torch.Tensor) -> torch.Tensor:
        """z: (D,) or (B, D) -> distances (K,) or (B, K)."""
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        if self.metric == "cosine":
            d = 1.0 - z @ self.prototypes.T
        else:
            diff = z.unsqueeze(1) - self.prototypes.unsqueeze(0)
            d = 0.5 * (diff**2).sum(dim=-1)
        return d.squeeze(0) if single else d


class ZeroDayClassifier:
    """Applies the radius-ratio decision rule: s(x) = min_c d(z,p_c)/r_c."""

    def __init__(
        self,
        radii: torch.Tensor,
        families: list[str],
        kappa: float = 1.0,
        score_mode: str = "ratio",
    ):
        self.radii = radii  # (K,)
        self.families = families
        self.kappa = kappa
        self.score_mode = score_mode

    def classify(self, distances: torch.Tensor) -> dict:
        """distances: (K,) for one sample. Returns verdict dict."""
        ratios = distances / self.radii
        if self.score_mode == "ratio":
            best_idx = torch.argmin(ratios).item()
            score = ratios[best_idx].item()
        else:  # "min_distance" -- ignores radii, for AUROC comparisons
            best_idx = torch.argmin(distances).item()
            score = distances[best_idx].item()

        verdict = "KNOWN" if score <= self.kappa else "ZERO_DAY"
        return {
            "verdict": verdict,
            "nearest_family": self.families[best_idx],
            "score": score,
            "distances": {f: distances[i].item() for i, f in enumerate(self.families)},
            "ratios": {f: ratios[i].item() for i, f in enumerate(self.families)},
        }


@dataclass
class InferenceOutput:
    verdict: str
    nearest_family: str
    score: float
    distances: dict[str, float]
    ratios: dict[str, float]


def _model_hash(cfg: Config, network_state: dict) -> str:
    from lrmc.config import config_hash

    h = hashlib.sha256()
    h.update(config_hash(cfg).encode())
    for k in sorted(network_state.keys()):
        t = network_state[k]
        h.update(k.encode())
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class FrozenInferenceEngine:
    """Loads a trained checkpoint and serves frozen predictions: Distance
    Calculator -> Zero Day Classifier. This is the "Fixed Feature Extraction
    Network" + "Radius Based Decision Engine" half of the inference diagram."""

    def __init__(self, cfg: Config, checkpoint_path: str | Path):
        self.cfg = cfg
        ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
        self.label_map: dict[str, int] = ckpt["label_map"]
        self.families = [f for f, _ in sorted(self.label_map.items(), key=lambda kv: kv[1])]

        self.network = FeatureExtractionNetwork(cfg.backbone, cfg.projection_head)
        self.network.load_state_dict(ckpt["network_state"])
        self.network.eval()
        for p in self.network.parameters():
            p.requires_grad_(False)

        prototypes = ckpt["prototypes_state"]["prototypes"]  # (K, D)
        radii = torch.nn.functional.softplus(ckpt["radii_state"]["rho"])  # (K,)

        self.distance_calc = DistanceCalculator(
            prototypes, self.families, metric=cfg.loss.distance_metric
        )
        self.classifier = ZeroDayClassifier(
            radii, self.families, kappa=cfg.inference.kappa, score_mode=cfg.inference.score
        )
        self.model_hash = _model_hash(cfg, ckpt["network_state"])

    @torch.no_grad()
    def embed(self, image: torch.Tensor) -> torch.Tensor:
        """image: (1, H, W) or (B, 1, H, W) -> embedding(s)."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.network(image)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> InferenceOutput:
        z = self.embed(image).squeeze(0)
        distances = self.distance_calc.compute(z)
        result = self.classifier.classify(distances)
        return InferenceOutput(**result)
