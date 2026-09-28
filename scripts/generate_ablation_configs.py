#!/usr/bin/env python
"""Generates every ablation config listed in the mission spec's ABLATIONS
section from configs/base_malimg.yaml, via dotted-path overrides. Re-run this
whenever the base config changes -- it always overwrites configs/ablation/*.

Usage: python scripts/generate_ablation_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lrmc.config import load_config, merge_overrides, save_config  # noqa: E402

BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base_malimg.yaml"
OUT_DIR = REPO_ROOT / "configs" / "ablation"

# name -> (dotted overrides dict, one-line rationale)
ABLATIONS: dict[str, tuple[dict, str]] = {
    # (a) remove each loss term
    "no_l_in": ({"loss.alpha": 0.0}, "drop compactness term L_in (alpha=0)"),
    "no_l_out": ({"loss.beta": 0.0}, "drop separation term L_out (beta=0)"),
    "no_l_rad": (
        {"loss.gamma": 0.0},
        "drop tightness term L_rad (gamma=0); radii then grow unchecked",
    ),
    # (b) learnable vs fixed radii is the LRMC-vs-FixedRadiusPrototypeBaseline
    # comparison (see baselines/fixed_radius_prototype.py), not a Config field.
    # (c) EMA vs batch-mean prototypes
    "prototypes_batch_mean": ({"prototypes.mode": "batch_mean"}, "no cross-batch prototype memory"),
    # (d) cosine vs Euclidean distance
    "distance_euclidean": (
        {"loss.distance_metric": "euclidean"},
        "squared-Euclidean instead of cosine distance",
    ),
    # (e) temperature sweep
    "temperature_0.05": ({"loss.temperature": 0.05}, "sharper SupCon temperature"),
    "temperature_0.2": ({"loss.temperature": 0.2}, "softer SupCon temperature"),
    "temperature_0.5": ({"loss.temperature": 0.5}, "much softer SupCon temperature"),
    # (f) margin sweep
    "margin_0.05": ({"loss.margin": 0.05}, "tighter separation margin"),
    "margin_0.2": ({"loss.margin": 0.2}, "looser separation margin"),
    "margin_0.3": ({"loss.margin": 0.3}, "much looser separation margin"),
    # (g) embedding dim sweep
    "embed_dim_64": ({"projection_head.out_dim": 64}, "smaller embedding space"),
    "embed_dim_256": ({"projection_head.out_dim": 256}, "larger embedding space"),
    # (h) gamma sweep -- also the direct empirical test of the quantile claim
    "gamma_0.05": ({"loss.gamma": 0.05}, "lower target false-rejection rate (~5%)"),
    "gamma_0.2": ({"loss.gamma": 0.2}, "higher target false-rejection rate (~20%)"),
    "gamma_0.3": ({"loss.gamma": 0.3}, "higher target false-rejection rate (~30%)"),
    "gamma_0.5": ({"loss.gamma": 0.5}, "very high target false-rejection rate (~50%)"),
    # (i) ViT vs ResNet18 backbone, pretrained vs scratch
    "backbone_resnet18": (
        {"backbone.name": "resnet18", "backbone.pretrained": True, "backbone.img_size": 224},
        "CNN ablation backbone, to justify choosing ViT",
    ),
    "backbone_scratch": ({"backbone.pretrained": False}, "no ImageNet pretraining"),
    # (j) group-aware vs naive holdout
    "split_naive_holdout": (
        {"split.protocol": "random_k_unknown", "split.group_aware": False},
        "naive per-family holdout: variant groups CAN be split, measuring the leakage gap",
    ),
    # (k) number of known classes
    "n_unknown_1": ({"split.n_unknown": 1}, "fewer unknown / more known classes"),
    "n_unknown_5": ({"split.n_unknown": 5}, "more unknown / fewer known classes"),
    "n_unknown_8": ({"split.n_unknown": 8}, "many unknown / few known classes (high openness)"),
}


# Ablations exist to show a qualitative effect, not to reproduce the main
# result's full training budget -- running all 23 at the main run's 20 epochs
# would blow well past Kaggle's 30 GPU-hours/week quota (see
# docs/DECISIONS.md and experiments/queue.yaml). 15 epochs is documented here
# as the deliberate, shared reduction for every Tier-B ablation.
ABLATION_EPOCHS = 15


def main() -> None:
    base = load_config(BASE_CONFIG_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (overrides, _rationale) in ABLATIONS.items():
        full_overrides = {"optim.epochs": ABLATION_EPOCHS, **overrides}
        cfg = merge_overrides(base, full_overrides)
        cfg.run_name = f"ablation_{name}"
        cfg.output_dir = f"runs/ablation_{name}"
        out_path = OUT_DIR / f"{name}.yaml"
        save_config(cfg, out_path)
        print(f"wrote {out_path}")
    print(f"\n{len(ABLATIONS)} ablation configs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
