"""Post-hoc radius calibration (ablation): replace the learned r_c with a
per-class quantile of KNOWN-validation distances, never test or unknown data.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from lrmc.losses.lrmc_loss import gathered_distance


def compute_known_val_distances(
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    prototypes: torch.Tensor,
    families: list[str],
    metric: str = "cosine",
) -> dict[str, list[float]]:
    """z_val/y_val: embeddings + integer labels for the KNOWN validation split."""
    p_y = prototypes[y_val]
    d = gathered_distance(z_val, p_y, metric)
    out: dict[str, list[float]] = defaultdict(list)
    for i, lbl in enumerate(y_val.tolist()):
        out[families[lbl]].append(float(d[i].item()))
    return dict(out)


def calibrate_radii_from_quantile(
    known_val_distances: dict[str, list[float]], target_quantile: float, families: list[str]
) -> dict[str, float]:
    """Per-class quantile radius, computed ONLY from known-validation distances.

    A family missing from ``known_val_distances`` (e.g. none of its samples
    landed in the val split for this fold) keeps a NaN placeholder rather than
    silently defaulting to 0 -- callers must decide how to handle that rather
    than have it hidden.
    """
    out = {}
    for fam in families:
        dists = known_val_distances.get(fam, [])
        out[fam] = float(np.quantile(dists, target_quantile)) if dists else float("nan")
    return out
