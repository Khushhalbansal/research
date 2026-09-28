import math

import torch
import torch.nn.functional as F

from lrmc.eval.calibration import calibrate_radii_from_quantile, compute_known_val_distances


def test_compute_known_val_distances_groups_by_family():
    torch.manual_seed(0)
    prototypes = F.normalize(torch.randn(2, 4), dim=1)
    z_val = F.normalize(torch.randn(6, 4), dim=1)
    y_val = torch.tensor([0, 0, 0, 1, 1, 1])
    families = ["fam_a", "fam_b"]
    result = compute_known_val_distances(z_val, y_val, prototypes, families)
    assert set(result.keys()) == {"fam_a", "fam_b"}
    assert len(result["fam_a"]) == 3
    assert len(result["fam_b"]) == 3


def test_calibrate_radii_quantile_matches_numpy():
    known_val_distances = {"fam_a": [0.1, 0.2, 0.3, 0.4, 0.5], "fam_b": [1.0, 2.0, 3.0]}
    radii = calibrate_radii_from_quantile(
        known_val_distances, target_quantile=0.9, families=["fam_a", "fam_b"]
    )
    import numpy as np

    assert abs(radii["fam_a"] - np.quantile(known_val_distances["fam_a"], 0.9)) < 1e-9
    assert abs(radii["fam_b"] - np.quantile(known_val_distances["fam_b"], 0.9)) < 1e-9


def test_calibrate_radii_missing_family_is_nan_not_zero():
    radii = calibrate_radii_from_quantile(
        {"fam_a": [0.1, 0.2]}, 0.5, families=["fam_a", "fam_missing"]
    )
    assert math.isnan(radii["fam_missing"])
