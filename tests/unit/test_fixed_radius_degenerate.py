import numpy as np
import torch.nn as nn

from lrmc.baselines.fixed_radius_prototype import FixedRadiusPrototypeBaseline


def test_identical_embeddings_give_finite_scores():
    b = FixedRadiusPrototypeBaseline(nn.Linear(2, 2), {"A": 0, "B": 1})
    z = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
    y = np.array([0] * 5 + [1] * 5)
    b._fit_embeddings(z, y)
    stat = b._per_class_stat(np.array([0.6, 0.8]))
    assert np.all(np.isfinite(stat))
    assert np.all(b.radii >= 1e-6)
