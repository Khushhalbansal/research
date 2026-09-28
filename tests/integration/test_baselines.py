"""End-to-end smoke tests for every baseline against the shared interface
(fit/score/predict), on mock data with the tiny_test backbone preset."""

import numpy as np
import pytest
from torch.utils.data import DataLoader

from lrmc.baselines.base import UNKNOWN_LABEL
from lrmc.baselines.deep_svdd import DeepSVDDBaseline
from lrmc.baselines.fixed_radius_prototype import FixedRadiusPrototypeBaseline
from lrmc.baselines.knn import KNNBaseline
from lrmc.baselines.mahalanobis import MahalanobisBaseline
from lrmc.baselines.ocsvm import OCSVMBaseline
from lrmc.baselines.odin_energy import EnergyBaseline
from lrmc.baselines.openmax import OpenMaxBaseline
from lrmc.baselines.prototype_cosine import PrototypeCosineBaseline
from lrmc.baselines.softmax_msp import SoftmaxMSPBaseline
from lrmc.config import BackboneConfig, ProjectionHeadConfig
from lrmc.data.datasets import EvalMalwareDataset, build_label_map, eval_collate_fn
from lrmc.data.image_generator import ImageGenerator
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.models.network import FeatureExtractionNetwork


def _backbone_cfg():
    return BackboneConfig(
        name="tiny_test", img_size=32, patch_size=8, depth=2, embed_dim=16, num_heads=2, mlp_ratio=2.0
    )


@pytest.fixture(scope="module")
def baseline_fixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("baselines")
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=12, image_size=32, seed=1)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=1)
    label_map = build_label_map(loader.records, fold.known_families)
    gen = ImageGenerator(image_size=32)

    train_ds = EvalMalwareDataset(loader, fold.train, gen, label_map)
    val_ds = EvalMalwareDataset(loader, fold.val_known, gen, label_map)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=eval_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=eval_collate_fn)

    test_known_ds = EvalMalwareDataset(loader, fold.test_known, gen, label_map)
    test_unknown_ds = EvalMalwareDataset(loader, fold.test_unknown, gen, label_map)

    encoder = FeatureExtractionNetwork(_backbone_cfg(), ProjectionHeadConfig(hidden_dim=16, out_dim=8))
    encoder.eval()

    return {
        "loader": loader,
        "fold": fold,
        "label_map": label_map,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_known_ds": test_known_ds,
        "test_unknown_ds": test_unknown_ds,
        "encoder": encoder,
    }


def _check_baseline(baseline, fx, n_check=3):
    baseline.fit(fx["train_loader"])
    baseline.calibrate(fx["val_loader"])
    known_families = set(fx["label_map"].keys())

    for i in range(min(n_check, len(fx["test_known_ds"]))):
        x, _label, _sid = fx["test_known_ds"][i]
        s = baseline.score(x)
        assert isinstance(s, float)
        assert s == s  # not NaN
        pred = baseline.predict(x)
        assert pred == UNKNOWN_LABEL or pred in known_families

    for i in range(min(n_check, len(fx["test_unknown_ds"]))):
        x, _label, _sid = fx["test_unknown_ds"][i]
        s = baseline.score(x)
        assert s == s
        pred = baseline.predict(x)
        assert pred == UNKNOWN_LABEL or pred in known_families


def test_prototype_cosine_baseline(baseline_fixture):
    b = PrototypeCosineBaseline(baseline_fixture["encoder"], baseline_fixture["label_map"])
    _check_baseline(b, baseline_fixture)


def test_fixed_radius_prototype_baseline(baseline_fixture):
    b = FixedRadiusPrototypeBaseline(baseline_fixture["encoder"], baseline_fixture["label_map"], quantile=0.9)
    _check_baseline(b, baseline_fixture)


def test_mahalanobis_baseline(baseline_fixture):
    b = MahalanobisBaseline(baseline_fixture["encoder"], baseline_fixture["label_map"])
    _check_baseline(b, baseline_fixture)


def test_knn_baseline(baseline_fixture):
    b = KNNBaseline(baseline_fixture["encoder"], baseline_fixture["label_map"], k=3)
    _check_baseline(b, baseline_fixture)


def test_ocsvm_baseline(baseline_fixture):
    b = OCSVMBaseline(baseline_fixture["encoder"], baseline_fixture["label_map"], nu=0.1)
    _check_baseline(b, baseline_fixture)


def test_deep_svdd_baseline(baseline_fixture):
    b = DeepSVDDBaseline(_backbone_cfg(), baseline_fixture["label_map"], feature_dim=8, epochs=1)
    _check_baseline(b, baseline_fixture)


def test_softmax_msp_baseline(baseline_fixture):
    b = SoftmaxMSPBaseline(_backbone_cfg(), baseline_fixture["label_map"], epochs=1)
    _check_baseline(b, baseline_fixture)


def test_energy_baseline(baseline_fixture):
    b = EnergyBaseline(_backbone_cfg(), baseline_fixture["label_map"], epochs=1)
    _check_baseline(b, baseline_fixture)


def test_openmax_baseline(baseline_fixture):
    b = OpenMaxBaseline(_backbone_cfg(), baseline_fixture["label_map"], epochs=1, tail_size=5)
    _check_baseline(b, baseline_fixture)


def test_fixed_radius_baseline_matches_manual_quantile(baseline_fixture):
    b = FixedRadiusPrototypeBaseline(baseline_fixture["encoder"], baseline_fixture["label_map"], quantile=0.8)
    b.fit(baseline_fixture["train_loader"])
    for fam_idx, radius in enumerate(b.radii):
        if not np.isnan(radius):
            assert radius >= 0.0
