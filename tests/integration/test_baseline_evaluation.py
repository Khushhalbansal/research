from torch.utils.data import DataLoader

from lrmc.baselines.prototype_cosine import PrototypeCosineBaseline
from lrmc.config import BackboneConfig, Config, ProjectionHeadConfig
from lrmc.data.datasets import EvalMalwareDataset, build_label_map, eval_collate_fn
from lrmc.data.image_generator import ImageGenerator
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.eval.evaluate import run_baseline_evaluation
from lrmc.models.network import FeatureExtractionNetwork


def test_run_baseline_evaluation_produces_full_metrics_bundle(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=9)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=9)
    label_map = build_label_map(loader.records, fold.known_families)
    gen = ImageGenerator(image_size=32)

    encoder = FeatureExtractionNetwork(
        BackboneConfig(
            name="tiny_test", img_size=32, patch_size=8, depth=2, embed_dim=16, num_heads=2
        ),
        ProjectionHeadConfig(hidden_dim=16, out_dim=8),
    )
    encoder.eval()

    train_loader = DataLoader(
        EvalMalwareDataset(loader, fold.train, gen, label_map),
        batch_size=8,
        collate_fn=eval_collate_fn,
    )
    val_loader = DataLoader(
        EvalMalwareDataset(loader, fold.val_known, gen, label_map),
        batch_size=8,
        collate_fn=eval_collate_fn,
    )

    baseline = PrototypeCosineBaseline(encoder, label_map)
    baseline.fit(train_loader)
    baseline.calibrate(val_loader)

    cfg = Config()
    cfg.data.image_size = 32
    metrics = run_baseline_evaluation(
        cfg, loader, fold, baseline, gen, method_name="prototype_cosine"
    )

    assert metrics["method"] == "prototype_cosine"
    assert 0.0 <= metrics["closed_set"]["accuracy"] <= 1.0
    assert 0.0 <= metrics["open_set"]["auroc"] <= 1.0
    assert "oscr_auc" in metrics["oscr"]
