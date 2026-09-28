import json

from lrmc.config import Config
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.engine.train import Trainer
from lrmc.eval.evaluate import run_evaluation, write_metrics_json


def _tiny_cfg(output_dir, epochs=2):
    cfg = Config()
    cfg.backbone.name = "tiny_test"
    cfg.backbone.img_size = 32
    cfg.backbone.patch_size = 8
    cfg.backbone.depth = 2
    cfg.backbone.embed_dim = 16
    cfg.backbone.num_heads = 2
    cfg.projection_head.hidden_dim = 16
    cfg.projection_head.out_dim = 8
    cfg.data.image_size = 32
    cfg.data.batch_size = 4
    cfg.data.num_workers = 0
    cfg.optim.epochs = epochs
    cfg.is_synthetic = True
    cfg.output_dir = str(output_dir)
    return cfg


def test_full_train_then_evaluate_produces_valid_metrics(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=5)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=5)

    cfg = _tiny_cfg(tmp_path / "run", epochs=2)
    trainer = Trainer(cfg, loader, fold)
    trainer.fit()

    metrics = run_evaluation(cfg, loader, fold, trainer.checkpoint_path)

    assert metrics["is_synthetic"] is True
    assert 0.0 <= metrics["closed_set"]["accuracy"] <= 1.0
    assert 0.0 <= metrics["open_set"]["auroc"] <= 1.0
    assert 0.0 <= metrics["oscr"]["oscr_auc"] <= 1.0
    assert metrics["efficiency"]["total_params"] > 0
    assert metrics["model_hash"]

    out_path = tmp_path / "run" / "metrics.json"
    write_metrics_json(metrics, out_path)
    reloaded = json.loads(out_path.read_text())
    assert reloaded["config_hash"] == metrics["config_hash"]
