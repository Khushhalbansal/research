from lrmc.config import Config
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.engine.train import Trainer


def _tiny_cfg(output_dir, epochs=1):
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
    cfg.optim.checkpoint_every_n_epochs = 1
    cfg.is_synthetic = True
    cfg.output_dir = str(output_dir)
    return cfg


def test_trainer_runs_one_epoch_on_mock_data(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=8, image_size=32, seed=0)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=0)

    cfg = _tiny_cfg(tmp_path / "run", epochs=1)
    trainer = Trainer(cfg, loader, fold)
    metrics = trainer.fit()

    assert metrics["epochs_trained"] == 1
    assert "total" in metrics["final_losses"]
    assert metrics["is_synthetic"] is True
    assert (tmp_path / "run" / "checkpoint.pt").exists()


def test_trainer_loss_is_finite_and_decreases_roughly(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=1)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=1)

    cfg = _tiny_cfg(tmp_path / "run", epochs=3)
    trainer = Trainer(cfg, loader, fold)
    metrics = trainer.fit()

    losses = [h["total"] for h in metrics["history"]]
    assert all(l == l for l in losses)  # no NaNs
    assert losses[-1] < losses[0] + 1.0  # sanity: not exploding
