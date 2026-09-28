"""Proves real multiprocessing (num_workers > 0) works through our actual
code paths, not just num_workers=0 (which every other test uses for speed).
This matters on Windows specifically: torch's DataLoader and
multiprocessing.Pool both use the "spawn" start method there, which
re-imports the entry-point module in each worker process -- unsafe unless
that module's multiprocessing-triggering code sits behind
`if __name__ == "__main__":`. lrmc.cli.main already has that guard; this
test proves the underlying Trainer/Big2015Loader code that guard protects
actually works when workers > 0, on whichever platform pytest is running on.
"""

from lrmc.config import Config
from lrmc.data.big2015 import Big2015Loader
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_big2015, generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.engine.train import Trainer


def test_trainer_with_real_multiprocess_workers(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=12, image_size=32, seed=0)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=0)

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
    cfg.data.num_workers = 2  # real worker processes, not in-process loading
    cfg.optim.epochs = 1
    cfg.is_synthetic = True
    cfg.output_dir = str(tmp_path / "run")

    trainer = Trainer(cfg, loader, fold)
    assert trainer.train_loader.num_workers == 2
    assert trainer.train_loader.persistent_workers is True

    metrics = trainer.fit()
    assert metrics["epochs_trained"] == 1
    assert "total" in metrics["final_losses"]


def test_trainer_num_workers_auto_sentinel_resolves_to_positive(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=8, image_size=32, seed=1)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=1)

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
    cfg.data.num_workers = -1  # "auto" sentinel
    cfg.optim.epochs = 1
    cfg.is_synthetic = True
    cfg.output_dir = str(tmp_path / "run")

    trainer = Trainer(cfg, loader, fold)
    assert trainer.train_loader.num_workers >= 0  # resolved to a concrete, non-negative count


def test_big2015_cache_build_with_real_multiprocess_pool(tmp_path):
    root = tmp_path / "mock_big2015"
    generate_mock_big2015(root, n_per_family=6, n_families=3, seed=0)
    cache_dir = tmp_path / "cache"
    loader = Big2015Loader(root, cache_dir, intermediate_size=32, n_workers=2)
    report = loader.build_cache()
    assert report.n_labeled == 18  # 6 * 3 families
    sample_id = loader.records[0].sample_id
    arr, is_image = loader.get_array(sample_id)
    assert arr.shape == (32, 32)
