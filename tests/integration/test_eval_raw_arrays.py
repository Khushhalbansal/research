import numpy as np

from lrmc.config import Config
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.engine.train import Trainer
from lrmc.eval.evaluate import run_evaluation


def test_run_evaluation_saves_raw_arrays_npz(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=11)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=11)

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
    cfg.optim.epochs = 1
    cfg.is_synthetic = True
    cfg.output_dir = str(tmp_path / "run")

    trainer = Trainer(cfg, loader, fold)
    trainer.fit()

    npz_path = tmp_path / "run" / "raw_eval_arrays.npz"
    run_evaluation(cfg, loader, fold, trainer.checkpoint_path, save_arrays_path=npz_path)

    assert npz_path.exists()
    data = np.load(npz_path, allow_pickle=True)
    assert "known_scores" in data
    assert "known_embeddings" in data
    assert "prototypes" in data
    assert "radii" in data
    assert data["known_embeddings"].shape[0] == len(fold.test_known)
    assert data["prototypes"].shape[1] == 8  # embed dim
