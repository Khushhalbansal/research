"""Proves the CUDA-OOM backoff (shrink batch, grow grad accumulation, retry)
and per-epoch status.json writing, without needing a real GPU -- the OOM is
simulated by monkeypatching the forward pass to raise once.
"""

from lrmc.config import Config
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.engine.train import Trainer
from lrmc.orchestration.status import read_job_status


def _tiny_cfg(output_dir, epochs=2, batch_size=8):
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
    cfg.data.batch_size = batch_size
    cfg.data.num_workers = 0
    cfg.optim.epochs = epochs
    cfg.optim.grad_accumulation_steps = 1
    cfg.is_synthetic = True
    cfg.output_dir = str(output_dir)
    return cfg


def _make_trainer(tmp_path, seed=0, **cfg_kwargs):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=16, image_size=32, seed=seed)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=seed)
    cfg = _tiny_cfg(tmp_path / "run", **cfg_kwargs)
    return Trainer(cfg, loader, fold)


def test_status_json_written_every_epoch(tmp_path):
    trainer = _make_trainer(tmp_path, epochs=2)
    trainer.fit()
    status = read_job_status(trainer.output_dir)
    assert status is not None
    assert status["epoch"] == 1  # last epoch (0-indexed)
    assert status["total_epochs"] == 2
    assert "epoch_seconds" in status
    assert status["batch_size"] == 8


def test_cuda_oom_backoff_shrinks_batch_and_grows_accumulation(tmp_path):
    trainer = _make_trainer(tmp_path, epochs=1, batch_size=8)
    original_step = trainer._train_one_epoch
    call_count = {"n": 0}

    def flaky_step():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return original_step()

    trainer._train_one_epoch = flaky_step
    metrics = trainer.fit()

    assert call_count["n"] == 2  # failed once, succeeded on retry
    assert trainer.cfg.data.batch_size == 4  # halved from 8
    assert trainer.cfg.optim.grad_accumulation_steps == 2  # doubled to compensate
    assert metrics["epochs_trained"] == 1


def test_cuda_oom_backoff_gives_up_after_max_retries(tmp_path):
    trainer = _make_trainer(tmp_path, epochs=1, batch_size=1)

    def always_oom():
        raise RuntimeError("CUDA out of memory.")

    trainer._train_one_epoch = always_oom
    try:
        trainer.fit()
        raised = False
    except RuntimeError as exc:
        raised = "cannot reduce further" in str(exc) or "giving up" in str(exc)
    assert raised


def test_non_oom_runtime_error_is_not_swallowed(tmp_path):
    trainer = _make_trainer(tmp_path, epochs=1)

    def boom():
        raise RuntimeError("some unrelated failure")

    trainer._train_one_epoch = boom
    threw_unrelated = False
    try:
        trainer.fit()
    except RuntimeError as exc:
        threw_unrelated = str(exc) == "some unrelated failure"
    assert threw_unrelated
