"""Proves kill-and-resume produces an identical result to an uninterrupted run.

Trains 4 epochs straight through; separately trains 2 epochs, constructs a
BRAND NEW Trainer instance (simulating a fresh process after a kill) pointed
at the same output_dir, and continues for 2 more epochs. The two final
network/prototype/radii states must match exactly.
"""

from __future__ import annotations

import copy

import torch

from lrmc.config import Config
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.engine.train import Trainer


def _cfg(output_dir, epochs):
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
    cfg.data.class_balanced_sampling = True
    cfg.optim.epochs = epochs
    cfg.optim.checkpoint_every_n_epochs = 1
    cfg.seed = 123
    cfg.is_synthetic = True
    cfg.output_dir = str(output_dir)
    return cfg


def _state_signature(trainer: Trainer) -> dict:
    net = trainer._bare_network()
    net_state = {k: v.clone() for k, v in net.state_dict().items()}
    proto_state = {k: v.clone() for k, v in trainer.prototypes.state_dict().items()}
    radii_state = {k: v.clone() for k, v in trainer.radii.state_dict().items()}
    return {"net": net_state, "proto": proto_state, "radii": radii_state}


def _assert_signatures_equal(sig_a: dict, sig_b: dict) -> None:
    for group in ("net", "proto", "radii"):
        keys_a, keys_b = set(sig_a[group]), set(sig_b[group])
        assert keys_a == keys_b, f"{group} state_dict keys differ"
        for k in keys_a:
            a, b = sig_a[group][k], sig_b[group][k]
            if a.dtype == torch.bool:
                assert torch.equal(a, b), f"{group}.{k} differs"
            else:
                assert torch.allclose(a, b, atol=1e-6), f"{group}.{k} differs"


def test_kill_and_resume_matches_uninterrupted_run(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=7)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=7)

    # Uninterrupted: 4 epochs straight through.
    uninterrupted_dir = tmp_path / "uninterrupted"
    cfg_full = _cfg(uninterrupted_dir, epochs=4)
    trainer_full = Trainer(cfg_full, loader, copy.deepcopy(fold))
    trainer_full.fit()
    sig_full = _state_signature(trainer_full)

    # Interrupted: 2 epochs, then a BRAND NEW Trainer resumes for 2 more.
    resumed_dir = tmp_path / "resumed"
    cfg_part1 = _cfg(resumed_dir, epochs=2)
    trainer_part1 = Trainer(cfg_part1, loader, copy.deepcopy(fold))
    trainer_part1.fit()

    cfg_part2 = _cfg(resumed_dir, epochs=4)  # same output_dir -> resumes checkpoint
    trainer_part2 = Trainer(cfg_part2, loader, copy.deepcopy(fold))
    trainer_part2.fit()
    sig_resumed = _state_signature(trainer_part2)

    _assert_signatures_equal(sig_full, sig_resumed)
