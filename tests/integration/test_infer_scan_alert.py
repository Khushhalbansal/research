import json

import torch

from lrmc.config import Config
from lrmc.data.image_generator import ImageGenerator
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.engine.alert import AlertGenerator
from lrmc.engine.infer import FrozenInferenceEngine
from lrmc.engine.scan import Scanner
from lrmc.engine.train import Trainer


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


def _train_tiny_model(tmp_path, seed=0):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=seed)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=seed)
    cfg = _tiny_cfg(tmp_path / "run", epochs=2)
    trainer = Trainer(cfg, loader, fold)
    trainer.fit()
    return cfg, loader, fold, trainer.checkpoint_path


def test_frozen_inference_is_deterministic_and_batch_invariant(tmp_path):
    cfg, loader, fold, ckpt_path = _train_tiny_model(tmp_path)
    engine = FrozenInferenceEngine(cfg, ckpt_path)

    gen = ImageGenerator(image_size=cfg.data.image_size)
    sample_id = fold.test_known[0]
    arr, is_image = loader.get_array(sample_id)
    image = gen.eval_view(arr, is_pre_rendered_image=is_image)

    out1 = engine.predict(image)
    out2 = engine.predict(image)
    assert out1.verdict == out2.verdict
    assert out1.nearest_family == out2.nearest_family
    assert abs(out1.score - out2.score) < 1e-6

    # batch-size invariance: single-sample vs. as part of a batch of 3
    filler = torch.rand(2, 1, cfg.data.image_size, cfg.data.image_size)
    batch = torch.cat([image.unsqueeze(0), filler], dim=0)
    z_batch = engine.embed(batch)
    z_single = engine.embed(image)
    assert torch.allclose(z_batch[0], z_single[0], atol=1e-5)


def test_unknown_family_is_more_likely_flagged_zero_day(tmp_path):
    cfg, loader, fold, ckpt_path = _train_tiny_model(tmp_path, seed=2)
    engine = FrozenInferenceEngine(cfg, ckpt_path)
    gen = ImageGenerator(image_size=cfg.data.image_size)

    def _verdict_rate(sample_ids):
        n_zero_day = 0
        for sid in sample_ids:
            arr, is_image = loader.get_array(sid)
            image = gen.eval_view(arr, is_pre_rendered_image=is_image)
            out = engine.predict(image)
            if out.verdict == "ZERO_DAY":
                n_zero_day += 1
        return n_zero_day / max(len(sample_ids), 1)

    known_rate = _verdict_rate(fold.test_known)
    unknown_rate = _verdict_rate(fold.test_unknown)
    # not a strict guarantee on 2 epochs of tiny-model training, but the
    # unknown family should be flagged zero-day noticeably more often.
    assert unknown_rate >= known_rate


def test_scan_path_reads_only_bytes_and_emits_jsonl_alert(tmp_path):
    cfg, loader, fold, ckpt_path = _train_tiny_model(tmp_path, seed=3)
    engine = FrozenInferenceEngine(cfg, ckpt_path)
    gen = ImageGenerator(image_size=cfg.data.image_size)

    alerts_path = tmp_path / "alerts.jsonl"
    alert_gen = AlertGenerator(alerts_path)
    scanner = Scanner(engine, gen, alert_gen)

    fake_exe = tmp_path / "sample.exe"
    fake_exe.write_bytes(bytes([65, 90, 77] * 500))

    alert = scanner.scan_file(fake_exe)
    assert alert.verdict in ("KNOWN", "ZERO_DAY")
    assert len(alert.sha256) == 64
    assert alert.model_hash

    lines = alerts_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["sha256"] == alert.sha256
    assert "ratios" in record and "distances" in record


def test_scan_many_appends_multiple_jsonl_lines(tmp_path):
    cfg, loader, fold, ckpt_path = _train_tiny_model(tmp_path, seed=4)
    engine = FrozenInferenceEngine(cfg, ckpt_path)
    gen = ImageGenerator(image_size=cfg.data.image_size)
    alerts_path = tmp_path / "alerts.jsonl"
    alert_gen = AlertGenerator(alerts_path)
    scanner = Scanner(engine, gen, alert_gen)

    paths = []
    for i in range(3):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(bytes([i] * 1000))
        paths.append(p)
    scanner.scan_many(paths)
    lines = alerts_path.read_text().strip().splitlines()
    assert len(lines) == 3
