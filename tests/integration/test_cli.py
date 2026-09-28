import json

from lrmc.cli.main import main
from lrmc.data.mock import generate_mock_malimg


def _write_cfg(tmp_path, root, out_dir):
    return f"""
run_name: cli_test
seed: 0
device: cpu
is_synthetic: true
backbone:
  name: tiny_test
  img_size: 32
  patch_size: 8
  depth: 2
  embed_dim: 16
  num_heads: 2
projection_head:
  hidden_dim: 16
  out_dim: 8
data:
  dataset: malimg
  root: {root}
  image_size: 32
  batch_size: 4
  num_workers: 0
split:
  protocol: random_k_unknown
  n_unknown: 2
  seed: 0
optim:
  epochs: 1
  checkpoint_every_n_epochs: 1
output_dir: {out_dir}
"""


def test_cli_train_and_evaluate(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=0)
    out_dir = tmp_path / "run"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(_write_cfg(tmp_path, root, out_dir))

    main(["train", str(cfg_path)])
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "train_metrics.json").exists()
    assert (out_dir / "fold.json").exists()

    main(["evaluate", str(cfg_path)])
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["is_synthetic"] is True


def test_cli_infer_on_a_malimg_style_image(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=1)
    out_dir = tmp_path / "run"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(_write_cfg(tmp_path, root, out_dir))
    main(["train", str(cfg_path)])

    sample_image = next((root).rglob("*.png"))
    main(["infer", str(cfg_path), str(sample_image)])  # just checks it runs without raising


def test_cli_scan_on_raw_bytes(tmp_path, capsys):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=2)
    out_dir = tmp_path / "run"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(_write_cfg(tmp_path, root, out_dir))
    main(["train", str(cfg_path)])

    fake_exe = tmp_path / "test.exe"
    fake_exe.write_bytes(bytes([1, 2, 3, 4] * 200))
    alerts_out = tmp_path / "alerts.jsonl"
    main(["scan", str(cfg_path), str(fake_exe), "--alerts-out", str(alerts_out)])
    assert alerts_out.exists()
    lines = alerts_out.read_text().strip().splitlines()
    assert len(lines) == 1


def test_cli_run_queue_dry_run(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    shutil.copytree(repo_root / "configs", tmp_path / "configs")
    shutil.copytree(repo_root / "experiments", tmp_path / "experiments")

    main(["run-queue", "experiments/queue_rehearsal.yaml", "--dry-run"])
    assert (tmp_path / "runs" / "queue_state.json").exists()
