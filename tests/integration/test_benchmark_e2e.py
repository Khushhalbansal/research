import yaml

from lrmc.orchestration.benchmark import run_benchmark
from lrmc.utils.hardware import resolve_device


def _write_tiny_config(path, run_name, dataset="mock_malimg", epochs=2):
    path.write_text(f"""
run_name: {run_name}
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
  dataset: {dataset}
  root: data_cache/mock
  batch_size: 8
  image_size: 32
split:
  protocol: random_k_unknown
  n_unknown: 2
  seed: 0
optim:
  epochs: {epochs}
output_dir: runs/{run_name}
""")


def test_run_benchmark_rewrites_and_expands_queue(tmp_path):
    main_cfg = tmp_path / "main.yaml"
    _write_tiny_config(main_cfg, "main_lrmc")
    embed_cfg = tmp_path / "embed.yaml"
    _write_tiny_config(embed_cfg, "embed_baseline")

    queue_path = tmp_path / "queue.yaml"
    queue_data = {
        "session_budget_minutes": 60,
        "tiers": {
            "A": {
                "jobs": [
                    {
                        "name": "main_lrmc",
                        "type": "train_lrmc",
                        "config": str(main_cfg),
                        "estimated_gpu_minutes": 999,
                    },
                    {
                        "name": "embed_baseline",
                        "type": "baseline_embedding",
                        "config": str(embed_cfg),
                        "depends_on": "main_lrmc",
                        "estimated_gpu_minutes": 1,
                    },
                ]
            },
            "B": {"jobs": []},
        },
    }
    with open(queue_path, "w") as fh:
        yaml.safe_dump(queue_data, fh)

    device = resolve_device("cpu")
    result = run_benchmark(
        queue_path,
        device,
        n_seeds=2,
        seed_tiers=("A", "B"),
        configs_out_dir=tmp_path / "generated",
        n_train_samples_by_dataset={"mock_malimg": 500},
    )

    assert result["device"] == "cpu"
    assert result["n_seeds"] == 2
    backup = tmp_path / "queue.yaml.pre_benchmark_backup"
    assert backup.exists()
    backup_data = yaml.safe_load(backup.read_text())
    assert backup_data["tiers"]["A"]["jobs"][0]["estimated_gpu_minutes"] == 999  # untouched

    with open(queue_path) as fh:
        new_queue = yaml.safe_load(fh)
    names = [j["name"] for j in new_queue["tiers"]["A"]["jobs"]]
    assert names == [
        "main_lrmc_seed0",
        "embed_baseline_seed0",
        "main_lrmc_seed1",
        "embed_baseline_seed1",
    ]
    for job in new_queue["tiers"]["A"]["jobs"]:
        assert job["estimated"] is False
        assert job["estimated_gpu_minutes"] != 999
