import torch

from lrmc.config import BackboneConfig, ProjectionHeadConfig
from lrmc.orchestration.benchmark import (
    expand_queue_with_seeds,
    measure_max_batch_size,
    measure_throughput_per_sample,
    reprioritize_tier,
    rewrite_queue_estimates,
)


def _tiny_backbone():
    return BackboneConfig(
        name="tiny_test",
        img_size=32,
        patch_size=8,
        depth=2,
        embed_dim=16,
        num_heads=2,
        mlp_ratio=2.0,
    )


def test_measure_throughput_per_sample_positive_on_cpu():
    sec = measure_throughput_per_sample(
        _tiny_backbone(),
        ProjectionHeadConfig(hidden_dim=16, out_dim=8),
        torch.device("cpu"),
        batch_size=4,
        n_steps=2,
    )
    assert sec > 0


def test_measure_max_batch_size_returns_start_batch_on_cpu():
    result = measure_max_batch_size(
        _tiny_backbone(),
        ProjectionHeadConfig(hidden_dim=16, out_dim=8),
        torch.device("cpu"),
        start_batch=8,
    )
    assert result == 8


def test_rewrite_queue_estimates_marks_measured_false(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("run_name: t\ndata:\n  dataset: malimg\noptim:\n  epochs: 10\n")
    queue = {
        "tiers": {
            "A": {
                "jobs": [
                    {
                        "name": "job1",
                        "config": str(cfg_path),
                        "estimated_gpu_minutes": 999,
                        "estimated": True,
                    }
                ]
            }
        }
    }
    sec_by_config = {str(cfg_path): 0.01}
    out = rewrite_queue_estimates(queue, sec_by_config, {"malimg": 100})
    job = out["tiers"]["A"]["jobs"][0]
    assert job["estimated"] is False
    expected = round(10 * 100 * 0.01 / 60.0, 2)
    assert job["estimated_gpu_minutes"] == expected
    # original untouched
    assert queue["tiers"]["A"]["jobs"][0]["estimated_gpu_minutes"] == 999


def test_reprioritize_tier_dependency_free_jobs_come_first():
    jobs = [
        {"name": "embed_a", "depends_on": "main", "estimated_gpu_minutes": 1},
        {"name": "main", "estimated_gpu_minutes": 100},
        {"name": "baseline_b", "estimated_gpu_minutes": 50},
    ]
    ordered = reprioritize_tier(jobs)
    names = [j["name"] for j in ordered]
    assert names.index("main") < names.index("embed_a")
    assert names.index("baseline_b") < names.index("embed_a")
    # cheapest-first among dependency-free jobs
    assert names.index("baseline_b") < names.index("main")


def test_expand_queue_with_seeds_creates_n_variants(tmp_path):
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text("run_name: base\nseed: 0\nsplit:\n  seed: 0\n")
    queue = {
        "tiers": {
            "A": {
                "jobs": [
                    {
                        "name": "main_lrmc",
                        "type": "train_lrmc",
                        "config": str(cfg_path),
                        "estimated_gpu_minutes": 10,
                    },
                    {
                        "name": "embed_baseline",
                        "type": "baseline_embedding",
                        "config": str(cfg_path),
                        "depends_on": "main_lrmc",
                        "estimated_gpu_minutes": 1,
                    },
                ]
            },
            "B": {
                "jobs": [
                    {"name": "ablation_x", "config": str(cfg_path), "estimated_gpu_minutes": 5}
                ]
            },
            "C": {
                "jobs": [
                    {"name": "big2015_main", "config": str(cfg_path), "estimated_gpu_minutes": 20}
                ]
            },
        }
    }
    out = expand_queue_with_seeds(
        queue, n_seeds=3, configs_out_dir=tmp_path / "generated", tiers=("A", "B")
    )

    # seed-major order: a seed's dependent job can start as soon as that
    # seed's own dependency finishes, without waiting on other seeds.
    tier_a_names = [j["name"] for j in out["tiers"]["A"]["jobs"]]
    assert tier_a_names == [
        "main_lrmc_seed0",
        "embed_baseline_seed0",
        "main_lrmc_seed1",
        "embed_baseline_seed1",
        "main_lrmc_seed2",
        "embed_baseline_seed2",
    ]
    embed0 = next(j for j in out["tiers"]["A"]["jobs"] if j["name"] == "embed_baseline_seed0")
    assert embed0["depends_on"] == "main_lrmc_seed0"

    tier_b_names = [j["name"] for j in out["tiers"]["B"]["jobs"]]
    assert len(tier_b_names) == 3

    # Tier C untouched (not in the expanded tiers)
    assert [j["name"] for j in out["tiers"]["C"]["jobs"]] == ["big2015_main"]

    from lrmc.config import load_config

    generated_cfg = load_config(tmp_path / "generated" / "main_lrmc_seed2.yaml")
    assert generated_cfg.seed == 2
    assert generated_cfg.split.seed == 2
    assert generated_cfg.run_name == "main_lrmc_seed2"
