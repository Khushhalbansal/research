"""GPU-aware auto-tuning: `lrmc benchmark` measures REAL throughput and max
batch size on the actual device, rewrites experiments/queue.yaml's
estimated_gpu_minutes as measured (estimated: false), re-prioritizes each
tier (dependency-free jobs first, cheapest-first within a priority group),
and -- with no external session quota on a dedicated workstation -- expands
Tier A/B with multi-seed repeats for statistical power.
"""

from __future__ import annotations

import copy
import logging
import shutil
import time
from pathlib import Path

import torch
import yaml

from lrmc.config import (
    BackboneConfig,
    Config,
    ProjectionHeadConfig,
    load_config,
    merge_overrides,
    save_config,
)
from lrmc.models.network import FeatureExtractionNetwork

logger = logging.getLogger(__name__)

# Approximate known-train sample counts, used to turn a measured seconds/
# sample into an epoch/job estimate when no real fold.json is available yet
# to read the exact count from. The mock_* entries exist so `lrmc benchmark`
# also works meaningfully against experiments/queue_rehearsal.yaml (CPU
# rehearsal), not just the real queue.
DEFAULT_N_TRAIN_SAMPLES = {"malimg": 6000, "big2015": 7000, "mock_malimg": 500, "mock_big2015": 500}


def measure_throughput_per_sample(
    backbone_cfg: BackboneConfig,
    head_cfg: ProjectionHeadConfig,
    device: torch.device,
    batch_size: int,
    n_warmup: int = 3,
    n_steps: int = 10,
) -> float:
    """Real measured seconds/sample for one forward+backward step, shaped
    like one Trainer step (two views concatenated -> batch_size*2 images)."""
    net = FeatureExtractionNetwork(backbone_cfg, head_cfg).to(device)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    x = torch.rand(batch_size * 2, 1, backbone_cfg.img_size, backbone_cfg.img_size, device=device)

    for _ in range(n_warmup):
        optimizer.zero_grad()
        net(x).sum().backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        net(x).sum().backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return elapsed / n_steps / batch_size


def measure_max_batch_size(
    backbone_cfg: BackboneConfig,
    head_cfg: ProjectionHeadConfig,
    device: torch.device,
    start_batch: int = 8,
    max_batch: int = 1024,
) -> int:
    """CUDA-specific max-batch-size search (doubling + OOM detection). On CPU
    there's no VRAM ceiling to search for in the same sense, so this just
    returns start_batch unchanged."""
    if device.type != "cuda":
        return start_batch
    net = FeatureExtractionNetwork(backbone_cfg, head_cfg).to(device)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    batch, last_good = start_batch, start_batch
    while batch <= max_batch:
        try:
            x = torch.rand(
                batch * 2, 1, backbone_cfg.img_size, backbone_cfg.img_size, device=device
            )
            optimizer.zero_grad()
            net(x).sum().backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            last_good = batch
            del x
            torch.cuda.empty_cache()
            batch *= 2
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                break
            raise
    return last_good


def _backbone_signature(cfg: Config) -> tuple:
    b = cfg.backbone
    return (
        b.name,
        b.img_size,
        b.patch_size,
        b.depth,
        b.embed_dim,
        b.num_heads,
        b.mlp_ratio,
        b.in_chans_mode,
        cfg.projection_head.hidden_dim,
        cfg.projection_head.out_dim,
    )


def measure_all_configs(
    config_paths: list[str], device: torch.device, batch_size: int = 16
) -> dict:
    """Measures throughput/max-batch once per UNIQUE backbone signature
    (many configs -- e.g. every ablation -- share the same architecture),
    then fans the result back out to every config path. Returns
    {"sec_per_sample": {path: float}, "max_batch": {path: int}}."""
    sig_cache: dict[tuple, tuple[float, int]] = {}
    sec_per_sample, max_batch_out = {}, {}
    for path in config_paths:
        cfg = load_config(path)
        sig = _backbone_signature(cfg)
        if sig not in sig_cache:
            bs = min(batch_size, cfg.data.batch_size)
            sec = measure_throughput_per_sample(cfg.backbone, cfg.projection_head, device, bs)
            mb = measure_max_batch_size(cfg.backbone, cfg.projection_head, device, start_batch=bs)
            sig_cache[sig] = (sec, mb)
            logger.info("Measured %s: %.4f s/sample, max_batch=%d", path, sec, mb)
        sec_per_sample[path], max_batch_out[path] = sig_cache[sig]
    return {"sec_per_sample": sec_per_sample, "max_batch": max_batch_out}


def rewrite_queue_estimates(
    queue: dict,
    sec_per_sample_by_config: dict[str, float],
    n_train_samples_by_dataset: dict[str, int] | None = None,
) -> dict:
    """Pure data transform (no I/O): recomputes estimated_gpu_minutes for
    every job whose config was measured, marking it estimated: false."""
    n_train_samples_by_dataset = n_train_samples_by_dataset or DEFAULT_N_TRAIN_SAMPLES
    queue = copy.deepcopy(queue)
    for tier in queue.get("tiers", {}).values():
        for job in tier.get("jobs", []):
            cfg_path = job.get("config")
            if not cfg_path or cfg_path not in sec_per_sample_by_config:
                continue
            cfg = load_config(cfg_path)
            n_samples = n_train_samples_by_dataset.get(cfg.data.dataset)
            if n_samples is None:
                continue
            epochs = job.get("params", {}).get("epochs", cfg.optim.epochs)
            minutes = epochs * n_samples * sec_per_sample_by_config[cfg_path] / 60.0
            job["estimated_gpu_minutes"] = round(minutes, 2)
            job["estimated"] = False
    return queue


def reprioritize_tier(jobs: list[dict]) -> list[dict]:
    """Dependency-free jobs (main runs, network baselines) before jobs that
    `depends_on` one of them (embedding baselines) -- correctness-preserving
    since every dependency target IS dependency-free by construction; within
    each group, cheapest-first so more jobs finish before any budget cutoff."""

    def key(job: dict) -> tuple[int, float]:
        has_dep = 1 if job.get("depends_on") else 0
        return (has_dep, job.get("estimated_gpu_minutes", 0.0))

    return sorted(jobs, key=key)


def expand_queue_with_seeds(
    queue: dict,
    n_seeds: int,
    configs_out_dir: str | Path,
    tiers: tuple[str, ...] = ("A", "B"),
) -> dict:
    """Replaces each job in the given tiers with n_seeds variants
    (seed=0..n_seeds-1), each pointing at its own generated config (seed and
    split.seed both overridden to the repeat index, so each repeat is an
    independently re-split, independently-initialized run -- the
    statistically rigorous form of "multi-seed"). depends_on references are
    rewritten to the matching seed variant, preserving the paired-run
    structure eval/metrics.py's significance test expects."""
    configs_out_dir = Path(configs_out_dir)
    configs_out_dir.mkdir(parents=True, exist_ok=True)
    queue = copy.deepcopy(queue)

    for tier_name, tier in queue.get("tiers", {}).items():
        if tier_name not in tiers:
            continue
        new_jobs = []
        # Seed-major (outer loop over seed, inner over jobs): a seed's
        # dependent baseline_embedding job can start as soon as THAT seed's
        # main run finishes, instead of waiting for every seed's main run to
        # finish first -- better for a queue that may be interrupted at any
        # point (see task 4's AnyDesk-disconnect robustness).
        for seed in range(n_seeds):
            for job in tier.get("jobs", []):
                base_name = job["name"]
                base_cfg_path = job["config"]
                new_job = copy.deepcopy(job)
                new_name = f"{base_name}_seed{seed}"
                new_job["name"] = new_name
                if job.get("depends_on"):
                    new_job["depends_on"] = f"{job['depends_on']}_seed{seed}"

                cfg = load_config(base_cfg_path)
                cfg = merge_overrides(cfg, {"seed": seed, "split.seed": seed})
                cfg.run_name = new_name
                out_cfg_path = configs_out_dir / f"{new_name}.yaml"
                save_config(cfg, out_cfg_path)
                new_job["config"] = str(out_cfg_path)
                new_jobs.append(new_job)
        tier["jobs"] = new_jobs
    return queue


def run_benchmark(
    queue_path: str | Path,
    device: torch.device,
    n_seeds: int = 5,
    seed_tiers: tuple[str, ...] = ("A", "B"),
    configs_out_dir: str | Path = "configs/generated",
    n_train_samples_by_dataset: dict[str, int] | None = None,
    expand_seeds: bool = True,
) -> dict:
    """Full benchmark-and-rewrite: measures every distinct config referenced
    by the queue, rewrites estimates + priority order, optionally expands
    with multi-seed repeats, and overwrites queue_path (after backing up the
    original once)."""
    with open(queue_path, encoding="utf-8") as fh:
        queue = yaml.safe_load(fh)

    config_paths = sorted(
        {
            job["config"]
            for tier in queue.get("tiers", {}).values()
            for job in tier.get("jobs", [])
            if job.get("config")
        }
    )
    measurements = measure_all_configs(config_paths, device)

    queue = rewrite_queue_estimates(
        queue, measurements["sec_per_sample"], n_train_samples_by_dataset
    )
    for tier in queue.get("tiers", {}).values():
        tier["jobs"] = reprioritize_tier(tier["jobs"])
    if expand_seeds:
        queue = expand_queue_with_seeds(queue, n_seeds, configs_out_dir, tiers=seed_tiers)

    backup_path = Path(str(queue_path) + ".pre_benchmark_backup")
    if not backup_path.exists():
        shutil.copy(queue_path, backup_path)
    with open(queue_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(queue, fh, sort_keys=False)

    return {
        "device": str(device),
        "measurements": measurements,
        "n_seeds": n_seeds if expand_seeds else 0,
        "queue_path": str(queue_path),
        "backup_path": str(backup_path),
        "n_jobs_after": sum(len(t.get("jobs", [])) for t in queue.get("tiers", {}).values()),
    }
