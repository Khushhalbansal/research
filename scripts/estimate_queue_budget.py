#!/usr/bin/env python
"""Prints honest, MEASURED GPU-minute estimates (see
src/lrmc/orchestration/budget.py) for the job shapes used in
experiments/queue.yaml, so the numbers in that file can be refreshed after
any architecture/epoch-count change instead of hand-guessed.

Usage: python scripts/estimate_queue_budget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lrmc.config import BackboneConfig, ProjectionHeadConfig  # noqa: E402
from lrmc.orchestration.budget import (  # noqa: E402
    estimate_gpu_minutes,
    measure_cpu_step_seconds,
    measured_flops_ratio,
)


def main() -> None:
    tiny = BackboneConfig(
        name="tiny_test", img_size=32, patch_size=8, depth=2, embed_dim=16, num_heads=2, mlp_ratio=2.0
    )
    head = ProjectionHeadConfig(hidden_dim=16, out_dim=8)
    sec_per_sample = measure_cpu_step_seconds(tiny, head, batch_size=8, n_steps=6)

    vit_real = BackboneConfig(name="vit_tiny_patch16_224", img_size=224, pretrained=False)
    resnet_real = BackboneConfig(name="resnet18", img_size=224, pretrained=False)
    ratio_vit = measured_flops_ratio(tiny, vit_real)
    ratio_resnet = measured_flops_ratio(tiny, resnet_real)

    print(f"measured cpu_sec_per_sample (tiny_test): {sec_per_sample:.6f}")
    print(f"measured flops_ratio vit_tiny_patch16_224 / tiny_test: {ratio_vit:.1f}")
    print(f"measured flops_ratio resnet18 / tiny_test: {ratio_resnet:.1f}")
    print()

    scenarios = [
        ("malimg_main_lrmc (20ep, ~6000 known-train)", 20, 6000, ratio_vit),
        ("network baseline (15ep, ~6000)", 15, 6000, ratio_vit),
        ("ablation (15ep, ~6000)", 15, 6000, ratio_vit),
        ("ablation backbone_resnet18 (15ep, ~6000)", 15, 6000, ratio_resnet),
        ("big2015_main_lrmc (15ep, ~7000 known-train)", 15, 7000, ratio_vit),
        ("big2015 network baseline (15ep, ~7000)", 15, 7000, ratio_vit),
    ]
    for name, epochs, n, ratio in scenarios:
        est = estimate_gpu_minutes(
            n_epochs=epochs, n_train_samples=n, cpu_sec_per_sample_tiny=sec_per_sample, flops_ratio=ratio
        )
        print(f"{name}: {est['estimated_gpu_minutes']:.1f} min (estimated=True)")


if __name__ == "__main__":
    main()
