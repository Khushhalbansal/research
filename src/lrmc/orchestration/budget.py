"""CPU-throughput -> GPU-minute estimation, honestly labeled as an estimate.

Measures an ACTUAL forward+backward step time on this CPU using the tiny_test
preset, scales it to a real backbone by the ratio of approximate FLOPs
(measured, not guessed), and divides by an explicitly stated, documented GPU
speedup assumption. Every number this module produces is either measured or
comes with its assumption spelled out in the returned dict -- never presented
as a real timing. See docs/DECISIONS.md.
"""

from __future__ import annotations

import time

import torch

from lrmc.config import BackboneConfig, ProjectionHeadConfig
from lrmc.eval.efficiency import approx_flops
from lrmc.models.network import FeatureExtractionNetwork

# Rough, documented assumption: a single Kaggle T4 running fp16/AMP is commonly
# 10-25x faster than one CPU core for this kind of small-ViT forward+backward
# workload. We use the conservative (slower) end of that range so time
# estimates err on the side of "this will take longer than we hoped" rather
# than the reverse, which is the safer failure mode for session-budget
# planning. Kaggle gives 2xT4, but DataParallel scaling is sub-linear, so we
# do NOT multiply by 2 here.
ASSUMED_GPU_SPEEDUP = 10.0


def measure_cpu_step_seconds(
    backbone_cfg: BackboneConfig, head_cfg: ProjectionHeadConfig, batch_size: int, n_steps: int = 5
) -> float:
    """Actual measured mean wall-clock seconds for one forward+backward step
    (batch of ``batch_size`` images) on this CPU, using the tiny_test preset."""
    net = FeatureExtractionNetwork(backbone_cfg, head_cfg)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    x = torch.rand(batch_size, 1, backbone_cfg.img_size, backbone_cfg.img_size)

    for _ in range(2):  # warmup
        optimizer.zero_grad()
        z = net(x)
        z.sum().backward()
        optimizer.step()

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        z = net(x)
        z.sum().backward()
        optimizer.step()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) / batch_size  # seconds per sample


def measured_flops_ratio(tiny_cfg: BackboneConfig, real_cfg: BackboneConfig) -> float:
    """Ratio of approx FLOPs(real backbone) / approx FLOPs(tiny_test backbone),
    both measured at the REAL backbone's input resolution (tiny_test is
    resized up for this one comparison call only -- not used for training)."""
    from copy import deepcopy

    tiny_at_real_res = deepcopy(tiny_cfg)
    tiny_at_real_res.img_size = real_cfg.img_size
    tiny_at_real_res.pretrained = False

    real_cfg_no_download = deepcopy(real_cfg)
    real_cfg_no_download.pretrained = False  # architecture-only comparison, no network calls

    from lrmc.models.backbone import ViTBackbone

    tiny_net = ViTBackbone(tiny_at_real_res)
    real_net = ViTBackbone(real_cfg_no_download)
    x_tiny = torch.rand(1, tiny_net.in_chans, real_cfg.img_size, real_cfg.img_size)
    x_real = torch.rand(1, real_net.in_chans, real_cfg.img_size, real_cfg.img_size)
    tiny_flops = approx_flops(tiny_net, x_tiny)
    real_flops = approx_flops(real_net, x_real)
    return real_flops / max(tiny_flops, 1)


def estimate_gpu_minutes(
    n_epochs: int,
    n_train_samples: int,
    cpu_sec_per_sample_tiny: float,
    flops_ratio: float,
    gpu_speedup: float = ASSUMED_GPU_SPEEDUP,
) -> dict:
    """Honest extrapolation from measured CPU tiny_test throughput to an
    estimated GPU wall-clock budget for a real-backbone run. Returns the
    estimate AND every assumption/measurement that produced it."""
    cpu_seconds_scaled_per_sample = cpu_sec_per_sample_tiny * flops_ratio
    total_cpu_seconds = n_epochs * n_train_samples * cpu_seconds_scaled_per_sample
    gpu_seconds = total_cpu_seconds / gpu_speedup
    return {
        "estimated_gpu_minutes": gpu_seconds / 60.0,
        "estimated": True,
        "assumptions": {
            "gpu_speedup": gpu_speedup,
            "flops_ratio_real_vs_tiny": flops_ratio,
            "measured_cpu_sec_per_sample_tiny": cpu_sec_per_sample_tiny,
            "n_epochs": n_epochs,
            "n_train_samples": n_train_samples,
        },
    }
