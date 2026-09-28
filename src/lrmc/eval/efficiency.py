"""Efficiency measurements for the paper's efficiency claims: parameter count,
approximate FLOPs, per-sample inference latency/throughput, peak memory.

Measured on whatever device the run actually used (recorded as ``device``);
never conflated with the separate, explicitly-labelled GPU-minute ESTIMATES in
``experiments/queue.yaml`` (see ``orchestration/budget.py`` and
docs/DECISIONS.md).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def approx_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """Rough multiply-add FLOP count via forward hooks on Linear/Conv2d layers
    (the layers that dominate ViT and ResNet compute). Documented as
    approximate -- attention softmax/normalization ops are not counted."""
    flop_total = {"value": 0}
    hooks = []

    def linear_hook(module: nn.Linear, inp, out):
        batch_elems = inp[0].numel() // module.in_features
        flop_total["value"] += 2 * batch_elems * module.in_features * module.out_features

    def conv_hook(module: nn.Conv2d, inp, out):
        out_h, out_w = out.shape[-2], out.shape[-1]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        flop_total["value"] += 2 * out.shape[0] * out_h * out_w * module.out_channels * kernel_ops

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))

    model.eval()
    with torch.no_grad():
        model(input_tensor)
    for h in hooks:
        h.remove()
    return flop_total["value"]


def measure_latency(
    forward_fn, input_tensor: torch.Tensor, n_warmup: int = 5, n_runs: int = 20
) -> dict:
    for _ in range(n_warmup):
        forward_fn(input_tensor)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        forward_fn(input_tensor)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    batch_size = input_tensor.shape[0]
    return {
        "mean_latency_s_per_batch": float(times.mean()),
        "p50_latency_s_per_batch": float(np.percentile(times, 50)),
        "p95_latency_s_per_batch": float(np.percentile(times, 95)),
        "mean_latency_s_per_sample": float(times.mean() / batch_size),
        "throughput_samples_per_s": float(batch_size / times.mean()),
        "batch_size": batch_size,
        "n_runs": n_runs,
    }


def measure_peak_memory(forward_fn, input_tensor: torch.Tensor, device: torch.device) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        forward_fn(input_tensor)
        peak = torch.cuda.max_memory_allocated(device)
        return {"peak_memory_bytes": int(peak), "measured_on": "cuda"}
    else:
        import psutil

        process = psutil.Process(os.getpid())
        before = process.memory_info().rss
        forward_fn(input_tensor)
        after = process.memory_info().rss
        return {"peak_memory_bytes": int(max(after - before, 0)), "measured_on": "cpu_rss_delta"}


def efficiency_report(model: nn.Module, input_tensor: torch.Tensor, device: torch.device) -> dict:
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    def forward_fn(x):
        with torch.no_grad():
            return model(x)

    report = {"device": str(device)}
    report.update(count_parameters(model))
    report["approx_flops"] = approx_flops(model, input_tensor)
    report.update(measure_latency(forward_fn, input_tensor))
    report.update(measure_peak_memory(forward_fn, input_tensor, device))
    return report
