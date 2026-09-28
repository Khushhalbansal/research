"""Hardware detection and resolution: device selection (incl. "auto" and
least-loaded-GPU picking for a shared machine), worker-count auto-tuning,
and the GPU/disk/CUDA facts printed by scripts/setup_env.py and `lrmc status`.

Nothing here is required for correctness -- every function degrades to a
safe default (cpu, 0 workers, "unavailable") when nvidia-smi/CUDA aren't
present, since the target workstation's OS/GPU/driver setup is unknown until
we actually connect to it.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch


def resolve_device(device_str: str, prefer_least_loaded: bool = True) -> torch.device:
    """"auto" -> least-loaded CUDA GPU if available else cpu.
    "cuda" -> a specific/least-loaded CUDA GPU if available else cpu (never
    raises just because CUDA is unavailable -- matches the existing
    Trainer/FrozenInferenceEngine fallback behavior).
    "cuda:N" / "cpu" -> used exactly as given (still falls back to cpu if
    "cuda:N" is requested but CUDA isn't available)."""
    if device_str in ("auto", "cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if prefer_least_loaded and torch.cuda.device_count() > 1:
            idx = pick_least_loaded_gpu()
            return torch.device(f"cuda:{idx}" if idx is not None else "cuda")
        return torch.device("cuda")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def auto_num_workers(cap: int = 8) -> int:
    """A conservative default: leave one core free for the main process,
    cap it (very high worker counts rarely help and can thrash I/O on a
    shared machine), and never go below 0."""
    cpu_count = os.cpu_count() or 1
    return max(0, min(cap, cpu_count - 1))


def resolve_num_workers(configured: int) -> int:
    """``configured < 0`` is the "auto" sentinel (see
    configs/base_malimg.yaml); any other value is used as-is, including 0
    (explicit single-process loading, the safe default for tests/sandboxes)."""
    return auto_num_workers() if configured < 0 else configured


def free_disk_bytes(path: str | Path = ".") -> int:
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free


def _nvidia_smi_query(fields: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def gpu_status_list() -> list[dict]:
    """Per-GPU status via nvidia-smi (utilization %, memory used/total MB,
    name). Returns [] if nvidia-smi isn't on PATH or no GPU is present --
    this is expected and handled, not an error, since the workstation's
    driver setup is unknown ahead of time."""
    lines = _nvidia_smi_query("index,name,utilization.gpu,memory.used,memory.total")
    if lines is None:
        return []
    statuses = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, util, mem_used, mem_total = parts
        try:
            statuses.append(
                {
                    "index": int(idx),
                    "name": name,
                    "utilization_pct": float(util),
                    "memory_used_mb": float(mem_used),
                    "memory_total_mb": float(mem_total),
                    "memory_free_mb": float(mem_total) - float(mem_used),
                }
            )
        except ValueError:
            continue
    return statuses


def pick_least_loaded_gpu() -> int | None:
    """Index of the GPU with the most free VRAM (ties broken by lowest
    utilization). Falls back to torch's own per-device memory query if
    nvidia-smi is unavailable; returns None if neither works or there is
    only one GPU (torch.cuda's default device is fine then)."""
    statuses = gpu_status_list()
    if statuses:
        statuses.sort(key=lambda s: (-s["memory_free_mb"], s["utilization_pct"]))
        return statuses[0]["index"]
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        free_by_device = []
        for i in range(torch.cuda.device_count()):
            free, _total = torch.cuda.mem_get_info(i)
            free_by_device.append((free, i))
        free_by_device.sort(reverse=True)
        return free_by_device[0][1]
    return None


def free_vram_bytes(device_index: int | None = None) -> int | None:
    if not torch.cuda.is_available():
        return None
    idx = device_index if device_index is not None else torch.cuda.current_device()
    free, _total = torch.cuda.mem_get_info(idx)
    return free


def system_summary() -> dict:
    """GPU name/VRAM/CUDA version/free disk/Python version -- the facts
    scripts/setup_env.py prints and `lrmc status` reuses."""
    summary = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
        "free_disk_gb": round(free_disk_bytes(".") / 1e9, 2),
    }
    nvidia_smi_gpus = gpu_status_list()
    if nvidia_smi_gpus:
        summary["gpus"] = nvidia_smi_gpus
    elif torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            summary["gpus"].append(
                {
                    "index": i,
                    "name": props.name,
                    "memory_total_mb": props.total_memory / 1e6,
                }
            )
    return summary
