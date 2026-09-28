"""Capture the environment (git commit, package versions, hardware) for every run."""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version


def get_git_commit(cwd: str | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def get_git_dirty(cwd: str | None = None) -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True


_TRACKED_PACKAGES = [
    "torch",
    "torchvision",
    "timm",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "pillow",
    "matplotlib",
    "pyyaml",
    "py7zr",
]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def hardware_info() -> dict:
    import torch

    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or "unknown",
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    else:
        info["gpu_name"] = None
        info["gpu_count"] = 0
    return info


def env_snapshot(cwd: str | None = None) -> dict:
    return {
        "git_commit": get_git_commit(cwd),
        "git_dirty": get_git_dirty(cwd),
        "package_versions": package_versions(),
        "hardware": hardware_info(),
    }
