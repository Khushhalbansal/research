"""Environment detection: informational only, never load-bearing for dataset
paths. Every dataset path in this repo comes from config or a CLI flag (see
``lrmc.data.factory``) -- this module exists so scripts/logging can print a
sensible label and suggest a default working directory, not so anything
silently assumes it's running on a particular host.

The primary deployment target is a local workstation (physical GPU machine
reached over AnyDesk); Kaggle is kept as an optional profile
(`configs/base_malimg.yaml` etc. default to local-style relative paths, and
``notebooks/kaggle_run.py`` / ``experiments/queue.yaml``'s original
`/kaggle/...` paths still work if you explicitly point config at them).
"""

from __future__ import annotations

import os
import platform
from pathlib import Path


def detect_environment() -> str:
    """Returns "kaggle", "colab", or "local" (the default)."""
    if Path("/kaggle").exists():
        return "kaggle"
    if Path("/content").exists() and "COLAB_GPU" in os.environ:
        return "colab"
    return "local"


def default_working_dir(env: str | None = None) -> Path:
    """A sensible default working/output root for the detected environment.
    Purely a suggestion for scripts (e.g. where to write a results zip);
    never used to construct a dataset path -- those always come from config."""
    env = env or detect_environment()
    if env == "kaggle":
        return Path("/kaggle/working")
    return Path.cwd()


def os_name() -> str:
    """ "windows" | "linux" | "darwin" | whatever platform.system() lowercases to."""
    return platform.system().lower()


def is_windows() -> bool:
    return os_name() == "windows"


def env_summary() -> dict:
    return {
        "environment": detect_environment(),
        "os": os_name(),
        "python_version": platform.python_version(),
    }
