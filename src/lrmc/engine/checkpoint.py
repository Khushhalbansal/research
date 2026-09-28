"""Checkpoint save/load: model, optimizer, prototypes, radii, RNG state, epoch.

A single ``torch.save`` payload (dict) captures everything needed to resume
training bit-for-bit, including the RNG snapshot from :mod:`lrmc.utils.seed` --
see ``tests/integration/test_resume_equivalence.py`` for the proof that
kill-and-resume reproduces an uninterrupted run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    tmp_path = Path(str(path) + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)  # atomic-ish: never leave a half-written checkpoint


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
