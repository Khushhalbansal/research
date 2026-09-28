"""Lightweight status files so `lrmc status` can report progress without
tailing logs: Trainer writes one per epoch (current job's progress), and
QueueRunner writes one when it starts a job (which job is "current").

Writes are atomic (write to .tmp, then replace) so `lrmc status` never reads
a half-written file, which matters because it may be invoked at any moment
against a job that's actively running.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    tmp_path.replace(path)


def write_job_status(output_dir: str | Path, **fields: Any) -> None:
    path = Path(output_dir) / "status.json"
    _atomic_write_json(path, {"timestamp": time.time(), **fields})


def read_job_status(output_dir: str | Path) -> dict | None:
    path = Path(output_dir) / "status.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def write_current_job(runs_dir: str | Path, name: str, tier: str, job_type: str) -> None:
    path = Path(runs_dir) / "current_job.json"
    _atomic_write_json(
        path, {"name": name, "tier": tier, "type": job_type, "started_at": time.time()}
    )


def read_current_job(runs_dir: str | Path) -> dict | None:
    path = Path(runs_dir) / "current_job.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
