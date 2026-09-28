"""Gathers every run's metrics.json under runs/ into a flat list. Returns
EVERYTHING (real and synthetic) -- individual table/figure generators are
responsible for filtering out ``is_synthetic: true`` runs, so the exclusion
is visible and countable at the call site (see cli.main.cmd_aggregate).
"""

from __future__ import annotations

import json
from pathlib import Path


def collect_runs(runs_dir: str | Path) -> list[dict]:
    runs_dir = Path(runs_dir)
    out = []
    if not runs_dir.exists():
        return out
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                metrics = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        metrics.setdefault("is_synthetic", True)  # fail safe: unlabeled -> treat as synthetic
        metrics["_run_dir"] = str(metrics_path.parent)
        metrics["_run_name"] = metrics_path.parent.name
        out.append(metrics)
    return out
