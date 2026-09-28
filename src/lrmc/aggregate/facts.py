"""paper_facts.json: dataset counts, splits, hyperparameters, package
versions, hardware, and timing for every REAL run -- so the paper's
Experimental Setup section can be written from facts, not memory.
"""

from __future__ import annotations

import json
from pathlib import Path


def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def write_paper_facts(runs: list[dict], out_path: str | Path) -> None:
    real = [r for r in runs if not r.get("is_synthetic", True)]
    facts = {
        "n_runs_total": len(runs),
        "n_runs_real": len(real),
        "n_runs_synthetic_excluded": len(runs) - len(real),
        "runs": [
            {
                "run_name": r.get("run_name") or r.get("_run_name"),
                "method": r.get("method"),
                "config_hash": r.get("config_hash"),
                "protocol": r.get("protocol"),
                "known_families": r.get("known_families"),
                "unknown_families": r.get("unknown_families"),
                "n_known_families": len(r.get("known_families") or []),
                "n_unknown_families": len(r.get("unknown_families") or []),
                "n_test_known": r.get("n_test_known"),
                "n_test_unknown": r.get("n_test_unknown"),
                "hyperparameters": r.get("config"),
                "pretrained_source": r.get("pretrained_source"),
                "package_versions": _get(r, "env.package_versions"),
                "hardware": _get(r, "env.hardware"),
                "git_commit": _get(r, "env.git_commit"),
                "git_dirty": _get(r, "env.git_dirty"),
                "elapsed_seconds": r.get("elapsed_seconds"),
            }
            for r in real
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=2, sort_keys=True, default=str)
