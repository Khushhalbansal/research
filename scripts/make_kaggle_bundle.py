#!/usr/bin/env python
"""Portable implementation of make_kaggle_bundle.sh (pure stdlib zipfile, so
it works identically whether invoked from bash, PowerShell, or directly --
this machine doesn't have a `zip` binary, so the .sh wrapper delegates here).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = ["src", "configs", "experiments", "scripts", "notebooks", "docs", "paper"]
INCLUDE_FILES = ["pyproject.toml", "Makefile", "HANDOFF.md"]
EXCLUDE_SUFFIXES = {".pyc"}
EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def _should_include(path: Path) -> bool:
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
        return False
    return True


def main(out_name: str = "lrmc_kaggle_bundle.zip") -> None:
    out_path = REPO_ROOT / out_name
    if out_path.exists():
        out_path.unlink()

    n_files = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirname in INCLUDE_DIRS:
            base = REPO_ROOT / dirname
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and _should_include(path):
                    zf.write(path, path.relative_to(REPO_ROOT))
                    n_files += 1
        for filename in INCLUDE_FILES:
            path = REPO_ROOT / filename
            if path.exists():
                zf.write(path, path.relative_to(REPO_ROOT))
                n_files += 1

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({n_files} files, {size_mb:.2f} MB)")
    print("Excluded (by design): data_cache/, runs/, paper_artifacts/figures/*.png|pdf, .git/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lrmc_kaggle_bundle.zip")
