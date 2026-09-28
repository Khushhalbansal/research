#!/usr/bin/env bash
# Zips the repo (code + configs + docs, NOT data/runs/caches) for upload as a
# Kaggle "Dataset" (Add Data -> Upload -> this zip), so the notebook can
# install it as a local package with no internet dependency beyond whatever
# pip wheels are already cached in the Kaggle image.
#
# Delegates the actual zipping to make_kaggle_bundle.py (pure stdlib
# zipfile), so this works even on a machine without a `zip` binary (e.g.
# plain Windows Git Bash) -- verified: this dev machine has no `zip`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-lrmc_kaggle_bundle.zip}"
python scripts/make_kaggle_bundle.py "$OUT"
