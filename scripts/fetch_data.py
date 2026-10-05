#!/usr/bin/env python
"""Downloads Malimg and BIG 2015 via the Kaggle API, with a free-disk
pre-check, size verification, and idempotent "skip if already fetched"
resumability.

Requires `pip install kaggle` (see requirements-dev.txt's `kaggle-fetch`
extra) and a kaggle.json API token: Kaggle account -> Settings -> API ->
"Create New Token", then place the downloaded kaggle.json at
``~/.kaggle/kaggle.json`` (Linux/Mac) or ``%USERPROFILE%\\.kaggle\\kaggle.json``
(Windows), or point ``KAGGLE_CONFIG_DIR`` at the folder that contains it.

Resumability note: Kaggle's public API does not support HTTP byte-range
resume for dataset/competition downloads (it always re-downloads the whole
zip). "Resumable" here means IDEMPOTENT: if a dataset's zip already exists
locally with the size Kaggle's API reports for it, the download is skipped
entirely rather than repeated -- the same "skip what's already done"
philosophy the rest of this repo uses (queue runner, BIG2015 cache builder).
A genuinely interrupted (partial) download is NOT resumed byte-for-byte; it
is detected as a size mismatch and re-downloaded from scratch.

Manual alternative (no Kaggle API / no internet on the workstation): copy
the data from a USB drive or network share into the destination folder
yourself, then run:
    python -m lrmc verify-data --dataset malimg --root <path>
    python -m lrmc verify-data --dataset big2015 --root <path>
which reproduces the same layout/count/duplicate report this script runs
automatically after downloading.

Usage:
    python scripts/fetch_data.py --dataset malimg   --dest data/malimg
    python scripts/fetch_data.py --dataset big2015  --dest data/big2015
    python scripts/fetch_data.py --dataset both
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Kaggle dataset/competition slugs. Malimg has several public mirrors; this
# is the most commonly used one as of writing -- override with --slug if it
# has moved or you prefer a different mirror.
DEFAULT_MALIMG_SLUG = "manmandes/malimg"
DEFAULT_BIG2015_SLUG = "malware-classification"

DISK_SAFETY_MULTIPLIER = 2.5  # zip + extracted + working headroom


def check_disk_budget(dest: Path, required_bytes: int) -> None:
    """Raises RuntimeError (does not start the download) if free disk space
    at ``dest`` is less than ``required_bytes``. Pure w.r.t. the filesystem
    query only -- easy to unit test by pointing at a tmp_path."""
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < required_bytes:
        raise RuntimeError(
            f"Refusing to start download: {required_bytes / 1e9:.1f} GB needed at "
            f"{dest}, only {free / 1e9:.1f} GB free. Free up space or point --dest "
            "at a drive with more room."
        )


def should_skip_download(zip_path: Path, expected_size_bytes: int | None) -> bool:
    """Pure decision function: skip re-downloading if a file already exists
    at zip_path and (when Kaggle reports a size) that size matches exactly."""
    if not zip_path.exists():
        return False
    if expected_size_bytes is None:
        return True  # no size to compare against; trust an existing file
    return zip_path.stat().st_size == expected_size_bytes


def _get_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is not installed. Run "
            "`pip install kaggle` (or `pip install -e .[kaggle-fetch]`), or use the "
            "manual-copy + `lrmc verify-data` path instead -- see this script's docstring."
        ) from exc
    api = KaggleApi()
    api.authenticate()  # reads kaggle.json
    return api


def _dataset_total_size(api, slug: str) -> int | None:
    try:
        files = api.dataset_list_files(slug).files
        return sum(int(f.total_bytes) for f in files)
    except Exception:  # noqa: BLE001 - size is advisory; missing it shouldn't block
        return None


def fetch_kaggle_dataset(slug: str, dest: Path, is_competition: bool = False) -> Path:
    """Downloads (or skips, if already present with a matching size) a
    Kaggle dataset/competition zip into dest, extracts it, and returns dest."""
    api = _get_kaggle_api()
    dest.mkdir(parents=True, exist_ok=True)

    if is_competition:
        expected_size = None  # competitions don't expose a simple file-size list
    else:
        expected_size = _dataset_total_size(api, slug)

    if expected_size is not None:
        check_disk_budget(dest, int(expected_size * DISK_SAFETY_MULTIPLIER))

    zip_name = f"{slug.split('/')[-1].replace(':', '_')}.zip"
    zip_path = dest / zip_name

    if should_skip_download(zip_path, expected_size):
        print(
            f"{zip_path} already present"
            + (" with matching size" if expected_size else "")
            + " -- skipping download."
        )
    else:
        print(f"Downloading {slug} to {dest} ...")
        if is_competition:
            api.competition_download_files(slug, path=str(dest), quiet=False)
            # competition downloads sometimes land under a different filename;
            # find whatever .zip appeared if our guessed name doesn't exist.
            if not zip_path.exists():
                zips = sorted(dest.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
                if zips:
                    zip_path = zips[0]
        else:
            api.dataset_download_files(slug, path=str(dest), quiet=False)
            if not zip_path.exists():
                zips = sorted(dest.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
                if zips:
                    zip_path = zips[0]

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=["malimg", "big2015", "both"], required=True)
    parser.add_argument("--dest", default=None, help="destination dir (default: data/<dataset>)")
    parser.add_argument("--malimg-slug", default=DEFAULT_MALIMG_SLUG)
    parser.add_argument("--big2015-slug", default=DEFAULT_BIG2015_SLUG)
    parser.add_argument(
        "--skip-verify", action="store_true", help="skip the post-download verify-data check"
    )
    args = parser.parse_args(argv)

    targets = ["malimg", "big2015"] if args.dataset == "both" else [args.dataset]

    for name in targets:
        dest = (
            Path(args.dest) if args.dest and args.dataset != "both" else REPO_ROOT / "data" / name
        )
        if name == "malimg":
            fetch_kaggle_dataset(args.malimg_slug, dest, is_competition=False)
        else:
            fetch_kaggle_dataset(args.big2015_slug, dest, is_competition=True)

        if not args.skip_verify:
            print(f"\nVerifying {name} layout ...")
            from lrmc.cli.main import main as lrmc_main

            try:
                lrmc_main(["verify-data", "--dataset", name, "--root", str(dest)])
            except SystemExit as exc:
                if exc.code:
                    print(
                        f"WARNING: {name} layout verification reported a mismatch "
                        "(see report above) -- inspect before spending GPU time."
                    )


if __name__ == "__main__":
    main()
