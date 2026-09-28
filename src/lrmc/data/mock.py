"""Synthetic data generators so the whole pipeline runs end-to-end on CPU today.

Two generators:

* :func:`generate_mock_malimg` -- class-per-folder grayscale PNGs, mimicking Malimg's
  real layout (including a couple of "variant family" groups so group-aware splitting
  has something real to hold out).
* :func:`generate_mock_big2015` -- ``.bytes`` hex-dump files plus a ``labels.csv``,
  mimicking the BIG 2015 layout.

Both produce class structure that is *separable* (each family has its own base
intensity pattern + per-sample noise), so a tiny model can actually learn something
meaningful in a few CPU seconds -- this is a rehearsal of the pipeline, not a claim
about real malware.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

# Mirrors real Malimg's variant-family groups (see docs/DECISIONS.md); mock family
# names are deliberately styled the same way so splits.py group logic gets exercised
# identically to how it will run against the real dataset tomorrow.
MOCK_MALIMG_GROUPS: dict[str, list[str]] = {
    "Allaple": ["Allaple.A", "Allaple.L"],
    "C2LOP": ["C2LOP.P", "C2LOP.gen!g"],
    "Lolyda.AA": ["Lolyda.AA1", "Lolyda.AA2", "Lolyda.AA3"],
    "Swizzor.gen": ["Swizzor.gen!C", "Swizzor.gen!E"],
}
_STANDALONE_FAMILIES = ["Fakerean", "Instantaccess", "Dontovo.A", "VB.AT", "Yuner.A"]


def _all_mock_malimg_families() -> list[str]:
    fams: list[str] = []
    for members in MOCK_MALIMG_GROUPS.values():
        fams.extend(members)
    fams.extend(_STANDALONE_FAMILIES)
    return fams


def generate_mock_malimg(
    root: str | Path,
    n_per_family: int = 12,
    image_size: int = 64,
    seed: int = 0,
    n_duplicates_per_family: int = 1,
) -> dict:
    """Write a Malimg-style class-per-folder PNG dataset. Returns a summary report."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    families = _all_mock_malimg_families()
    report = {"families": {}, "total_images": 0, "n_duplicates_written": 0}

    for fam_idx, family in enumerate(families):
        fam_dir = root / family
        fam_dir.mkdir(parents=True, exist_ok=True)
        # Each family gets a stable base pattern so samples are separable.
        base = rng.integers(30, 225, size=(image_size, image_size), dtype=np.int64)
        base = (base + fam_idx * 7) % 256
        n_written = 0
        first_sample_bytes = None
        for i in range(n_per_family):
            noise = rng.integers(-25, 25, size=(image_size, image_size))
            img = np.clip(base + noise, 0, 255).astype(np.uint8)
            fname = fam_dir / f"{family}_{i:04d}.png"
            Image.fromarray(img).save(fname)
            n_written += 1
            if i == 0:
                first_sample_bytes = fname.read_bytes()
        # write a few exact duplicates of the first sample, as real Malimg has.
        for d in range(n_duplicates_per_family):
            dup_name = fam_dir / f"{family}_dup{d:02d}.png"
            dup_name.write_bytes(first_sample_bytes)
            n_written += 1
            report["n_duplicates_written"] += 1
        report["families"][family] = n_written
        report["total_images"] += n_written
    return report


def generate_mock_big2015(
    root: str | Path,
    n_per_family: int = 12,
    seed: int = 0,
    n_families: int = 9,
    min_bytes: int = 2000,
    max_bytes: int = 12000,
    unknown_byte_frac: float = 0.05,
) -> dict:
    """Write BIG-2015-style ``.bytes`` files plus ``labels.csv``. Returns a summary."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "train").mkdir(parents=True, exist_ok=True)
    families = [f"Family{i}" for i in range(n_families)]
    rows = []
    report = {"families": {}, "total_files": 0}

    for fam_idx, family in enumerate(families):
        base_val = (fam_idx * 23 + 11) % 256
        n_written = 0
        for i in range(n_per_family):
            n_bytes = int(rng.integers(min_bytes, max_bytes))
            values = np.clip(base_val + rng.integers(-20, 20, size=n_bytes), 0, 255).astype(
                np.uint8
            )
            sample_id = hashlib.md5(f"{family}_{i}_{seed}".encode()).hexdigest()[:12]
            fpath = root / "train" / f"{sample_id}.bytes"
            _write_bytes_file(fpath, values, unknown_byte_frac, rng)
            rows.append([sample_id, family])
            n_written += 1
        report["families"][family] = n_written
        report["total_files"] += n_written

    with open(root / "labels.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Id", "Class"])
        writer.writerows(rows)
    return report


def _write_bytes_file(
    path: Path, values: np.ndarray, unknown_frac: float, rng: np.random.Generator
) -> None:
    """Write a byte array as a BIG-2015-style hex dump, with an address prefix per
    16-byte row and some `??` tokens to exercise the sanitized-byte handling path."""
    lines = []
    addr = 0
    for row_start in range(0, len(values), 16):
        row = values[row_start : row_start + 16]
        tokens = []
        for v in row:
            if rng.random() < unknown_frac:
                tokens.append("??")
            else:
                tokens.append(f"{int(v):02X}")
        lines.append(f"{addr:08X} " + " ".join(tokens))
        addr += 16
    path.write_text("\n".join(lines) + "\n")
