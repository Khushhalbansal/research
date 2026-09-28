"""BIG 2015 loader: real File Preprocessor + Image Generator for ``.bytes`` files.

Responsibilities:

* Auto-discover ``.bytes`` files, either already extracted or packed in ``.7z``
  archives (streamed to disk with a soft ~20GB budget check, never held fully
  in memory).
* Join against ``trainLabels.csv`` / ``labels.csv`` -- only the labeled training
  portion (~10.8k samples, 9 families) has usable labels; anything unlabeled is
  reported but excluded from supervised splits.
* Cache every sample to a fixed-resolution uint8 memmap plus a JSON manifest
  (sha256, family, original size, unknown-byte fraction, cache row index),
  resumable across interruptions.

Byte-truncate/pad augmentation (see ``image_generator.py``) only applies when
reading directly from a ``.bytes`` file; the cache stores the *already
rendered* fixed-size grid (that's the point of caching), so samples served
from cache are treated as pre-rendered images for augmentation purposes --
documented in docs/DECISIONS.md.
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lrmc.data.image_generator import ImageGenerator
from lrmc.data.preprocessor import FilePreprocessor
from lrmc.data.splits import SampleRecord

logger = logging.getLogger(__name__)

EXPECTED_N_SAMPLES = 10868
EXPECTED_N_FAMILIES = 9
DISK_SAFETY_MARGIN_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
DEFAULT_DISK_BUDGET_BYTES = 20 * 1024 * 1024 * 1024  # 20 GB


@dataclass
class Big2015CacheEntry:
    sample_id: str
    family: str | None
    sha256: str
    original_size: int
    unknown_byte_frac: float
    cache_index: int


@dataclass
class Big2015Report:
    n_labeled: int
    n_unlabeled: int
    n_families: int
    family_counts: dict[str, int]
    mismatches: list[str]


def _resize_grid(grid: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(grid)
    im = im.resize((size, size), Image.BILINEAR)
    return np.array(im, dtype=np.uint8)


def _process_bytes_file(args: tuple[str, str, str | None, int]) -> dict:
    """Top-level (picklable) worker for multiprocessing: parse + render one file."""
    path_str, sample_id, family, size = args
    pre = FilePreprocessor()
    result = pre.load_bytes_file(path_str)
    gen = ImageGenerator(image_size=size)
    grid = gen.bytes_to_image(result.byte_stream)
    resized = _resize_grid(grid, size)
    return {
        "sample_id": sample_id,
        "family": family,
        "sha256": result.sha256,
        "original_size": result.original_size,
        "unknown_byte_frac": result.unknown_byte_frac,
        "image": resized,
    }


class Big2015Loader:
    def __init__(
        self,
        root: str | Path,
        cache_dir: str | Path,
        intermediate_size: int = 256,
        n_workers: int = 1,
        disk_budget_bytes: int = DEFAULT_DISK_BUDGET_BYTES,
    ):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_size = intermediate_size
        self.n_workers = max(1, n_workers)
        self.disk_budget_bytes = disk_budget_bytes

        self._extracted_dir = self.cache_dir / "extracted"
        self._memmap_path = self.cache_dir / "big2015_cache.dat"
        self._manifest_path = self.cache_dir / "big2015_manifest.json"

        self.records: list[SampleRecord] = []
        self.report: Big2015Report | None = None
        self._manifest: list[Big2015CacheEntry] = []
        self._memmap: np.memmap | None = None

    # -- discovery ---------------------------------------------------

    def _extract_archives(self) -> None:
        import py7zr

        archives = sorted(self.root.rglob("*.7z"))
        if not archives:
            return
        self._extracted_dir.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            free = shutil.disk_usage(self._extracted_dir).free
            if free < DISK_SAFETY_MARGIN_BYTES:
                logger.warning(
                    "Stopping 7z extraction: only %.2f GB free (< safety margin)", free / 1e9
                )
                break
            with py7zr.SevenZipFile(archive, mode="r") as z:
                z.extractall(path=self._extracted_dir)
            logger.info("Extracted %s", archive)

    def discover(self) -> list[tuple[str, str, str | None]]:
        """Return list of (path, sample_id, family|None) for every ``.bytes`` file."""
        self._extract_archives()
        search_roots = [self.root, self._extracted_dir]
        bytes_files: dict[str, Path] = {}
        for sr in search_roots:
            if not sr.exists():
                continue
            for p in sr.rglob("*.bytes"):
                bytes_files[p.stem] = p

        labels: dict[str, str] = {}
        for label_name in ("trainLabels.csv", "labels.csv"):
            for candidate in list(self.root.rglob(label_name)) + list(
                self._extracted_dir.rglob(label_name) if self._extracted_dir.exists() else []
            ):
                with open(candidate, newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        sid = row.get("Id") or row.get("id")
                        cls = row.get("Class") or row.get("class")
                        if sid is not None:
                            labels[sid] = cls
                break  # first match wins

        out = []
        n_unlabeled = 0
        for stem, path in sorted(bytes_files.items()):
            family = labels.get(stem)
            if family is None:
                n_unlabeled += 1
            out.append((str(path), stem, family))
        self._n_unlabeled_discovered = n_unlabeled
        return out

    # -- caching -------------------------------------------------------

    def build_cache(self, resume: bool = True) -> Big2015Report:
        discovered = self.discover()
        labeled = [(p, sid, fam) for p, sid, fam in discovered if fam is not None]
        n_unlabeled = len(discovered) - len(labeled)

        existing_manifest: list[Big2015CacheEntry] = []
        if resume and self._manifest_path.exists():
            with open(self._manifest_path, encoding="utf-8") as fh:
                existing_manifest = [Big2015CacheEntry(**e) for e in json.load(fh)]

        n_total = len(labeled)
        can_resume_in_place = (
            resume and self._memmap_path.exists() and len(existing_manifest) <= n_total
        )
        size = self.intermediate_size
        mode = "r+" if can_resume_in_place and self._memmap_path.exists() else "w+"
        memmap = np.memmap(
            self._memmap_path, dtype=np.uint8, mode=mode, shape=(max(n_total, 1), size, size)
        )

        manifest = list(existing_manifest) if can_resume_in_place else []
        already_done = {e.sample_id for e in manifest}
        todo = [(p, sid, fam, size) for p, sid, fam in labeled if sid not in already_done]

        if todo:
            next_index = len(manifest)
            if self.n_workers > 1:
                with mp.Pool(self.n_workers) as pool:
                    results = pool.imap(_process_bytes_file, todo, chunksize=8)
                    for res in results:
                        memmap[next_index] = res["image"]
                        manifest.append(
                            Big2015CacheEntry(
                                sample_id=res["sample_id"],
                                family=res["family"],
                                sha256=res["sha256"],
                                original_size=res["original_size"],
                                unknown_byte_frac=res["unknown_byte_frac"],
                                cache_index=next_index,
                            )
                        )
                        next_index += 1
            else:
                for args in todo:
                    res = _process_bytes_file(args)
                    memmap[next_index] = res["image"]
                    manifest.append(
                        Big2015CacheEntry(
                            sample_id=res["sample_id"],
                            family=res["family"],
                            sha256=res["sha256"],
                            original_size=res["original_size"],
                            unknown_byte_frac=res["unknown_byte_frac"],
                            cache_index=next_index,
                        )
                    )
                    next_index += 1
            memmap.flush()
            with open(self._manifest_path, "w", encoding="utf-8") as fh:
                json.dump([asdict(e) for e in manifest], fh, indent=2)

        self._manifest = manifest
        self._memmap = np.memmap(
            self._memmap_path, dtype=np.uint8, mode="r", shape=(max(n_total, 1), size, size)
        )
        self.records = [
            SampleRecord(sample_id=e.sample_id, family=e.family, sha256=e.sha256) for e in manifest
        ]

        family_counts: dict[str, int] = {}
        for e in manifest:
            family_counts[e.family] = family_counts.get(e.family, 0) + 1

        mismatches = []
        if len(manifest) != EXPECTED_N_SAMPLES:
            mismatches.append(
                f"expected {EXPECTED_N_SAMPLES} labeled samples, found {len(manifest)}"
            )
        if len(family_counts) != EXPECTED_N_FAMILIES:
            mismatches.append(
                f"expected {EXPECTED_N_FAMILIES} families, found {len(family_counts)}"
            )

        self.report = Big2015Report(
            n_labeled=len(manifest),
            n_unlabeled=n_unlabeled,
            n_families=len(family_counts),
            family_counts=family_counts,
            mismatches=mismatches,
        )
        if mismatches:
            logger.warning("BIG2015 dataset mismatch: %s", mismatches)
        return self.report

    def get_array(self, sample_id: str):
        """Returns (2D uint8 array from cache, is_pre_rendered_image=True)."""
        if self._memmap is None:
            raise RuntimeError("call build_cache() before get_array()")
        idx = next(e.cache_index for e in self._manifest if e.sample_id == sample_id)
        return np.array(self._memmap[idx]), True
