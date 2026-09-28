"""Malimg loader: auto-detects a class-per-folder PNG layout, dedups by hash, and
reports class imbalance / any mismatch against the known Malimg counts (25
families, 9,339 images)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from lrmc.data.preprocessor import FilePreprocessor
from lrmc.data.splits import SampleRecord

logger = logging.getLogger(__name__)

EXPECTED_N_FAMILIES = 25
EXPECTED_N_IMAGES = 9339
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


@dataclass
class MalimgReport:
    n_families: int
    n_images: int
    n_duplicate_hashes: int
    n_duplicate_files: int
    family_counts: dict[str, int]
    mismatches: list[str]
    max_to_min_ratio: float


class MalimgLoader:
    """Scans a Malimg-style directory tree and exposes ``SampleRecord``s.

    Layout auto-detection: any directory under ``root`` whose immediate children
    are image files is treated as one family folder; the family name is the
    folder's basename. This tolerates both ``root/<family>/*.png`` and a single
    extra nesting level such as ``root/malimg_dataset/<family>/*.png``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._pre = FilePreprocessor()
        self.records: list[SampleRecord] = []
        self._path_by_id: dict[str, Path] = {}
        self.report: MalimgReport | None = None
        self._scan()

    def _find_family_dirs(self) -> list[Path]:
        candidates = []
        for d in sorted(self.root.rglob("*")):
            if not d.is_dir():
                continue
            children = [c for c in d.iterdir() if c.is_file()]
            images = [c for c in children if c.suffix.lower() in _IMAGE_EXTS]
            if images and len(images) == len(children):
                candidates.append(d)
        if not candidates:
            # root itself may directly contain family folders with no images at
            # this level; fall back to immediate subdirectories of root.
            for d in sorted(self.root.iterdir()):
                if d.is_dir():
                    candidates.append(d)
        return candidates

    def _scan(self) -> None:
        family_dirs = self._find_family_dirs()
        hash_to_ids: dict[str, list[str]] = {}
        family_counts: dict[str, int] = {}

        for fam_dir in family_dirs:
            family = fam_dir.name
            images = sorted(
                p for p in fam_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
            )
            if not images:
                continue
            count = 0
            for img_path in images:
                result = self._pre.load_image_array(img_path)
                sample_id = f"{family}/{img_path.name}"
                rec = SampleRecord(
                    sample_id=sample_id, family=family, sha256=result.sha256, path=str(img_path)
                )
                self.records.append(rec)
                self._path_by_id[sample_id] = img_path
                hash_to_ids.setdefault(result.sha256, []).append(sample_id)
                count += 1
            family_counts[family] = count

        n_dup_hashes = sum(1 for ids in hash_to_ids.values() if len(ids) > 1)
        n_dup_files = sum(len(ids) - 1 for ids in hash_to_ids.values() if len(ids) > 1)

        mismatches = []
        n_families = len(family_counts)
        n_images = sum(family_counts.values())
        if n_families != EXPECTED_N_FAMILIES:
            mismatches.append(
                f"expected {EXPECTED_N_FAMILIES} families, found {n_families}"
            )
        if n_images != EXPECTED_N_IMAGES:
            mismatches.append(f"expected {EXPECTED_N_IMAGES} images, found {n_images}")

        ratio = (max(family_counts.values()) / min(family_counts.values())) if family_counts else 0.0

        self.report = MalimgReport(
            n_families=n_families,
            n_images=n_images,
            n_duplicate_hashes=n_dup_hashes,
            n_duplicate_files=n_dup_files,
            family_counts=family_counts,
            mismatches=mismatches,
            max_to_min_ratio=ratio,
        )
        if mismatches:
            logger.warning("Malimg layout mismatch at %s: %s", self.root, mismatches)

    def get_array(self, sample_id: str):
        """Returns (2D uint8 array, is_pre_rendered_image=True)."""
        path = self._path_by_id[sample_id]
        result = self._pre.load_image_array(path)
        return result.byte_stream, True
