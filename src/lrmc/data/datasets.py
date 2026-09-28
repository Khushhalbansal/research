"""torch Dataset wrappers tying a family loader (Malimg/BIG2015/mock) + a
:class:`~lrmc.data.splits.Fold` + :class:`~lrmc.data.image_generator.ImageGenerator`
together, plus class-balanced sampling / class-weighting utilities.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from lrmc.data.image_generator import ImageGenerator
from lrmc.data.splits import SampleRecord


def build_label_map(records: list[SampleRecord], known_families: list[str]) -> dict[str, int]:
    """Stable family -> integer-label map, sorted so it's reproducible across runs."""
    return {fam: i for i, fam in enumerate(sorted(known_families))}


class ContrastiveMalwareDataset(Dataset):
    """Training-time dataset: yields two augmented views + integer label."""

    def __init__(
        self,
        loader,
        sample_ids: list[str],
        image_gen: ImageGenerator,
        label_map: dict[str, int],
    ):
        self.loader = loader
        self.sample_ids = sample_ids
        self.image_gen = image_gen
        self.label_map = label_map
        self._family_by_id = {r.sample_id: r.family for r in loader.records}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]
        arr, is_image = self.loader.get_array(sid)
        family = self._family_by_id[sid]
        label = self.label_map[family]
        v1, v2 = self.image_gen.two_views(arr, is_pre_rendered_image=is_image)
        return v1, v2, label


class EvalMalwareDataset(Dataset):
    """Eval-time dataset: single deterministic view + integer label (-1 = unknown)."""

    def __init__(
        self,
        loader,
        sample_ids: list[str],
        image_gen: ImageGenerator,
        label_map: dict[str, int],
        unknown_label: int = -1,
    ):
        self.loader = loader
        self.sample_ids = sample_ids
        self.image_gen = image_gen
        self.label_map = label_map
        self.unknown_label = unknown_label
        self._family_by_id = {r.sample_id: r.family for r in loader.records}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]
        arr, is_image = self.loader.get_array(sid)
        family = self._family_by_id[sid]
        label = self.label_map.get(family, self.unknown_label)
        v = self.image_gen.eval_view(arr, is_pre_rendered_image=is_image)
        return v, label, sid


def class_balanced_sampler(
    loader, sample_ids: list[str], label_map: dict[str, int]
) -> WeightedRandomSampler:
    """Inverse-frequency sampling weights so rare families are seen as often as
    common ones within a contrastive batch (used ONLY for the training loader;
    prototype/radius estimation must run on the natural distribution -- see
    docs/DECISIONS.md)."""
    family_by_id = {r.sample_id: r.family for r in loader.records}
    families = [family_by_id[sid] for sid in sample_ids]
    counts = Counter(families)
    weights = [1.0 / counts[f] for f in families]
    return WeightedRandomSampler(weights, num_samples=len(sample_ids), replacement=True)


def compute_class_weights(loader, sample_ids: list[str], label_map: dict[str, int]) -> torch.Tensor:
    """Inverse-frequency class weights, indexed by label id, for weighted CE losses."""
    family_by_id = {r.sample_id: r.family for r in loader.records}
    counts = Counter(family_by_id[sid] for sid in sample_ids)
    n_classes = len(label_map)
    weights = np.ones(n_classes, dtype=np.float32)
    for fam, idx in label_map.items():
        c = counts.get(fam, 0)
        weights[idx] = 1.0 / c if c > 0 else 0.0
    weights = weights * (n_classes / weights.sum()) if weights.sum() > 0 else weights
    return torch.from_numpy(weights)


def contrastive_collate_fn(batch):
    v1s, v2s, labels = zip(*batch, strict=True)
    return torch.stack(v1s), torch.stack(v2s), torch.tensor(labels, dtype=torch.long)


def eval_collate_fn(batch):
    vs, labels, ids = zip(*batch, strict=True)
    return torch.stack(vs), torch.tensor(labels, dtype=torch.long), list(ids)
