import torch
from torch.utils.data import DataLoader

from lrmc.config import AugmentationConfig
from lrmc.data.datasets import (
    ContrastiveMalwareDataset,
    EvalMalwareDataset,
    build_label_map,
    class_balanced_sampler,
    compute_class_weights,
    contrastive_collate_fn,
    eval_collate_fn,
)
from lrmc.data.image_generator import ImageGenerator
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import assert_fold_disjoint, random_k_unknown


def test_end_to_end_mock_malimg_pipeline(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=48, seed=0)
    loader = MalimgLoader(root)
    assert loader.report.mismatches  # mock is intentionally smaller than real Malimg

    fold = random_k_unknown(loader.records, n_unknown=2, seed=0, val_frac=0.15, test_frac=0.2)
    assert_fold_disjoint(fold, loader.records)

    label_map = build_label_map(loader.records, fold.known_families)
    gen = ImageGenerator(image_size=32, augmentation=AugmentationConfig())

    train_ds = ContrastiveMalwareDataset(loader, fold.train, gen, label_map)
    val_ds = EvalMalwareDataset(loader, fold.val_known, gen, label_map)
    test_unknown_ds = EvalMalwareDataset(loader, fold.test_unknown, gen, label_map)

    sampler = class_balanced_sampler(loader, fold.train, label_map)
    train_loader = DataLoader(
        train_ds, batch_size=4, sampler=sampler, collate_fn=contrastive_collate_fn
    )
    v1, v2, labels = next(iter(train_loader))
    assert v1.shape == (4, 1, 32, 32)
    assert v2.shape == (4, 1, 32, 32)
    assert labels.dtype == torch.long

    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=eval_collate_fn)
    v, labels, ids = next(iter(val_loader))
    assert v.shape[1:] == (1, 32, 32)
    assert all(lbl.item() in label_map.values() for lbl in labels)

    unk_loader = DataLoader(test_unknown_ds, batch_size=4, collate_fn=eval_collate_fn)
    v, labels, ids = next(iter(unk_loader))
    assert all(lbl.item() == -1 for lbl in labels)  # unknown families map to -1

    weights = compute_class_weights(loader, fold.train, label_map)
    assert weights.shape[0] == len(label_map)
    assert (weights >= 0).all()
