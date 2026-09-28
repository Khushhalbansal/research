"""Config -> (loader, Fold) factory shared by the CLI and the queue runner, so
"which dataset / which split protocol" is decided in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

from lrmc.config import Config
from lrmc.data.big2015 import Big2015Loader
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_big2015, generate_mock_malimg
from lrmc.data.splits import (
    Fold,
    group_aware_holdout,
    leave_one_family_out,
    load_family_groups,
    random_k_unknown,
)
from lrmc.utils.hardware import resolve_num_workers


def build_loader(cfg: Config):
    dataset = cfg.data.dataset
    if dataset == "malimg":
        return MalimgLoader(cfg.data.root)
    if dataset == "big2015":
        loader = Big2015Loader(
            cfg.data.root,
            cfg.data.cache_dir,
            intermediate_size=cfg.data.intermediate_size,
            n_workers=max(1, resolve_num_workers(cfg.data.num_workers)),
        )
        loader.build_cache()
        return loader
    if dataset == "mock_malimg":
        if not Path(cfg.data.root).exists():
            generate_mock_malimg(cfg.data.root, seed=cfg.seed)
        return MalimgLoader(cfg.data.root)
    if dataset == "mock_big2015":
        if not Path(cfg.data.root).exists():
            generate_mock_big2015(cfg.data.root, seed=cfg.seed)
        loader = Big2015Loader(
            cfg.data.root, cfg.data.cache_dir, intermediate_size=cfg.data.intermediate_size
        )
        loader.build_cache()
        return loader
    raise ValueError(f"unknown dataset '{dataset}'")


def build_fold(cfg: Config, loader) -> Fold:
    records = loader.records
    protocol = cfg.split.protocol
    if protocol == "random_k_unknown":
        return random_k_unknown(
            records,
            n_unknown=cfg.split.n_unknown,
            seed=cfg.split.seed,
            val_frac=cfg.split.val_frac,
            test_frac=cfg.split.test_frac,
            pseudo_unknown=cfg.split.pseudo_unknown,
        )
    if protocol == "group_aware_holdout":
        groups = load_family_groups(cfg.split.groups_file) if cfg.split.groups_file else {}
        return group_aware_holdout(
            records,
            groups,
            n_unknown_groups=cfg.split.n_unknown,
            seed=cfg.split.seed,
            val_frac=cfg.split.val_frac,
            test_frac=cfg.split.test_frac,
            pseudo_unknown=cfg.split.pseudo_unknown,
        )
    if protocol == "leave_one_family_out":
        folds = leave_one_family_out(
            records,
            val_frac=cfg.split.val_frac,
            test_frac=cfg.split.test_frac,
            seed=cfg.split.seed,
            pseudo_unknown=cfg.split.pseudo_unknown,
        )
        return folds[cfg.split.fold_index]
    raise ValueError(f"unknown split protocol '{protocol}'")
