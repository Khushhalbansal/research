"""Dataclass-schema configs, YAML loading, and stable config hashing.

Every run in this repo is driven by one resolved :class:`Config` instance. The
resolved (defaults-applied) config is hashed with :func:`config_hash` and that hash
is written into every ``metrics.json`` alongside the seed, git commit and package
versions, so two runs are only "the same experiment" if their effective config
matches -- not merely their source YAML file.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackboneConfig:
    name: str = "tiny_test"  # "tiny_test" | "vit_tiny_patch16_224" | "resnet18"
    img_size: int = 64
    patch_size: int = 8
    depth: int = 2
    embed_dim: int = 64
    num_heads: int = 2
    mlp_ratio: float = 2.0
    pretrained: bool = False
    local_weights_path: str | None = None
    in_chans_mode: str = "replicate"  # "replicate" | "adapt_patch_embed"


@dataclass
class ProjectionHeadConfig:
    hidden_dim: int = 128
    out_dim: int = 64


@dataclass
class PrototypeConfig:
    mode: str = "ema"  # "ema" | "batch_mean"
    momentum: float = 0.9


@dataclass
class RadiusConfig:
    r_init: float = 0.5
    lr_multiplier: float = 10.0  # radii LR = base_lr / lr_multiplier (slower moving)


@dataclass
class LossConfig:
    temperature: float = 0.1
    alpha: float = 1.0  # weight on L_in (compactness)
    beta: float = 1.0  # weight on L_out (separation)
    gamma: float = 0.1  # weight on L_rad (tightness / target FRR)
    margin: float = 0.1  # margin used in L_out
    distance_metric: str = "cosine"  # "cosine" | "euclidean"


@dataclass
class AugmentationConfig:
    random_resized_crop_scale: tuple[float, float] = (0.7, 1.0)
    max_translate_frac: float = 0.1
    gaussian_noise_std: float = 0.03
    random_erasing_prob: float = 0.25
    random_erasing_scale: tuple[float, float] = (0.02, 0.15)
    byte_truncate_pad_prob: float = 0.3
    byte_truncate_pad_frac: float = 0.1


@dataclass
class DataConfig:
    dataset: str = "mock_malimg"  # "malimg" | "big2015" | "mock_malimg" | "mock_big2015"
    root: str = "data_cache/mock_malimg"
    cache_dir: str = "data_cache/cache"
    image_size: int = 64
    intermediate_size: int = 256
    batch_size: int = 16
    num_workers: int = 0
    class_balanced_sampling: bool = True
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class SplitConfig:
    protocol: str = (
        "random_k_unknown"  # "leave_one_family_out" | "random_k_unknown" | "group_aware"
    )
    n_unknown: int = 2
    seed: int = 0
    val_frac: float = 0.15
    test_frac: float = 0.2
    pseudo_unknown: bool = False
    pseudo_unknown_family: str | None = None
    group_aware: bool = False
    groups_file: str | None = None
    fold_index: int = 0  # which fold to use when protocol == "leave_one_family_out"


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 2
    warmup_epochs: int = 0
    grad_accumulation_steps: int = 1
    checkpoint_every_n_epochs: int = 1


@dataclass
class InferenceConfig:
    kappa: float = 1.0
    score: str = "ratio"  # "ratio" (d/r) | "min_distance"


@dataclass
class Config:
    run_name: str = "dev_run"
    seed: int = 0
    device: str = "cpu"
    is_synthetic: bool = False
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    projection_head: ProjectionHeadConfig = field(default_factory=ProjectionHeadConfig)
    prototypes: PrototypeConfig = field(default_factory=PrototypeConfig)
    radius: RadiusConfig = field(default_factory=RadiusConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output_dir: str = "runs/dev_run"
    extra: dict[str, Any] = field(default_factory=dict)


def _dataclass_from_dict(cls, data: dict) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict."""
    if data is None:
        return cls()
    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in fieldtypes:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = fieldtypes[key]
        nested_cls = _NESTED_DATACLASSES.get((cls, key))
        if nested_cls is not None and isinstance(value, dict):
            kwargs[key] = _dataclass_from_dict(nested_cls, value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value) if _is_tuple_field(ftype) else value
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _is_tuple_field(ftype: Any) -> bool:
    return "tuple" in str(ftype).lower()


_NESTED_DATACLASSES = {
    (Config, "backbone"): BackboneConfig,
    (Config, "projection_head"): ProjectionHeadConfig,
    (Config, "prototypes"): PrototypeConfig,
    (Config, "radius"): RadiusConfig,
    (Config, "loss"): LossConfig,
    (Config, "data"): DataConfig,
    (Config, "split"): SplitConfig,
    (Config, "optim"): OptimConfig,
    (Config, "inference"): InferenceConfig,
    (DataConfig, "augmentation"): AugmentationConfig,
}


def load_config(path: str | Path) -> Config:
    """Load a YAML config, applying dataclass defaults for anything unspecified."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _dataclass_from_dict(Config, raw)


def merge_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Apply dotted-path overrides, e.g. {'optim.lr': 0.01}, returning a new Config."""
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def config_hash(cfg: Config) -> str:
    """Stable sha256 hash of the fully-resolved config (sorted-key canonical JSON)."""
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
