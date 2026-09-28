"""Vision Transformer Backbone (diagram box: "Vision Transformer Backbone").

Config presets:

* ``tiny_test`` -- a hand-sized ViT (img 64, patch 8, depth 2, embed_dim 64,
  2 heads) built directly from ``timm``'s ``VisionTransformer`` class, used
  for every local/CPU test and rehearsal so a full smoke run takes seconds.
* Any real ``timm`` model name (default preset: ``vit_tiny_patch16_224``).
* ``resnet18`` -- CNN ablation backbone, to justify choosing ViT over a CNN.

Pretrained-weight loading follows a robust three-step fallback (see
docs/DECISIONS.md): timm download -> local weights file -> random init with a
loud warning. Whichever path was actually used is recorded on
``self.pretrained_source`` so it ends up in every run's ``metrics.json``.
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from lrmc.config import BackboneConfig

logger = logging.getLogger(__name__)

PRETRAINED_SOURCE_TIMM = "timm_pretrained"
PRETRAINED_SOURCE_LOCAL = "local_weights"
PRETRAINED_SOURCE_RANDOM = "random_init"


class ViTBackbone(nn.Module):
    """Wraps either a from-scratch tiny_test ViT or a timm model (ViT/ResNet)."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.in_chans = 3 if cfg.in_chans_mode == "replicate" else 1
        self.pretrained_source = PRETRAINED_SOURCE_RANDOM
        self.model, self.feature_dim = self._build(cfg)

    def _build(self, cfg: BackboneConfig):
        if cfg.name == "tiny_test":
            model = VisionTransformer(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                in_chans=self.in_chans,
                num_classes=0,
                embed_dim=cfg.embed_dim,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
            )
            # tiny_test has no matching pretrained checkpoint by construction.
            self.pretrained_source = PRETRAINED_SOURCE_RANDOM
            return model, model.num_features
        model = self._load_with_fallback(cfg.name, cfg)
        return model, model.num_features

    def _load_with_fallback(self, timm_name: str, cfg: BackboneConfig) -> nn.Module:
        if cfg.pretrained:
            try:
                model = timm.create_model(
                    timm_name, pretrained=True, num_classes=0, in_chans=self.in_chans
                )
                self.pretrained_source = PRETRAINED_SOURCE_TIMM
                return model
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any
                # download/network failure must fall through to the next tier,
                # not crash the run (Kaggle may have internet off).
                logger.warning(
                    "timm pretrained download failed for '%s' (%s); trying local weights",
                    timm_name,
                    exc,
                )
            if cfg.local_weights_path:
                try:
                    model = timm.create_model(
                        timm_name, pretrained=False, num_classes=0, in_chans=self.in_chans
                    )
                    state = torch.load(cfg.local_weights_path, map_location="cpu")
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    logger.info(
                        "Loaded local weights from %s (missing=%d, unexpected=%d)",
                        cfg.local_weights_path,
                        len(missing),
                        len(unexpected),
                    )
                    self.pretrained_source = PRETRAINED_SOURCE_LOCAL
                    return model
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Local weights load failed (%s); falling back to RANDOM INIT", exc
                    )
            logger.warning(
                "No pretrained weights available for '%s' (offline and no local "
                "weights configured); using RANDOM INIT. This will hurt "
                "small-dataset performance -- see docs/DECISIONS.md.",
                timm_name,
            )
        model = timm.create_model(
            timm_name, pretrained=False, num_classes=0, in_chans=self.in_chans
        )
        self.pretrained_source = PRETRAINED_SOURCE_RANDOM
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_chans == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)
