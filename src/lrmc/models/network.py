"""Feature Extraction Network: bundles Backbone + Projection Head + Embedding
Normalizer, since the diagram treats them as one contiguous flow used
identically in training and (frozen) inference."""

from __future__ import annotations

import torch.nn as nn

from lrmc.config import BackboneConfig, ProjectionHeadConfig
from lrmc.models.backbone import ViTBackbone
from lrmc.models.normalizer import EmbeddingNormalizer
from lrmc.models.projection_head import ProjectionHead


class FeatureExtractionNetwork(nn.Module):
    def __init__(self, backbone_cfg: BackboneConfig, head_cfg: ProjectionHeadConfig):
        super().__init__()
        self.backbone = ViTBackbone(backbone_cfg)
        self.head = ProjectionHead(
            in_dim=self.backbone.feature_dim,
            hidden_dim=head_cfg.hidden_dim,
            out_dim=head_cfg.out_dim,
        )
        self.normalizer = EmbeddingNormalizer()
        self.embed_dim = head_cfg.out_dim

    @property
    def pretrained_source(self) -> str:
        return self.backbone.pretrained_source

    def forward(self, x):
        h = self.backbone(x)
        z_tilde = self.head(h)
        z = self.normalizer(z_tilde)
        return z
