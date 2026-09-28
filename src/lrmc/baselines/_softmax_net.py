"""Shared internal helper: a plain softmax classifier (backbone + linear head)
trained with cross-entropy, reused by the MSP, energy/ODIN, and OpenMax
baselines. Not a baseline itself (no fit/score/predict) -- each baseline
class below still owns and trains its own independent instance.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lrmc.config import BackboneConfig
from lrmc.models.backbone import ViTBackbone


class SoftmaxNet(nn.Module):
    def __init__(self, backbone_cfg: BackboneConfig, num_classes: int):
        super().__init__()
        self.backbone = ViTBackbone(backbone_cfg)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits, feat


def train_softmax_net(
    backbone_cfg: BackboneConfig,
    num_classes: int,
    train_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> SoftmaxNet:
    net = SoftmaxNet(backbone_cfg, num_classes).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    net.train()
    for _epoch in range(epochs):
        for images, labels, _ids in train_loader:
            optimizer.zero_grad()
            logits, _feat = net(images.to(device))
            loss = ce(logits, labels.to(device))
            loss.backward()
            optimizer.step()
    net.eval()
    return net
