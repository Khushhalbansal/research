"""Supervised Contrastive Loss (Khosla et al., 2020).

L_supcon = sum_i (-1/|P(i)|) sum_{p in P(i)} log( exp(z_i . z_p / tau) /
sum_{a != i} exp(z_i . z_a / tau) )

``P(i)`` is every other sample in the (multi-view) batch sharing i's label;
with two augmented views per sample every anchor has at least one positive
(its own other view), so this reduces cleanly even for singleton classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """features: (B, n_views, D) L2-normalized. labels: (B,) int64."""
        device = features.device
        batch_size, n_views, dim = features.shape

        # [view0_sample0..B, view1_sample0..B, ...]
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)  # (B*n_views, D)
        flat_labels = labels.repeat(n_views)  # (B*n_views,)

        anchor_dot_contrast = (
            torch.matmul(contrast_features, contrast_features.T) / self.temperature
        )
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        n = batch_size * n_views
        label_eq = flat_labels.unsqueeze(0) == flat_labels.unsqueeze(1)  # (n, n)
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = label_eq & ~self_mask
        logits_mask = ~self_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = positive_mask.sum(dim=1).clamp(min=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / pos_count

        loss = -mean_log_prob_pos.mean()
        return loss
