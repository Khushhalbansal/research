"""LRMC Loss Calculator (diagram box: "LRMC Loss Calculator").

L = L_supcon + alpha * L_in + beta * L_out + gamma * L_rad

  d(z, p)   = 1 - <z, p>                                  (cosine; default)
            = 0.5 * ||z - p||^2                            (euclidean option)
  L_in      = mean_i max(0, d(z_i, p_{y_i}) - r_{y_i})            (compactness)
  L_out     = mean_i [ (1/(K-1)) sum_{c!=y_i} max(0, r_c + margin - d(z_i,p_c)) ]
                                                                    (separation)
  L_rad     = mean_c r_c                                            (tightness)

See docs/LRMC.md for the full derivation, including the empirically-verified
claim that at equilibrium r_c converges to roughly the (1 - gamma/alpha)-quantile
of in-class distances.

Gradients: prototypes ``p_c`` are treated as constants here (they are already a
stop-gradient buffer produced by :class:`~lrmc.models.prototypes.ClassPrototypeEstimator`);
radii ``r_c`` are differentiable (they come straight from
:class:`~lrmc.models.radii.LearnableRadiiParameters`) and gradients flow from
this loss back into ``rho_c`` as well as into the encoder/projection head via
``z``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from lrmc.losses.supcon import SupConLoss


def pairwise_distance(
    z: torch.Tensor, prototypes: torch.Tensor, metric: str = "cosine"
) -> torch.Tensor:
    """z: (N, D), prototypes: (K, D) -> distances (N, K)."""
    if metric == "cosine":
        return 1.0 - z @ prototypes.T
    elif metric == "euclidean":
        diff = z.unsqueeze(1) - prototypes.unsqueeze(0)  # (N, K, D)
        return 0.5 * (diff**2).sum(dim=-1)
    raise ValueError(f"unknown distance metric: {metric}")


def gathered_distance(z: torch.Tensor, p: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """Elementwise distance between matched z_i and p_i (both (N, D)) -> (N,)."""
    if metric == "cosine":
        return 1.0 - (z * p).sum(dim=-1)
    elif metric == "euclidean":
        return 0.5 * ((z - p) ** 2).sum(dim=-1)
    raise ValueError(f"unknown distance metric: {metric}")


@dataclass
class LRMCLossOutput:
    total: torch.Tensor
    supcon: torch.Tensor
    l_in: torch.Tensor
    l_out: torch.Tensor
    l_rad: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total.item(),
            "supcon": self.supcon.item(),
            "l_in": self.l_in.item(),
            "l_out": self.l_out.item(),
            "l_rad": self.l_rad.item(),
        }


class LRMCLossCalculator(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        margin: float = 0.1,
        distance_metric: str = "cosine",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.distance_metric = distance_metric
        self.supcon = SupConLoss(temperature=temperature)

    def forward(
        self,
        features: torch.Tensor,  # (B, n_views, D), L2-normalized
        labels: torch.Tensor,  # (B,) int64, known classes only (>= 0)
        prototypes: torch.Tensor,  # (K, D)
        radii: torch.Tensor,  # (K,)
    ) -> LRMCLossOutput:
        assert (labels >= 0).all(), "LRMC loss expects only known-class labels"
        device = features.device
        batch_size, n_views, _dim = features.shape

        supcon_loss = self.supcon(features, labels)

        z_flat = torch.cat(torch.unbind(features, dim=1), dim=0)  # (B*n_views, D)
        y_flat = labels.repeat(n_views)  # (B*n_views,)

        prototypes = prototypes.detach()  # already a stop-gradient buffer; defensive
        p_y = prototypes[y_flat]  # (B*n_views, D)
        r_y = radii[y_flat]  # (B*n_views,)

        d_in = gathered_distance(z_flat, p_y, self.distance_metric)
        l_in = torch.relu(d_in - r_y).mean()

        num_classes = prototypes.shape[0]
        d_all = pairwise_distance(z_flat, prototypes, self.distance_metric)  # (N, K)
        class_ids = torch.arange(num_classes, device=device).unsqueeze(0)  # (1, K)
        neg_mask = class_ids != y_flat.unsqueeze(1)  # (N, K)
        hinge = torch.relu(radii.unsqueeze(0) + self.margin - d_all)  # (N, K)
        hinge = hinge * neg_mask
        n_neg = neg_mask.sum(dim=1).clamp(min=1)
        l_out = (hinge.sum(dim=1) / n_neg).mean()

        l_rad = radii.mean()

        total = supcon_loss + self.alpha * l_in + self.beta * l_out + self.gamma * l_rad
        return LRMCLossOutput(total=total, supcon=supcon_loss, l_in=l_in, l_out=l_out, l_rad=l_rad)
