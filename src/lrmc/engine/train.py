"""Training loop: File Preprocessor/Image Generator -> Feature Extraction Network
-> LRMC Training Module, wired together with checkpointing and resume.

Only KNOWN-class data (``fold.train``) is ever touched here -- no test-set or
unknown-family data is used for fitting weights, prototypes, or radii.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lrmc.config import Config, config_hash
from lrmc.data.datasets import (
    ContrastiveMalwareDataset,
    build_label_map,
    class_balanced_sampler,
    contrastive_collate_fn,
)
from lrmc.data.image_generator import ImageGenerator
from lrmc.data.splits import Fold
from lrmc.engine.checkpoint import load_checkpoint, save_checkpoint
from lrmc.losses.lrmc_loss import LRMCLossCalculator
from lrmc.models.network import FeatureExtractionNetwork
from lrmc.models.prototypes import ClassPrototypeEstimator
from lrmc.models.radii import LearnableRadiiParameters
from lrmc.utils.env_info import env_snapshot
from lrmc.utils.seed import load_rng_state, rng_state, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: Config, loader, fold: Fold):
        self.cfg = cfg
        self.loader = loader
        self.fold = fold

        # Seeding here (not just once at process start) means constructing a
        # fresh Trainer always reproduces the SAME initial weights; resuming
        # then overwrites that initial state with the checkpoint and restores
        # the exact RNG stream, rather than relying on process-level ordering.
        set_seed(cfg.seed)

        self.label_map = build_label_map(loader.records, fold.known_families)
        self.num_classes = len(self.label_map)
        self.image_gen = ImageGenerator(
            image_size=cfg.data.image_size, augmentation=cfg.data.augmentation
        )

        self.network = FeatureExtractionNetwork(cfg.backbone, cfg.projection_head)
        self.prototypes = ClassPrototypeEstimator(
            self.num_classes,
            cfg.projection_head.out_dim,
            mode=cfg.prototypes.mode,
            momentum=cfg.prototypes.momentum,
        )
        self.radii = LearnableRadiiParameters(self.num_classes, r_init=cfg.radius.r_init)
        self.loss_calc = LRMCLossCalculator(
            temperature=cfg.loss.temperature,
            alpha=cfg.loss.alpha,
            beta=cfg.loss.beta,
            gamma=cfg.loss.gamma,
            margin=cfg.loss.margin,
            distance_metric=cfg.loss.distance_metric,
        )

        self.device = torch.device(
            cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.network.to(self.device)
        self.radii.to(self.device)
        self.prototypes.to(self.device)

        # DataParallel wraps ONLY the feature-extraction network (backbone+head),
        # never the loss. SupCon needs the FULL batch of negatives; if the loss
        # were computed per-GPU-replica on a DataParallel-split batch, each
        # replica would only see its own shard as negatives, silently
        # weakening the contrastive signal. So we parallelize feature
        # extraction, gather the full-batch embeddings back onto the primary
        # device, and compute the loss / prototype-radius updates there.
        self.is_data_parallel = self.device.type == "cuda" and torch.cuda.device_count() > 1
        if self.is_data_parallel:
            self.network = nn.DataParallel(self.network)

        main_params = list(self._bare_network().parameters())
        radii_params = list(self.radii.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": main_params, "lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
                {
                    "params": radii_params,
                    "lr": cfg.optim.lr / cfg.radius.lr_multiplier,
                    "weight_decay": 0.0,
                },
            ]
        )

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.pt"
        self.start_epoch = 0
        self.history: list[dict] = []

        self.train_dataset = ContrastiveMalwareDataset(
            loader, fold.train, self.image_gen, self.label_map
        )
        sampler = None
        if cfg.data.class_balanced_sampling:
            sampler = class_balanced_sampler(loader, fold.train, self.label_map)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.data.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg.data.num_workers,
            collate_fn=contrastive_collate_fn,
        )

    def _bare_network(self) -> FeatureExtractionNetwork:
        return self.network.module if isinstance(self.network, nn.DataParallel) else self.network

    def _forward_features(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        v1 = v1.to(self.device)
        v2 = v2.to(self.device)
        batch_size = v1.shape[0]
        both = torch.cat([v1, v2], dim=0)
        z_both = self.network(both)
        z1, z2 = z_both[:batch_size], z_both[batch_size:]
        return torch.stack([z1, z2], dim=1)  # (B, 2, D)

    def _resume_if_available(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        ckpt = load_checkpoint(self.checkpoint_path, map_location=self.device)
        if ckpt.get("config_hash") != config_hash(self.cfg):
            logger.warning(
                "Checkpoint config hash %s != current config hash %s; resuming anyway "
                "but this run is not a pure continuation of the same experiment.",
                ckpt.get("config_hash"),
                config_hash(self.cfg),
            )
        self._bare_network().load_state_dict(ckpt["network_state"])
        self.prototypes.load_state_dict(ckpt["prototypes_state"])
        self.radii.load_state_dict(ckpt["radii_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        load_rng_state(ckpt["rng_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.history = ckpt.get("history", [])
        logger.info("Resumed from checkpoint at epoch %d", ckpt["epoch"])
        return True

    def _save_checkpoint(self, epoch: int) -> None:
        payload = {
            "epoch": epoch,
            "network_state": self._bare_network().state_dict(),
            "prototypes_state": self.prototypes.state_dict(),
            "radii_state": self.radii.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "rng_state": rng_state(),
            "label_map": self.label_map,
            "config_hash": config_hash(self.cfg),
            "history": self.history,
            "pretrained_source": self._bare_network().pretrained_source,
        }
        save_checkpoint(self.checkpoint_path, payload)

    def _train_one_epoch(self) -> dict[str, float]:
        self.network.train()
        accum_steps = max(1, self.cfg.optim.grad_accumulation_steps)
        use_amp = self.device.type == "cuda"
        self.optimizer.zero_grad()

        totals = {"total": 0.0, "supcon": 0.0, "l_in": 0.0, "l_out": 0.0, "l_rad": 0.0}
        n_batches = 0
        for i, (v1, v2, labels) in enumerate(self.train_loader):
            labels = labels.to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                features = self._forward_features(v1, v2)
                prototypes_tensor = self.prototypes.get()
                radii_tensor = self.radii.radii
                out = self.loss_calc(features, labels, prototypes_tensor, radii_tensor)
                loss = out.total / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Prototype update happens AFTER the loss uses this step's
            # (pre-update) prototypes, using this step's fresh embeddings --
            # avoids an embedding chasing a target it just moved itself.
            z_flat = torch.cat([features[:, 0], features[:, 1]], dim=0).detach()
            y_flat = labels.repeat(2)
            self.prototypes.update(z_flat, y_flat)

            comp = out.as_dict()
            for k in totals:
                totals[k] += comp[k]
            n_batches += 1

        if n_batches % accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def fit(self) -> dict:
        self._resume_if_available()
        start_time = time.time()
        for epoch in range(self.start_epoch, self.cfg.optim.epochs):
            epoch_losses = self._train_one_epoch()
            radii_snapshot = self.radii.radii.detach().cpu().tolist()
            self.history.append({"epoch": epoch, **epoch_losses, "radii": radii_snapshot})
            is_last = epoch == self.cfg.optim.epochs - 1
            if (epoch + 1) % self.cfg.optim.checkpoint_every_n_epochs == 0 or is_last:
                self._save_checkpoint(epoch)
        elapsed = time.time() - start_time
        return self._final_metrics(elapsed)

    def _final_metrics(self, elapsed_seconds: float) -> dict:
        return {
            "run_name": self.cfg.run_name,
            "config_hash": config_hash(self.cfg),
            "seed": self.cfg.seed,
            "is_synthetic": self.cfg.is_synthetic,
            "device": str(self.device),
            "epochs_trained": self.cfg.optim.epochs,
            "known_families": self.fold.known_families,
            "unknown_families": self.fold.unknown_families,
            "history": self.history,
            "final_losses": self.history[-1] if self.history else {},
            "pretrained_source": self._bare_network().pretrained_source,
            "elapsed_seconds": elapsed_seconds,
            "env": env_snapshot(),
        }
