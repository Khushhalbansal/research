"""Priority job runner over experiments/queue.yaml.

Executes tiers A -> B -> C in file order, skips jobs whose metrics.json
already exists (resumable across Kaggle sessions), and STOPS gracefully
(rather than skip-and-continue, which would break priority order) as soon as
the remaining session budget can no longer cover the next job's estimated
GPU-minutes. Every job failure is caught and logged so one bad job never
takes down the rest of the queue.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from lrmc.baselines.deep_svdd import DeepSVDDBaseline
from lrmc.baselines.fixed_radius_prototype import FixedRadiusPrototypeBaseline
from lrmc.baselines.knn import KNNBaseline
from lrmc.baselines.mahalanobis import MahalanobisBaseline
from lrmc.baselines.ocsvm import OCSVMBaseline
from lrmc.baselines.odin_energy import EnergyBaseline
from lrmc.baselines.openmax import OpenMaxBaseline
from lrmc.baselines.prototype_cosine import PrototypeCosineBaseline
from lrmc.baselines.softmax_msp import SoftmaxMSPBaseline
from lrmc.config import Config, load_config
from lrmc.data.datasets import EvalMalwareDataset, build_label_map, eval_collate_fn
from lrmc.data.factory import build_fold, build_loader
from lrmc.data.image_generator import ImageGenerator
from lrmc.data.splits import Fold
from lrmc.engine.infer import FrozenInferenceEngine
from lrmc.engine.train import Trainer
from lrmc.eval.evaluate import run_baseline_evaluation, run_evaluation, write_metrics_json

logger = logging.getLogger(__name__)

_EMBEDDING_BASELINES = {
    "prototype_cosine": PrototypeCosineBaseline,
    "fixed_radius_prototype": FixedRadiusPrototypeBaseline,
    "mahalanobis": MahalanobisBaseline,
    "knn": KNNBaseline,
}
_NETWORK_BASELINES = {
    "softmax_msp": SoftmaxMSPBaseline,
    "energy": EnergyBaseline,
    "openmax": OpenMaxBaseline,
    "deep_svdd": DeepSVDDBaseline,
}
# OCSVM is embedding-based but has a slightly different constructor signature
# (no separate "quantile"/"k" kwarg convention); handled explicitly below.


@dataclass
class JobResult:
    name: str
    tier: str
    status: str  # "completed" | "failed" | "skipped_done" | "skipped_budget" | "dry_run"
    elapsed_minutes: float | None = None
    estimated_gpu_minutes: float | None = None
    error: str | None = None


def _job_output_dir(job: dict) -> Path:
    return Path("runs") / job["name"]


def _build_eval_loaders(cfg: Config, loader, fold: Fold, label_map: dict[str, int]):
    gen = ImageGenerator(image_size=cfg.data.image_size)
    train_loader = DataLoader(
        EvalMalwareDataset(loader, fold.train, gen, label_map),
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    val_loader = DataLoader(
        EvalMalwareDataset(loader, fold.val_known, gen, label_map),
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    return gen, train_loader, val_loader


def _run_train_lrmc(job: dict) -> None:
    cfg = load_config(job["config"])
    # Pin output_dir to runs/<job name> (not whatever the YAML says) so a
    # dependent baseline_embedding job can find this checkpoint by job name,
    # and so re-running the queue's skip-completed check looks in the same
    # place this job wrote to.
    cfg.output_dir = str(_job_output_dir(job))
    loader = build_loader(cfg)
    fold = build_fold(cfg, loader)
    trainer = Trainer(cfg, loader, fold)
    train_metrics = trainer.fit()
    out_dir = Path(cfg.output_dir)
    write_metrics_json(train_metrics, out_dir / "train_metrics.json")
    metrics = run_evaluation(
        cfg, loader, fold, trainer.checkpoint_path, save_arrays_path=out_dir / "raw_eval_arrays.npz"
    )
    write_metrics_json(metrics, out_dir / "metrics.json")
    fold.to_json(out_dir / "fold.json")


def _run_baseline_network(job: dict) -> None:
    cfg = load_config(job["config"])
    loader = build_loader(cfg)
    fold = build_fold(cfg, loader)
    label_map = build_label_map(loader.records, fold.known_families)
    gen, train_loader, val_loader = _build_eval_loaders(cfg, loader, fold, label_map)

    baseline_name = job["baseline"]
    params = job.get("params", {})
    cls = _NETWORK_BASELINES[baseline_name]
    baseline = cls(cfg.backbone, label_map, device=cfg.device, **params)
    baseline.fit(train_loader)
    baseline.calibrate(val_loader)

    out_dir = _job_output_dir(job)
    metrics = run_baseline_evaluation(
        cfg,
        loader,
        fold,
        baseline,
        gen,
        method_name=baseline_name,
        save_arrays_path=out_dir / "raw_eval_arrays.npz",
    )
    write_metrics_json(metrics, out_dir / "metrics.json")


def _run_baseline_embedding(job: dict) -> None:
    cfg = load_config(job["config"])
    loader = build_loader(cfg)
    fold = build_fold(cfg, loader)

    depends_on = job["depends_on"]
    ckpt_path = Path("runs") / depends_on / "checkpoint.pt"
    if not ckpt_path.exists():
        raise RuntimeError(
            f"dependency job '{depends_on}' has not produced a checkpoint at {ckpt_path} yet"
        )
    engine = FrozenInferenceEngine(cfg, ckpt_path)
    gen, train_loader, val_loader = _build_eval_loaders(cfg, loader, fold, engine.label_map)

    baseline_name = job["baseline"]
    params = job.get("params", {})
    if baseline_name == "ocsvm":
        baseline = OCSVMBaseline(engine.network, engine.label_map, **params)
    else:
        cls = _EMBEDDING_BASELINES[baseline_name]
        baseline = cls(engine.network, engine.label_map, **params)
    baseline.fit(train_loader)
    baseline.calibrate(val_loader)

    out_dir = _job_output_dir(job)
    metrics = run_baseline_evaluation(
        cfg,
        loader,
        fold,
        baseline,
        gen,
        method_name=baseline_name,
        save_arrays_path=out_dir / "raw_eval_arrays.npz",
    )
    write_metrics_json(metrics, out_dir / "metrics.json")


_DISPATCH = {
    "train_lrmc": _run_train_lrmc,
    "baseline_network": _run_baseline_network,
    "baseline_embedding": _run_baseline_embedding,
}


class QueueRunner:
    def __init__(
        self,
        queue_path: str | Path,
        session_budget_minutes: float | None = None,
        dry_run: bool = False,
        runs_dir: str | Path = "runs",
    ):
        with open(queue_path, encoding="utf-8") as fh:
            self.queue = yaml.safe_load(fh)
        self.session_budget_minutes = (
            session_budget_minutes
            if session_budget_minutes is not None
            else self.queue.get("session_budget_minutes", 660)
        )
        self.dry_run = dry_run
        self.runs_dir = Path(runs_dir)
        self.start_time = time.time()

    def _elapsed_minutes(self) -> float:
        return (time.time() - self.start_time) / 60.0

    def run(self) -> list[JobResult]:
        results: list[JobResult] = []
        for tier_name, tier in self.queue.get("tiers", {}).items():
            for job in tier.get("jobs", []):
                metrics_path = self.runs_dir / job["name"] / "metrics.json"
                if metrics_path.exists():
                    results.append(JobResult(job["name"], tier_name, "skipped_done"))
                    continue

                est = float(job.get("estimated_gpu_minutes", 0))
                remaining = self.session_budget_minutes - self._elapsed_minutes()
                if remaining < est:
                    logger.warning(
                        "Stopping queue: %.1f min remaining < %.1f min estimated for job '%s'",
                        remaining,
                        est,
                        job["name"],
                    )
                    results.append(
                        JobResult(
                            job["name"], tier_name, "skipped_budget", estimated_gpu_minutes=est
                        )
                    )
                    self._save_state(results)
                    return results

                if self.dry_run:
                    results.append(
                        JobResult(job["name"], tier_name, "dry_run", estimated_gpu_minutes=est)
                    )
                    continue

                t0 = time.time()
                try:
                    _DISPATCH[job["type"]](job)
                    elapsed = (time.time() - t0) / 60.0
                    results.append(
                        JobResult(
                            job["name"],
                            tier_name,
                            "completed",
                            elapsed_minutes=elapsed,
                            estimated_gpu_minutes=est,
                        )
                    )
                    logger.info("Completed job '%s' in %.1f min", job["name"], elapsed)
                except Exception as exc:  # noqa: BLE001 - one bad job must not kill the queue
                    logger.error(
                        "Job '%s' failed: %s\n%s", job["name"], exc, traceback.format_exc()
                    )
                    results.append(
                        JobResult(
                            job["name"],
                            tier_name,
                            "failed",
                            error=str(exc),
                            estimated_gpu_minutes=est,
                        )
                    )
                self._save_state(results)
        self._save_state(results)
        return results

    def _save_state(self, results: list[JobResult]) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.runs_dir / "queue_state.json"
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
