"""Full evaluation orchestration: loads a frozen checkpoint, runs it over a
fold's val/test splits, and produces the closed-set + open-set + efficiency
metrics bundle that becomes one run's ``metrics.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from lrmc.config import Config, config_hash, config_to_dict
from lrmc.data.image_generator import ImageGenerator
from lrmc.data.splits import Fold
from lrmc.engine.infer import FrozenInferenceEngine
from lrmc.eval.efficiency import efficiency_report
from lrmc.eval.metrics import (
    closed_set_metrics,
    known_false_rejection_rate,
    open_set_auroc_aupr,
    open_set_macro_f1,
    oscr_curve,
    unknown_detection_rate,
)
from lrmc.utils.env_info import env_snapshot
from lrmc.utils.hardware import resolve_device


def _predict_all(
    engine: FrozenInferenceEngine, loader, sample_ids: list[str], image_gen: ImageGenerator
):
    outputs = []
    embeddings = []
    family_by_id = {r.sample_id: r.family for r in loader.records}
    for sid in sample_ids:
        arr, is_image = loader.get_array(sid)
        image = image_gen.eval_view(arr, is_pre_rendered_image=is_image)
        out = engine.predict(image)
        z = engine.embed(image).squeeze(0).detach().cpu().numpy()
        outputs.append((sid, family_by_id[sid], out))
        embeddings.append(z)
    return outputs, np.stack(embeddings) if embeddings else np.zeros((0, 1))


def run_evaluation(
    cfg: Config,
    loader,
    fold: Fold,
    checkpoint_path: str | Path,
    save_arrays_path: str | Path | None = None,
) -> dict:
    """``save_arrays_path``, if given, additionally writes an .npz with raw
    per-sample scores/embeddings/prototypes/radii -- inputs the figures in
    ``aggregate/figures.py`` need (score histograms, embedding projections,
    radii-vs-epoch is read separately from train_metrics.json's history)."""
    start = time.time()
    engine = FrozenInferenceEngine(cfg, checkpoint_path)
    image_gen = ImageGenerator(image_size=cfg.data.image_size)
    families = engine.families

    # Calibrate the accept/reject threshold on KNOWN validation samples only
    # (same 95%-known-acceptance rule the baselines use) instead of the fixed
    # default kappa=1.0, which accepts nearly everything once radii grow.
    kappa = engine.classifier.kappa
    if getattr(fold, "val_known", None):
        val_outputs, _ = _predict_all(engine, loader, fold.val_known, image_gen)
        kappa = float(np.quantile([out.score for _, _, out in val_outputs], 0.95))
        engine.classifier.kappa = kappa

    known_outputs, known_embeddings = _predict_all(engine, loader, fold.test_known, image_gen)
    unknown_outputs, unknown_embeddings = _predict_all(engine, loader, fold.test_unknown, image_gen)

    known_true = [fam for _, fam, _ in known_outputs]
    known_pred = [out.nearest_family for _, _, out in known_outputs]
    known_scores = np.array([out.score for _, _, out in known_outputs])
    known_zd = np.array([1 if out.verdict == "ZERO_DAY" else 0 for _, _, out in known_outputs])

    unknown_pred = [out.nearest_family for _, _, out in unknown_outputs]
    unknown_scores = np.array([out.score for _, _, out in unknown_outputs])
    unknown_zd = np.array([1 if out.verdict == "ZERO_DAY" else 0 for _, _, out in unknown_outputs])

    dummy_batch = torch.rand(1, 1, cfg.data.image_size, cfg.data.image_size)
    eff = efficiency_report(engine.network, dummy_batch, resolve_device(cfg.device))

    if save_arrays_path is not None:
        Path(save_arrays_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_arrays_path,
            families=np.array(families),
            known_true=np.array(known_true),
            known_pred=np.array(known_pred),
            known_scores=known_scores,
            known_embeddings=known_embeddings,
            unknown_pred=np.array(unknown_pred),
            unknown_scores=unknown_scores,
            unknown_embeddings=unknown_embeddings,
            prototypes=engine.distance_calc.prototypes.detach().cpu().numpy(),
            radii=engine.classifier.radii.detach().cpu().numpy(),
        )

    return _assemble_metrics(
        cfg,
        fold,
        families,
        known_true,
        known_pred,
        known_scores,
        unknown_scores,
        known_zd,
        unknown_zd,
        unknown_pred,
        eff,
        model_hash=engine.model_hash,
        pretrained_source=engine.network.pretrained_source,
        start_time=start,
        extra={"kappa_calibrated": kappa},
    )


def _assemble_metrics(
    cfg: Config,
    fold: Fold,
    families: list[str],
    known_true: list[str],
    known_pred: list[str],
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_verdict_zero_day: np.ndarray,
    unknown_verdict_zero_day: np.ndarray,
    unknown_pred: list[str],
    efficiency: dict,
    model_hash: str,
    pretrained_source: str | None,
    start_time: float,
    extra: dict | None = None,
) -> dict:
    label_map = {f: i for i, f in enumerate(families)}
    y_true = np.array([label_map[f] for f in known_true])
    y_pred = np.array([label_map[f] for f in known_pred])
    closed = closed_set_metrics(y_true, y_pred, families)

    y_unknown_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    all_scores = np.concatenate([known_scores, unknown_scores])
    open_metrics = (
        open_set_auroc_aupr(y_unknown_true, all_scores) if len(unknown_scores) > 0 else {}
    )

    verdict_is_zero_day = np.concatenate([known_verdict_zero_day, unknown_verdict_zero_day])
    udr = (
        unknown_detection_rate(y_unknown_true, verdict_is_zero_day)
        if len(unknown_scores) > 0
        else float("nan")
    )
    frr = known_false_rejection_rate(y_unknown_true, verdict_is_zero_day)

    known_correct = y_true == y_pred
    oscr = (
        oscr_curve(known_scores, known_correct, unknown_scores) if len(unknown_scores) > 0 else {}
    )

    y_true_family_all = list(known_true) + ["UNKNOWN"] * len(unknown_pred)
    y_pred_family_all = [
        (p if not zd else "UNKNOWN")
        for p, zd in zip(known_pred, known_verdict_zero_day, strict=True)
    ] + [
        (p if not zd else "UNKNOWN")
        for p, zd in zip(unknown_pred, unknown_verdict_zero_day, strict=True)
    ]
    open_f1 = open_set_macro_f1(y_true_family_all, y_pred_family_all, families)

    result = {
        "run_name": cfg.run_name,
        "config_hash": config_hash(cfg),
        "config": config_to_dict(cfg),
        "seed": cfg.seed,
        "is_synthetic": cfg.is_synthetic,
        "known_families": fold.known_families,
        "unknown_families": fold.unknown_families,
        "protocol": fold.protocol,
        "fold_seed": fold.seed,
        "fold_index": fold.fold_index,
        "n_test_known": len(fold.test_known),
        "n_test_unknown": len(fold.test_unknown),
        "closed_set": closed,
        "open_set": {
            **open_metrics,
            "unknown_detection_rate": udr,
            "known_false_rejection_rate": frr,
        },
        "oscr": oscr,
        "open_set_macro_f1": open_f1,
        "efficiency": efficiency,
        "model_hash": model_hash,
        "pretrained_source": pretrained_source,
        "elapsed_seconds": time.time() - start_time,
        "env": env_snapshot(),
    }
    if extra:
        result.update(extra)
    return result


def run_baseline_evaluation(
    cfg: Config,
    loader,
    fold: Fold,
    baseline,
    image_gen: ImageGenerator,
    method_name: str,
    save_arrays_path: str | Path | None = None,
) -> dict:
    """Evaluates any BaselineDetector (fit/calibrate already called by the
    caller) with exactly the same metrics bundle as ``run_evaluation`` uses
    for LRMC, for a like-for-like comparison."""
    start = time.time()
    families = baseline.families
    family_by_id = {r.sample_id: r.family for r in loader.records}

    def _collect(sample_ids):
        trues, preds, scores, zero_day = [], [], [], []
        for sid in sample_ids:
            arr, is_image = loader.get_array(sid)
            image = image_gen.eval_view(arr, is_pre_rendered_image=is_image)
            trues.append(family_by_id[sid])
            preds.append(baseline.nearest_family(image))
            scores.append(baseline.score(image))
            zero_day.append(1 if baseline.predict(image) == "UNKNOWN" else 0)
        return trues, preds, np.array(scores), np.array(zero_day)

    known_true, known_pred, known_scores, known_zd = _collect(fold.test_known)
    _unk_true, unknown_pred, unknown_scores, unknown_zd = _collect(fold.test_unknown)

    dummy = torch.rand(1, 1, cfg.data.image_size, cfg.data.image_size)
    encoder = (
        getattr(baseline, "encoder", None)
        or getattr(baseline, "net", None)
        or getattr(baseline, "backbone", None)
    )
    eff = efficiency_report(encoder, dummy, resolve_device(cfg.device)) if encoder is not None else {}

    if save_arrays_path is not None:
        Path(save_arrays_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_arrays_path,
            families=np.array(families),
            known_true=np.array(known_true),
            known_pred=np.array(known_pred),
            known_scores=known_scores,
            unknown_pred=np.array(unknown_pred),
            unknown_scores=unknown_scores,
        )

    return _assemble_metrics(
        cfg,
        fold,
        families,
        known_true,
        known_pred,
        known_scores,
        unknown_scores,
        known_zd,
        unknown_zd,
        unknown_pred,
        eff,
        model_hash=f"baseline:{method_name}:{config_hash(cfg)}",
        pretrained_source=None,
        start_time=start,
        extra={"method": method_name},
    )


def write_metrics_json(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True, default=str)
