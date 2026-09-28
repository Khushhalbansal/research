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

from lrmc.config import Config, config_hash
from lrmc.data.image_generator import ImageGenerator
from lrmc.data.splits import Fold
from lrmc.eval.efficiency import efficiency_report
from lrmc.eval.metrics import (
    closed_set_metrics,
    known_false_rejection_rate,
    open_set_auroc_aupr,
    open_set_macro_f1,
    oscr_curve,
    unknown_detection_rate,
)
from lrmc.engine.infer import FrozenInferenceEngine
from lrmc.utils.env_info import env_snapshot


def _predict_all(engine: FrozenInferenceEngine, loader, sample_ids: list[str], image_gen: ImageGenerator):
    outputs = []
    family_by_id = {r.sample_id: r.family for r in loader.records}
    for sid in sample_ids:
        arr, is_image = loader.get_array(sid)
        image = image_gen.eval_view(arr, is_pre_rendered_image=is_image)
        out = engine.predict(image)
        outputs.append((sid, family_by_id[sid], out))
    return outputs


def run_evaluation(cfg: Config, loader, fold: Fold, checkpoint_path: str | Path) -> dict:
    start = time.time()
    engine = FrozenInferenceEngine(cfg, checkpoint_path)
    image_gen = ImageGenerator(image_size=cfg.data.image_size)
    label_map = engine.label_map
    families = engine.families

    known_outputs = _predict_all(engine, loader, fold.test_known, image_gen)
    unknown_outputs = _predict_all(engine, loader, fold.test_unknown, image_gen)

    # -- closed-set (known test samples, forced argmin classification) --
    y_true = np.array([label_map[fam] for _, fam, _ in known_outputs])
    y_pred = np.array([label_map[out.nearest_family] for _, _, out in known_outputs])
    closed = closed_set_metrics(y_true, y_pred, families)

    # -- open-set --
    known_scores = np.array([out.score for _, _, out in known_outputs])
    unknown_scores = np.array([out.score for _, _, out in unknown_outputs])
    all_scores = np.concatenate([known_scores, unknown_scores])
    y_unknown_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    open_metrics = open_set_auroc_aupr(y_unknown_true, all_scores) if len(unknown_scores) > 0 else {}

    verdict_is_zero_day = np.array(
        [1 if out.verdict == "ZERO_DAY" else 0 for _, _, out in known_outputs]
        + [1 if out.verdict == "ZERO_DAY" else 0 for _, _, out in unknown_outputs]
    )
    udr = unknown_detection_rate(y_unknown_true, verdict_is_zero_day) if len(unknown_scores) > 0 else float("nan")
    frr = known_false_rejection_rate(y_unknown_true, verdict_is_zero_day)

    known_correct = (y_true == y_pred)
    oscr = oscr_curve(known_scores, known_correct, unknown_scores) if len(unknown_scores) > 0 else {}

    y_true_family_all = [fam for _, fam, _ in known_outputs] + ["UNKNOWN"] * len(unknown_outputs)
    y_pred_family_all = [
        (out.nearest_family if out.verdict == "KNOWN" else "UNKNOWN") for _, _, out in known_outputs
    ] + [(out.nearest_family if out.verdict == "KNOWN" else "UNKNOWN") for _, _, out in unknown_outputs]
    open_f1 = open_set_macro_f1(y_true_family_all, y_pred_family_all, families)

    # -- efficiency --
    dummy_batch = torch.rand(1, 1, cfg.data.image_size, cfg.data.image_size)
    eff = efficiency_report(engine.network, dummy_batch, torch.device(cfg.device))

    elapsed = time.time() - start
    return {
        "run_name": cfg.run_name,
        "config_hash": config_hash(cfg),
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
        "open_set": {**open_metrics, "unknown_detection_rate": udr, "known_false_rejection_rate": frr},
        "oscr": oscr,
        "open_set_macro_f1": open_f1,
        "efficiency": eff,
        "model_hash": engine.model_hash,
        "pretrained_source": engine.network.pretrained_source,
        "elapsed_seconds": elapsed,
        "env": env_snapshot(),
    }


def write_metrics_json(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True, default=str)
