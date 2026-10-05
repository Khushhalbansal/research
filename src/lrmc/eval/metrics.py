"""Closed-set and open-set evaluation metrics, verified against sklearn in
tests/unit/test_metrics_parity.py.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def closed_set_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy, macro-F1, per-class report, confusion matrix -- known classes only."""
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def fpr_at_tpr(y_unknown_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """FPR at the operating point where TPR (unknown-detection recall) first
    reaches ``target_tpr``. Higher ``scores`` must mean "more likely unknown"."""
    fpr, tpr, _ = roc_curve(y_unknown_true, scores)
    idx = np.searchsorted(tpr, target_tpr)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def open_set_auroc_aupr(y_unknown_true: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC / AUPR for the known-vs-unknown binary decision. ``scores`` higher
    = more likely unknown; ``y_unknown_true`` is 1 for a true unknown sample."""
    return {
        "auroc": float(roc_auc_score(y_unknown_true, scores)),
        "aupr": float(average_precision_score(y_unknown_true, scores)),
        "fpr_at_95tpr": fpr_at_tpr(y_unknown_true, scores, 0.95),
    }


def unknown_detection_rate(y_unknown_true: np.ndarray, verdict_is_zero_day: np.ndarray) -> float:
    """Recall of the "unknown" class at the classifier's actual (kappa) threshold."""
    unknown_mask = y_unknown_true.astype(bool)
    if unknown_mask.sum() == 0:
        return float("nan")
    return float(verdict_is_zero_day[unknown_mask].mean())


def known_false_rejection_rate(
    y_unknown_true: np.ndarray, verdict_is_zero_day: np.ndarray
) -> float:
    """Fraction of KNOWN samples wrongly flagged zero-day, at the actual threshold."""
    known_mask = ~y_unknown_true.astype(bool)
    if known_mask.sum() == 0:
        return float("nan")
    return float(verdict_is_zero_day[known_mask].mean())


def open_set_macro_f1(
    y_true_family: list[str], y_pred_family: list[str], known_families: list[str]
) -> float:
    """Macro-F1 treating "UNKNOWN" as one extra class alongside the K known
    families (both true and predicted labels may be "UNKNOWN")."""
    classes = list(known_families) + ["UNKNOWN"]
    idx = {c: i for i, c in enumerate(classes)}
    y_true = [idx[c] for c in y_true_family]
    y_pred = [idx[c] for c in y_pred_family]
    return float(
        f1_score(y_true, y_pred, average="macro", labels=list(range(len(classes))), zero_division=0)
    )


def oscr_curve(
    known_scores: np.ndarray,
    known_correct: np.ndarray,
    unknown_scores: np.ndarray,
) -> dict:
    """Open Set Classification Rate curve (Dhamija et al., 2018 formulation).

    ``known_scores`` / ``unknown_scores``: the "unknown-ness" score s(x)
    (higher = more anomalous) for known-test and unknown-test samples.
    ``known_correct``: boolean array, whether each known sample's PREDICTED
    family equals its true family (independent of accept/reject).

    At threshold tau (a sample is "accepted as known" iff score <= tau):
      CCR(tau) = fraction of known samples accepted AND correctly classified
      FPR(tau) = fraction of unknown samples accepted (wrongly, as known)

    Returns the (fpr, ccr) points (sorted by fpr) and the trapezoidal AUC of
    the CCR-vs-FPR curve, ``oscr_auc``.
    """
    known_scores = np.asarray(known_scores, dtype=float)
    known_correct = np.asarray(known_correct, dtype=bool)
    unknown_scores = np.asarray(unknown_scores, dtype=float)
    n_known = len(known_scores)
    n_unknown = len(unknown_scores)

    all_taus = np.unique(np.concatenate([known_scores, unknown_scores, [np.inf]]))
    fprs, ccrs = [], []
    for tau in all_taus:
        accepted_known = known_scores <= tau
        ccr = float((accepted_known & known_correct).sum()) / max(n_known, 1)
        fpr = float((unknown_scores <= tau).sum()) / max(n_unknown, 1) if n_unknown > 0 else 0.0
        fprs.append(fpr)
        ccrs.append(ccr)

    order = np.argsort(fprs)
    fprs_sorted = np.array(fprs)[order]
    ccrs_sorted = np.array(ccrs)[order]
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trapezoid(ccrs_sorted, fprs_sorted))
    return {"fpr": fprs_sorted.tolist(), "ccr": ccrs_sorted.tolist(), "oscr_auc": auc}


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Mean +/- std and a bootstrap (1-alpha) CI for a metric measured across
    folds/seeds."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
    boot_means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    )
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": len(values),
    }


def paired_significance_test(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test between a method and the best baseline, paired
    by fold/seed. Falls back to a paired t-test if all differences are zero
    (Wilcoxon is undefined in that degenerate case)."""
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    diffs = values_a - values_b
    if np.allclose(diffs, 0):
        return {"test": "identical", "statistic": 0.0, "p_value": 1.0}
    try:
        stat, p = stats.wilcoxon(values_a, values_b)
        return {"test": "wilcoxon", "statistic": float(stat), "p_value": float(p)}
    except ValueError:
        stat, p = stats.ttest_rel(values_a, values_b)
        return {"test": "paired_t", "statistic": float(stat), "p_value": float(p)}


def aggregate_metric_across_runs(per_run_values: list[float]) -> dict:
    return bootstrap_ci(np.array(per_run_values))
