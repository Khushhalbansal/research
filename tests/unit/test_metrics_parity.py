import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from lrmc.eval.metrics import (
    bootstrap_ci,
    closed_set_metrics,
    known_false_rejection_rate,
    open_set_auroc_aupr,
    open_set_macro_f1,
    oscr_curve,
    paired_significance_test,
    unknown_detection_rate,
)


def test_closed_set_metrics_match_sklearn_directly():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=200)
    y_pred = rng.integers(0, 4, size=200)
    class_names = [f"c{i}" for i in range(4)]

    result = closed_set_metrics(y_true, y_pred, class_names)
    assert result["accuracy"] == accuracy_score(y_true, y_pred)
    assert abs(result["macro_f1"] - f1_score(y_true, y_pred, average="macro")) < 1e-9


def test_open_set_auroc_aupr_match_sklearn():
    rng = np.random.default_rng(1)
    y_unknown_true = rng.integers(0, 2, size=300)
    scores = rng.random(300) + y_unknown_true * 0.3  # correlated with label

    result = open_set_auroc_aupr(y_unknown_true, scores)
    assert abs(result["auroc"] - roc_auc_score(y_unknown_true, scores)) < 1e-9
    assert abs(result["aupr"] - average_precision_score(y_unknown_true, scores)) < 1e-9


def test_fpr_at_95tpr_is_in_valid_range():
    rng = np.random.default_rng(2)
    y_unknown_true = rng.integers(0, 2, size=500)
    scores = rng.random(500) + y_unknown_true * 0.5
    result = open_set_auroc_aupr(y_unknown_true, scores)
    assert 0.0 <= result["fpr_at_95tpr"] <= 1.0


def test_unknown_detection_and_known_frr_rates():
    y_unknown_true = np.array([0, 0, 1, 1, 1])
    verdict_is_zero_day = np.array([0, 1, 1, 1, 0])  # one false alarm on known, one miss on unknown
    udr = unknown_detection_rate(y_unknown_true, verdict_is_zero_day)
    frr = known_false_rejection_rate(y_unknown_true, verdict_is_zero_day)
    assert abs(udr - 2 / 3) < 1e-9
    assert abs(frr - 1 / 2) < 1e-9


def test_open_set_macro_f1_perfect_prediction_is_one():
    known = ["a", "b"]
    y_true = ["a", "a", "b", "UNKNOWN", "UNKNOWN"]
    y_pred = ["a", "a", "b", "UNKNOWN", "UNKNOWN"]
    f1 = open_set_macro_f1(y_true, y_pred, known)
    assert abs(f1 - 1.0) < 1e-9


def test_oscr_curve_perfect_separation_gives_auc_near_one():
    known_scores = np.random.default_rng(3).uniform(0.0, 0.4, size=100)
    known_correct = np.ones(100, dtype=bool)
    unknown_scores = np.random.default_rng(4).uniform(0.6, 1.0, size=100)
    result = oscr_curve(known_scores, known_correct, unknown_scores)
    assert result["oscr_auc"] > 0.95


def test_oscr_curve_no_separation_gives_lower_auc():
    rng = np.random.default_rng(5)
    known_scores = rng.uniform(0, 1, size=200)
    known_correct = np.ones(200, dtype=bool)
    unknown_scores = rng.uniform(0, 1, size=200)
    result = oscr_curve(known_scores, known_correct, unknown_scores)
    good = oscr_curve(
        np.random.default_rng(6).uniform(0, 0.3, size=200),
        np.ones(200, dtype=bool),
        np.random.default_rng(7).uniform(0.7, 1.0, size=200),
    )
    assert result["oscr_auc"] < good["oscr_auc"]


def test_bootstrap_ci_reasonable_bounds():
    values = np.array([0.8, 0.82, 0.79, 0.81, 0.83, 0.78])
    result = bootstrap_ci(values, n_boot=500, seed=0)
    assert result["ci_lo"] <= result["mean"] <= result["ci_hi"]
    assert abs(result["mean"] - values.mean()) < 1e-9


def test_paired_significance_test_identical_arrays():
    values = np.array([0.5, 0.6, 0.7, 0.8])
    result = paired_significance_test(values, values.copy())
    assert result["p_value"] == 1.0


def test_paired_significance_test_detects_difference():
    a = np.array([0.9, 0.92, 0.91, 0.93, 0.90, 0.94])
    b = np.array([0.7, 0.72, 0.71, 0.73, 0.70, 0.74])
    result = paired_significance_test(a, b)
    assert result["p_value"] < 0.05
