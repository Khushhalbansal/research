"""Publication-quality figures (PDF + PNG, colorblind-safe Okabe-Ito palette).

Every function here silently no-ops (logs and returns None) when its
required input is missing, rather than crash -- a run made before some field
was added, or a baseline that has no embeddings, must not take down the rest
of `aggregate`. Synthetic runs are filtered out up front, same as tables.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)

# Okabe & Ito (2008) colorblind-safe categorical palette.
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _filter_real(runs: list[dict]) -> list[dict]:
    return [r for r in runs if not r.get("is_synthetic", True)]


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path


def _load_npz(run: dict) -> dict | None:
    path = Path(run.get("_run_dir", "")) / "raw_eval_arrays.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def _load_history(run: dict) -> list[dict] | None:
    path = Path(run.get("_run_dir", "")) / "train_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("history")


def plot_training_curves(run: dict, out_dir: Path) -> Path | None:
    history = _load_history(run)
    if not history:
        return None
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, key in enumerate(["total", "supcon", "l_in", "l_out", "l_rad"]):
        vals = [h.get(key) for h in history]
        if any(v is None for v in vals):
            continue
        ax.plot(epochs, vals, label=key, color=PALETTE[i % len(PALETTE)], linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training curves: {run.get('run_name')}")
    ax.legend(frameon=False)
    return _save(fig, out_dir, f"training_curves_{run.get('run_name')}")


def plot_radii_over_epochs(run: dict, out_dir: Path) -> Path | None:
    history = _load_history(run)
    if not history or "radii" not in history[0]:
        return None
    epochs = [h["epoch"] for h in history]
    radii_matrix = np.array([h["radii"] for h in history])
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in range(radii_matrix.shape[1]):
        ax.plot(
            epochs, radii_matrix[:, c], color=PALETTE[c % len(PALETTE)], alpha=0.7, linewidth=1.2
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Radius $r_c$")
    ax.set_title(f"Learned radii over epochs: {run.get('run_name')}")
    return _save(fig, out_dir, f"radii_over_epochs_{run.get('run_name')}")


def plot_embedding_projection(run: dict, out_dir: Path) -> Path | None:
    """t-SNE projection of known/unknown test embeddings + prototypes.
    Note: this repo uses scikit-learn's t-SNE, not UMAP (avoids an extra
    heavy dependency for a CPU-only rehearsal today) -- documented in
    docs/DECISIONS.md; swap in umap-learn tomorrow if preferred."""
    data = _load_npz(run)
    if data is None or "known_embeddings" not in data or data["known_embeddings"].shape[0] < 4:
        return None
    from sklearn.manifold import TSNE

    known_emb = data["known_embeddings"]
    known_true = data["known_true"]
    unknown_emb = data.get("unknown_embeddings", np.zeros((0, known_emb.shape[1])))
    proto = data["prototypes"]

    combined = np.concatenate([known_emb, unknown_emb, proto], axis=0)
    perplexity = max(5, min(30, combined.shape[0] // 3))
    proj = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca").fit_transform(
        combined
    )

    n_known, n_unknown = known_emb.shape[0], unknown_emb.shape[0]
    known_proj = proj[:n_known]
    unknown_proj = proj[n_known : n_known + n_unknown]
    proto_proj = proj[n_known + n_unknown :]

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, fam in enumerate(sorted(set(known_true.tolist()))):
        mask = known_true == fam
        ax.scatter(
            known_proj[mask, 0],
            known_proj[mask, 1],
            s=12,
            color=PALETTE[i % len(PALETTE)],
            label=fam,
            alpha=0.7,
        )
    if n_unknown > 0:
        ax.scatter(
            unknown_proj[:, 0],
            unknown_proj[:, 1],
            s=14,
            color="gray",
            marker="x",
            label="unknown",
            alpha=0.7,
        )
    ax.scatter(
        proto_proj[:, 0], proto_proj[:, 1], s=140, color="black", marker="*", label="prototypes"
    )
    ax.legend(fontsize=6, frameon=False, loc="best", ncol=2)
    ax.set_title(f"t-SNE of embeddings: {run.get('run_name')}")
    return _save(fig, out_dir, f"embedding_tsne_{run.get('run_name')}")


def plot_radius_coverage(run: dict, out_dir: Path) -> Path | None:
    """2D PCA projection with a circle of radius r_c drawn around each
    projected prototype. The true acceptance region is a hyperspherical cap
    on the embedding sphere, not a disk in this 2D plane -- this plot is a
    QUALITATIVE coverage illustration, not an exact geometric one; noted in
    the figure title itself so it's never mistaken for the former."""
    data = _load_npz(run)
    if data is None or "known_embeddings" not in data or data["known_embeddings"].shape[0] < 4:
        return None
    from sklearn.decomposition import PCA

    known_emb = data["known_embeddings"]
    known_true = data["known_true"]
    proto = data["prototypes"]
    radii = data["radii"]
    families = list(data["families"])

    pca = PCA(n_components=2, random_state=0)
    combined = np.concatenate([known_emb, proto], axis=0)
    proj = pca.fit_transform(combined)
    known_proj = proj[: known_emb.shape[0]]
    proto_proj = proj[known_emb.shape[0] :]

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, fam in enumerate(sorted(set(known_true.tolist()))):
        mask = known_true == fam
        ax.scatter(
            known_proj[mask, 0],
            known_proj[mask, 1],
            s=10,
            color=PALETTE[i % len(PALETTE)],
            alpha=0.6,
        )
    for i, _fam in enumerate(families):
        if i >= len(proto_proj):
            continue
        circ = plt.Circle(
            proto_proj[i], radii[i], fill=False, color=PALETTE[i % len(PALETTE)], linewidth=1.2
        )
        ax.add_patch(circ)
        ax.scatter(
            *proto_proj[i], s=80, color=PALETTE[i % len(PALETTE)], marker="*", edgecolor="black"
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Radius coverage (qualitative, PCA plane): {run.get('run_name')}")
    return _save(fig, out_dir, f"radius_coverage_{run.get('run_name')}")


def plot_score_histograms(run: dict, out_dir: Path) -> Path | None:
    data = _load_npz(run)
    if data is None or "known_scores" not in data:
        return None
    known_scores = data["known_scores"]
    unknown_scores = data.get("unknown_scores", np.zeros(0))
    kappa = ((run.get("config") or {}).get("inference") or {}).get("kappa")

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(
        min(
            known_scores.min(), unknown_scores.min() if len(unknown_scores) else known_scores.min()
        ),
        max(
            known_scores.max(), unknown_scores.max() if len(unknown_scores) else known_scores.max()
        ),
        30,
    )
    ax.hist(known_scores, bins=bins, alpha=0.6, color=PALETTE[0], label="known")
    if len(unknown_scores) > 0:
        ax.hist(unknown_scores, bins=bins, alpha=0.6, color=PALETTE[5], label="unknown")
    if kappa is not None:
        ax.axvline(kappa, color="black", linestyle="--", linewidth=1.2, label=f"kappa={kappa:.2f}")
    ax.set_xlabel("Unknown-ness score s(x)")
    ax.set_ylabel("Count")
    ax.set_title(f"Known vs. unknown scores: {run.get('run_name')}")
    ax.legend(frameon=False)
    return _save(fig, out_dir, f"score_histogram_{run.get('run_name')}")


def plot_roc_curves(
    runs: list[dict], out_dir: Path, group_name: str = "roc_comparison"
) -> Path | None:
    """One ROC curve per method (each run that has raw known/unknown scores)."""
    plotted = False
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for i, run in enumerate(runs):
        data = _load_npz(run)
        if data is None or "known_scores" not in data:
            continue
        known_scores = data["known_scores"]
        unknown_scores = data.get("unknown_scores", np.zeros(0))
        if len(unknown_scores) == 0:
            continue
        y = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
        s = np.concatenate([known_scores, unknown_scores])
        fpr, tpr, _ = roc_curve(y, s)
        label = run.get("method") or run.get("run_name")
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], label=label, linewidth=1.5)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("FPR (known accepted as unknown)")
    ax.set_ylabel("TPR (unknown detected)")
    ax.set_title("ROC: known-vs-unknown detection")
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, out_dir, group_name)


def plot_confusion_matrix(run: dict, out_dir: Path) -> Path | None:
    cm = (run.get("closed_set") or {}).get("confusion_matrix")
    if not cm:
        return None
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {run.get('run_name')}")
    return _save(fig, out_dir, f"confusion_matrix_{run.get('run_name')}")


def plot_gamma_vs_frr(runs: list[dict], out_dir: Path) -> Path | None:
    """Tests the quantile-equilibrium claim (docs/LRMC.md sec. 5) on REAL
    results: config gamma vs. observed known false-rejection rate, with the
    y=x reference line the derivation predicts."""
    points = []
    for r in runs:
        if "ablation_gamma_" not in r.get("run_name", ""):
            continue
        gamma = ((r.get("config") or {}).get("loss") or {}).get("gamma")
        frr = (r.get("open_set") or {}).get("known_false_rejection_rate")
        if gamma is not None and frr is not None:
            points.append((gamma, frr))
    if len(points) < 2:
        return None
    points.sort()
    gammas, frrs = zip(*points, strict=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        [0, max(gammas + frrs)],
        [0, max(gammas + frrs)],
        color="gray",
        linestyle=":",
        label="y = x (predicted)",
    )
    ax.scatter(gammas, frrs, color=PALETTE[0], s=40, zorder=3, label="observed")
    ax.set_xlabel(r"Configured $\gamma/\alpha$ (target FRR)")
    ax.set_ylabel("Observed known false-rejection rate")
    ax.set_title("Gamma sweep vs. observed FRR")
    ax.legend(frameon=False)
    return _save(fig, out_dir, "gamma_vs_frr")


def plot_per_fold_bars(
    runs: list[dict], out_dir: Path, metric_path: str = "open_set.auroc"
) -> Path | None:
    def _get(d, path):
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    vals = [(r.get("run_name"), _get(r, metric_path)) for r in runs]
    vals = [(n, v) for n, v in vals if v is not None]
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(max(5, len(vals) * 0.6), 4))
    names, ys = zip(*vals, strict=True)
    ax.bar(range(len(ys)), ys, color=PALETTE[1])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(metric_path)
    ax.set_title(f"{metric_path} across runs")
    return _save(fig, out_dir, f"per_fold_bars_{metric_path.replace('.', '_')}")


def generate_all_figures(runs: list[dict], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    real = _filter_real(runs)
    if not real:
        logger.warning("No non-synthetic runs -- skipping ALL figure generation.")
        return []

    written: list[Path] = []
    for run in real:
        for fn in (
            plot_training_curves,
            plot_radii_over_epochs,
            plot_embedding_projection,
            plot_radius_coverage,
            plot_score_histograms,
            plot_confusion_matrix,
        ):
            try:
                p = fn(run, out_dir)
                if p:
                    written.append(p)
            except Exception as exc:  # noqa: BLE001 - one bad figure must not kill the rest
                logger.warning(
                    "Figure %s failed for run %s: %s", fn.__name__, run.get("run_name"), exc
                )

    try:
        p = plot_roc_curves(real, out_dir)
        if p:
            written.append(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROC comparison figure failed: %s", exc)

    try:
        p = plot_gamma_vs_frr(real, out_dir)
        if p:
            written.append(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gamma-vs-FRR figure failed: %s", exc)

    try:
        p = plot_per_fold_bars(real, out_dir)
        if p:
            written.append(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("per-fold bar figure failed: %s", exc)

    return written
