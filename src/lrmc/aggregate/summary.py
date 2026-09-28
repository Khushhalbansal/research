"""results_summary.md: plain-language findings, written directly from the
collected real runs -- including honest negative results and failure modes
(e.g. a baseline beating LRMC on some metric, or an ablation showing no
effect) rather than only reporting favorable comparisons.
"""

from __future__ import annotations

from pathlib import Path


def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _method_name(m: dict) -> str:
    return m.get("method") or m.get("run_name", "unknown")


def write_results_summary(runs: list[dict], out_path: str | Path) -> None:
    real = [r for r in runs if not r.get("is_synthetic", True)]
    lines = ["# Results summary", ""]

    if not real:
        lines += [
            "No non-synthetic runs are available yet -- every run collected so far "
            "carries `is_synthetic: true` (mock/rehearsal data) and has been "
            "excluded from this summary and from every table/figure, per the "
            "mission's non-negotiable no-fabrication rule.",
            "",
            "Next step: run `lrmc run-queue experiments/queue.yaml` on the real GPU "
            "workstation (see HANDOFF.md), then re-run `lrmc aggregate`.",
            "",
        ]
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines))
        return

    lines.append(f"{len(real)} real (non-synthetic) run(s) aggregated.\n")

    lrmc_runs = [r for r in real if r.get("method") is None]
    baseline_runs = [r for r in real if r.get("method") is not None]
    ablation_runs = [r for r in real if "ablation" in r.get("run_name", "")]

    def _best(rs, path, want_max=True):
        scored = [(r, _get(r, path)) for r in rs]
        scored = [(r, v) for r, v in scored if v is not None]
        if not scored:
            return None
        return max(scored, key=lambda rv: rv[1]) if want_max else min(scored, key=lambda rv: rv[1])

    lines.append("## Headline comparison\n")
    lrmc_best = _best(lrmc_runs, "open_set.auroc")
    baseline_best = _best(baseline_runs, "open_set.auroc")
    if lrmc_best and baseline_best:
        lrmc_run, lrmc_auroc = lrmc_best
        base_run, base_auroc = baseline_best
        verdict = (
            "beats"
            if lrmc_auroc > base_auroc
            else ("ties" if lrmc_auroc == base_auroc else "loses to")
        )
        lines.append(
            f"- LRMC's best AUROC ({lrmc_auroc:.3f}, run `{lrmc_run.get('run_name')}`) "
            f"**{verdict}** the best baseline's AUROC "
            f"({base_auroc:.3f}, `{_method_name(base_run)}`)."
        )
        if lrmc_auroc <= base_auroc:
            lines.append(
                "  - **This is a negative result and must be reported as such**, not "
                "smoothed over -- check docs/DECISIONS.md and HANDOFF.md's risk list "
                "for likely causes (leakage, radii collapse, undertrained baseline)."
            )
    else:
        lines.append("- Not enough runs to compare LRMC against baselines yet.")

    fixed_vs_learnable = [r for r in baseline_runs if r.get("method") == "fixed_radius_prototype"]
    if lrmc_runs and fixed_vs_learnable:
        lrmc_run = lrmc_runs[0]
        fixed_run = fixed_vs_learnable[0]
        l_auroc = _get(lrmc_run, "open_set.auroc")
        f_auroc = _get(fixed_run, "open_set.auroc")
        if l_auroc is not None and f_auroc is not None:
            delta = l_auroc - f_auroc
            lines.append(
                f"- Learnable radii vs. fixed-quantile radii (same embedding space): "
                f"AUROC delta = {delta:+.3f} (LRMC {l_auroc:.3f} vs. fixed {f_auroc:.3f})."
            )

    if ablation_runs:
        lines.append("\n## Ablations\n")
        for r in sorted(ablation_runs, key=lambda x: x.get("run_name", "")):
            auroc = _get(r, "open_set.auroc")
            acc = _get(r, "closed_set.accuracy")
            lines.append(f"- `{r.get('run_name')}`: AUROC={auroc}, closed-set accuracy={acc}")

    lines.append("\n## Coverage\n")
    lines.append(f"- LRMC runs: {len(lrmc_runs)}")
    lines.append(
        f"- Baseline runs: {len(baseline_runs)} ({sorted({_method_name(r) for r in baseline_runs})})"
    )
    lines.append(f"- Ablation runs: {len(ablation_runs)}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
