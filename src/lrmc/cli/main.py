"""CLI entry points: train, evaluate, infer, scan, run-queue, aggregate,
verify-data, status, benchmark.

Usage: ``lrmc <command> [args]`` (installed via pyproject.toml's
``[project.scripts]``) or ``python -m lrmc.cli.main <command> [args]``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

from lrmc.config import load_config, merge_overrides
from lrmc.data.factory import build_fold, build_loader
from lrmc.data.image_generator import ImageGenerator
from lrmc.engine.alert import AlertGenerator
from lrmc.engine.infer import FrozenInferenceEngine
from lrmc.engine.scan import Scanner
from lrmc.engine.train import Trainer
from lrmc.eval.evaluate import run_evaluation, write_metrics_json
from lrmc.orchestration.queue import QueueRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lrmc.cli")


def _parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--override expects key=value, got '{pair}'")
        key, value = pair.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # keep as plain string
        overrides[key] = value
    return overrides


def _load_cfg(args) -> object:
    cfg = load_config(args.config)
    if getattr(args, "override", None):
        cfg = merge_overrides(cfg, _parse_overrides(args.override))
    return cfg


def cmd_train(args) -> None:
    cfg = _load_cfg(args)
    loader = build_loader(cfg)
    fold = build_fold(cfg, loader)
    trainer = Trainer(cfg, loader, fold)
    metrics = trainer.fit()
    out_dir = Path(cfg.output_dir)
    write_metrics_json(metrics, out_dir / "train_metrics.json")
    fold.to_json(out_dir / "fold.json")
    logger.info("Training complete. Checkpoint: %s", trainer.checkpoint_path)


def cmd_evaluate(args) -> None:
    cfg = _load_cfg(args)
    loader = build_loader(cfg)
    fold = build_fold(cfg, loader)
    checkpoint_path = args.checkpoint or str(Path(cfg.output_dir) / "checkpoint.pt")
    metrics = run_evaluation(cfg, loader, fold, checkpoint_path)
    out_path = Path(args.output) if args.output else Path(cfg.output_dir) / "metrics.json"
    write_metrics_json(metrics, out_path)
    logger.info("Evaluation complete. Wrote %s", out_path)
    print(
        json.dumps(
            {
                "closed_set_accuracy": metrics["closed_set"]["accuracy"],
                "auroc": metrics["open_set"].get("auroc"),
            },
            indent=2,
        )
    )


def cmd_infer(args) -> None:
    cfg = _load_cfg(args)
    checkpoint_path = args.checkpoint or str(Path(cfg.output_dir) / "checkpoint.pt")
    engine = FrozenInferenceEngine(cfg, checkpoint_path)
    gen = ImageGenerator(image_size=cfg.data.image_size)

    from lrmc.data.preprocessor import FilePreprocessor

    pre = FilePreprocessor()
    if args.input.endswith(".bytes"):
        result = pre.load_bytes_file(args.input)
        image = gen.eval_view(result.byte_stream, is_pre_rendered_image=False)
    else:
        result = pre.load_image_array(args.input)
        image = gen.eval_view(result.byte_stream, is_pre_rendered_image=True)

    out = engine.predict(image)
    print(
        json.dumps(
            {
                "sha256": result.sha256,
                "verdict": out.verdict,
                "nearest_family": out.nearest_family,
                "score": out.score,
            },
            indent=2,
        )
    )


def cmd_scan(args) -> None:
    cfg = _load_cfg(args)
    checkpoint_path = args.checkpoint or str(Path(cfg.output_dir) / "checkpoint.pt")
    engine = FrozenInferenceEngine(cfg, checkpoint_path)
    gen = ImageGenerator(image_size=cfg.data.image_size)
    alert_gen = AlertGenerator(args.alerts_out)
    scanner = Scanner(engine, gen, alert_gen)

    paths = [args.path] if Path(args.path).is_file() else list(Path(args.path).rglob("*"))
    paths = [p for p in paths if Path(p).is_file()]
    for p in paths:
        alert = scanner.scan_file(p)
        logger.info("%s -> %s (%s)", p, alert.verdict, alert.nearest_family)
    logger.info("Scanned %d file(s). Alerts written to %s", len(paths), args.alerts_out)


def cmd_run_queue(args) -> None:
    runner = QueueRunner(
        args.queue, session_budget_minutes=args.budget_minutes, dry_run=args.dry_run
    )
    results = runner.run()
    for r in results:
        logger.info("[%s] %s: %s", r.tier, r.name, r.status)
    n_completed = sum(1 for r in results if r.status == "completed")
    n_failed = sum(1 for r in results if r.status == "failed")
    logger.info(
        "Queue run finished: %d completed, %d failed, %d total", n_completed, n_failed, len(results)
    )


def cmd_verify_data(args) -> None:
    """Reproduces the layout/count/duplicate report on REAL data (Malimg
    auto-detected folders, or a real BIG2015 root incl. .7z archives), for
    the manual-copy path (`scripts/fetch_data.py` runs the same check
    automatically after downloading)."""
    if args.dataset == "malimg":
        from lrmc.data.malimg import MalimgLoader

        loader = MalimgLoader(args.root)
        report = loader.report
    else:
        from lrmc.data.big2015 import Big2015Loader

        cache_dir = args.cache_dir or "data_cache/cache_big2015"
        loader = Big2015Loader(args.root, cache_dir, n_workers=args.workers or 1)
        report = loader.build_cache()

    print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    if report.mismatches:
        logger.error("%s layout mismatch: %s", args.dataset, report.mismatches)
        raise SystemExit(1)
    logger.info("%s layout OK.", args.dataset)


def cmd_status(args) -> None:
    """Current job, epoch, ETA, last checkpoint, GPU utilization, free disk --
    everything `lrmc run-queue` writes as it goes, read back without needing
    to tail a log file (useful right after reconnecting over AnyDesk)."""
    from lrmc.orchestration.status import read_current_job, read_job_status
    from lrmc.utils.hardware import free_disk_bytes, gpu_status_list

    runs_dir = Path(args.runs_dir)
    current = read_current_job(runs_dir)
    if current is None:
        print("No job currently recorded as running (queue hasn't started, or has finished).")
    else:
        print(f"Current job: {current['name']}  (tier {current['tier']}, type {current['type']})")
        started_at = current.get("started_at")
        if started_at:
            ago_min = (time.time() - started_at) / 60
            print(f"  started {ago_min:.1f} min ago")

        job_status = read_job_status(runs_dir / current["name"])
        if job_status:
            epoch, total = job_status.get("epoch"), job_status.get("total_epochs")
            if epoch is not None and total is not None:
                print(f"  epoch {epoch + 1}/{total}")
                epoch_seconds = job_status.get("epoch_seconds")
                if epoch_seconds:
                    remaining = total - (epoch + 1)
                    print(
                        f"  last epoch: {epoch_seconds:.1f}s -> ETA ~{remaining * epoch_seconds / 60:.1f} min"
                    )
            print(f"  losses: {job_status.get('losses')}")
            print(
                f"  batch_size={job_status.get('batch_size')} "
                f"grad_accumulation_steps={job_status.get('grad_accumulation_steps')}"
            )
        else:
            print("  (no per-epoch status yet)")

        ckpt_path = runs_dir / current["name"] / "checkpoint.pt"
        if ckpt_path.exists():
            age_min = (time.time() - ckpt_path.stat().st_mtime) / 60
            print(f"  last checkpoint: {ckpt_path} (saved {age_min:.1f} min ago)")
        else:
            print("  no checkpoint saved yet")

    state_path = runs_dir / "queue_state.json"
    if state_path.exists():
        with open(state_path, encoding="utf-8") as fh:
            results = json.load(fh)
        n_completed = sum(1 for r in results if r["status"] == "completed")
        print(
            f"Queue progress: {n_completed} completed / {len(results)} attempted so far ({state_path})"
        )

    gpus = gpu_status_list()
    if gpus:
        for g in gpus:
            print(
                f"GPU {g['index']} ({g['name']}): {g['utilization_pct']:.0f}% util, "
                f"{g['memory_free_mb']:.0f}/{g['memory_total_mb']:.0f} MB free"
            )
    else:
        print("GPU status: unavailable (no nvidia-smi found, or no GPU)")

    print(f"Free disk: {free_disk_bytes(runs_dir) / 1e9:.1f} GB")


def cmd_benchmark(args) -> None:
    from lrmc.orchestration.benchmark import run_benchmark
    from lrmc.utils.hardware import resolve_device, system_summary

    device = resolve_device(args.device)
    logger.info("Benchmarking on %s. System: %s", device, system_summary())
    result = run_benchmark(
        args.queue,
        device,
        n_seeds=args.seeds,
        seed_tiers=tuple(args.seed_tiers.split(",")),
        configs_out_dir=args.configs_out_dir,
        expand_seeds=not args.no_expand_seeds,
    )
    logger.info(
        "Benchmark complete: device=%s, %d jobs after expansion, queue rewritten (backup at %s)",
        result["device"],
        result["n_jobs_after"],
        result["backup_path"],
    )
    print(json.dumps(result["measurements"], indent=2))


def cmd_aggregate(args) -> None:
    from lrmc.aggregate.collect import collect_runs
    from lrmc.aggregate.facts import write_paper_facts
    from lrmc.aggregate.figures import generate_all_figures
    from lrmc.aggregate.summary import write_results_summary
    from lrmc.aggregate.tables import generate_all_tables

    runs = collect_runs(args.runs_dir)
    logger.info(
        "Collected %d run(s): %d real, %d synthetic (excluded from tables/figures)",
        len(runs),
        sum(1 for r in runs if not r["is_synthetic"]),
        sum(1 for r in runs if r["is_synthetic"]),
    )
    out_dir = Path(args.output_dir)
    generate_all_tables(runs, out_dir / "tables")
    generate_all_figures(runs, out_dir / "figures")
    write_paper_facts(runs, out_dir / "paper_facts.json")
    write_results_summary(runs, out_dir / "results_summary.md")
    logger.info("Paper artifacts written to %s", out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lrmc")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train an LRMC model from a config")
    p_train.add_argument("config")
    p_train.add_argument(
        "--override", action="append", default=[], help="key=value dotted override"
    )
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained checkpoint")
    p_eval.add_argument("config")
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--output", default=None)
    p_eval.add_argument("--override", action="append", default=[])
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = sub.add_parser("infer", help="Run frozen inference on a single file")
    p_infer.add_argument("config")
    p_infer.add_argument("input")
    p_infer.add_argument("--checkpoint", default=None)
    p_infer.add_argument("--override", action="append", default=[])
    p_infer.set_defaults(func=cmd_infer)

    p_scan = sub.add_parser(
        "scan", help="Scan a raw executable (or directory) and emit JSONL alerts"
    )
    p_scan.add_argument("config")
    p_scan.add_argument("path")
    p_scan.add_argument("--checkpoint", default=None)
    p_scan.add_argument("--alerts-out", default="alerts.jsonl")
    p_scan.add_argument("--override", action="append", default=[])
    p_scan.set_defaults(func=cmd_scan)

    p_queue = sub.add_parser("run-queue", help="Execute a prioritized job queue")
    p_queue.add_argument("queue", nargs="?", default="experiments/queue.yaml")
    p_queue.add_argument("--budget-minutes", type=float, default=None)
    p_queue.add_argument("--dry-run", action="store_true")
    p_queue.set_defaults(func=cmd_run_queue)

    p_verify = sub.add_parser(
        "verify-data", help="Verify real dataset layout/counts/duplicates against expectations"
    )
    p_verify.add_argument("--dataset", choices=["malimg", "big2015"], required=True)
    p_verify.add_argument("--root", required=True)
    p_verify.add_argument("--cache-dir", default=None, help="big2015 only: uint8 cache location")
    p_verify.add_argument(
        "--workers", type=int, default=None, help="big2015 only: cache-build workers"
    )
    p_verify.set_defaults(func=cmd_verify_data)

    p_status = sub.add_parser(
        "status", help="Report current job/epoch/ETA/checkpoint/GPU/disk status"
    )
    p_status.add_argument("--runs-dir", default="runs")
    p_status.set_defaults(func=cmd_status)

    p_bench = sub.add_parser(
        "benchmark",
        help="Measure real throughput/max batch size and rewrite experiments/queue.yaml",
    )
    p_bench.add_argument("--queue", default="experiments/queue.yaml")
    p_bench.add_argument("--device", default="auto")
    p_bench.add_argument("--seeds", type=int, default=5)
    p_bench.add_argument("--seed-tiers", default="A,B")
    p_bench.add_argument("--configs-out-dir", default="configs/generated")
    p_bench.add_argument("--no-expand-seeds", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    p_agg = sub.add_parser("aggregate", help="Build paper_artifacts/ from runs/")
    p_agg.add_argument("--runs-dir", default="runs")
    p_agg.add_argument("--output-dir", default="paper_artifacts")
    p_agg.set_defaults(func=cmd_aggregate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
