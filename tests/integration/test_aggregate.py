"""Tests the aggregate pipeline's mechanics (collect -> tables/figures/facts/
summary) using tiny mock-data runs. To exercise the "real run" code path
without claiming any of these numbers are genuine results, these tests
explicitly set cfg.is_synthetic = False on data that is still 100% mock/tiny
-- this never touches the repo's real runs/ directory (everything lives under
tmp_path) and is purely testing the aggregation logic, not producing a
paper artifact.
"""

import json

from lrmc.aggregate.collect import collect_runs
from lrmc.aggregate.facts import write_paper_facts
from lrmc.aggregate.figures import generate_all_figures
from lrmc.aggregate.summary import write_results_summary
from lrmc.aggregate.tables import generate_all_tables
from lrmc.config import Config
from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg
from lrmc.data.splits import random_k_unknown
from lrmc.engine.train import Trainer
from lrmc.eval.evaluate import run_evaluation, write_metrics_json


def _make_run(tmp_path, run_name, seed, is_synthetic):
    root = tmp_path / f"mock_{seed}"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=seed)
    loader = MalimgLoader(root)
    fold = random_k_unknown(loader.records, n_unknown=2, seed=seed)

    cfg = Config()
    cfg.run_name = run_name
    cfg.backbone.name = "tiny_test"
    cfg.backbone.img_size = 32
    cfg.backbone.patch_size = 8
    cfg.backbone.depth = 2
    cfg.backbone.embed_dim = 16
    cfg.backbone.num_heads = 2
    cfg.projection_head.hidden_dim = 16
    cfg.projection_head.out_dim = 8
    cfg.data.image_size = 32
    cfg.data.batch_size = 4
    cfg.data.num_workers = 0
    cfg.optim.epochs = 2
    cfg.is_synthetic = is_synthetic
    out_dir = tmp_path / "runs" / run_name
    cfg.output_dir = str(out_dir)

    trainer = Trainer(cfg, loader, fold)
    train_metrics = trainer.fit()
    write_metrics_json(train_metrics, out_dir / "train_metrics.json")
    metrics = run_evaluation(
        cfg, loader, fold, trainer.checkpoint_path, save_arrays_path=out_dir / "raw_eval_arrays.npz"
    )
    write_metrics_json(metrics, out_dir / "metrics.json")
    return out_dir


def test_aggregate_with_real_flagged_runs_produces_content(tmp_path):
    _make_run(tmp_path, "test_main_lrmc", seed=20, is_synthetic=False)
    runs = collect_runs(tmp_path / "runs")
    assert len(runs) == 1
    assert runs[0]["is_synthetic"] is False

    tables_dir = tmp_path / "paper_artifacts" / "tables"
    figures_dir = tmp_path / "paper_artifacts" / "figures"
    generate_all_tables(runs, tables_dir)
    figs = generate_all_figures(runs, figures_dir)

    main_tex = (tables_dir / "main_results.tex").read_text()
    assert "No non-synthetic runs" not in main_tex
    assert "test_main_lrmc" in main_tex or "\\begin{tabular}" in main_tex

    assert len(figs) > 0
    assert any(p.name.startswith("training_curves") for p in figs)
    assert any(p.name.startswith("radii_over_epochs") for p in figs)

    facts_path = tmp_path / "paper_artifacts" / "paper_facts.json"
    write_paper_facts(runs, facts_path)
    facts = json.loads(facts_path.read_text())
    assert facts["n_runs_real"] == 1
    assert facts["runs"][0]["hyperparameters"]["backbone"]["name"] == "tiny_test"

    summary_path = tmp_path / "paper_artifacts" / "results_summary.md"
    write_results_summary(runs, summary_path)
    summary = summary_path.read_text()
    assert "real (non-synthetic) run" in summary


def test_aggregate_refuses_synthetic_runs(tmp_path):
    _make_run(tmp_path, "test_synth_lrmc", seed=21, is_synthetic=True)
    runs = collect_runs(tmp_path / "runs")
    assert runs[0]["is_synthetic"] is True

    tables_dir = tmp_path / "paper_artifacts" / "tables"
    figures_dir = tmp_path / "paper_artifacts" / "figures"
    generate_all_tables(runs, tables_dir)
    figs = generate_all_figures(runs, figures_dir)

    main_tex = (tables_dir / "main_results.tex").read_text()
    assert "No non-synthetic runs" in main_tex
    assert figs == []

    summary_path = tmp_path / "paper_artifacts" / "results_summary.md"
    write_results_summary(runs, summary_path)
    assert "No non-synthetic runs" in summary_path.read_text()
