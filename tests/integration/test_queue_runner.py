import json

from lrmc.orchestration.queue import QueueRunner


def test_queue_runner_executes_rehearsal_queue_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil

    repo_root = _repo_root()
    shutil.copytree(repo_root / "configs", tmp_path / "configs")
    shutil.copytree(repo_root / "experiments", tmp_path / "experiments")

    runner = QueueRunner("experiments/queue_rehearsal.yaml", session_budget_minutes=30)
    results = runner.run()

    statuses = {r.name: r.status for r in results}
    assert statuses["rehearsal_malimg_main_lrmc"] == "completed"
    assert statuses["rehearsal_malimg_baseline_softmax_msp"] == "completed"
    assert statuses["rehearsal_malimg_baseline_prototype_cosine"] == "completed"
    assert statuses["rehearsal_malimg_baseline_fixed_radius_prototype"] == "completed"
    assert statuses["rehearsal_ablation_no_l_rad"] == "completed"
    assert statuses["rehearsal_big2015_main_lrmc"] == "completed"

    for name in statuses:
        metrics_path = tmp_path / "runs" / name / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert metrics["is_synthetic"] is True

    assert (tmp_path / "runs" / "queue_state.json").exists()


def test_queue_runner_skips_completed_jobs_on_second_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil

    repo_root = _repo_root()
    shutil.copytree(repo_root / "configs", tmp_path / "configs")
    shutil.copytree(repo_root / "experiments", tmp_path / "experiments")

    runner1 = QueueRunner("experiments/queue_rehearsal.yaml", session_budget_minutes=30)
    runner1.run()

    runner2 = QueueRunner("experiments/queue_rehearsal.yaml", session_budget_minutes=30)
    results2 = runner2.run()
    assert all(r.status == "skipped_done" for r in results2)


def test_queue_runner_stops_gracefully_on_tight_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil

    repo_root = _repo_root()
    shutil.copytree(repo_root / "configs", tmp_path / "configs")
    shutil.copytree(repo_root / "experiments", tmp_path / "experiments")

    runner = QueueRunner("experiments/queue_rehearsal.yaml", session_budget_minutes=0.0001)
    results = runner.run()
    assert results[0].status == "skipped_budget"
    assert len(results) == 1  # stopped immediately, did not skip ahead to other jobs


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]
