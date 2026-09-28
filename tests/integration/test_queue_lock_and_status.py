import json

from lrmc.orchestration.lock import QueueLock
from lrmc.orchestration.queue import QueueRunner


def _copy_configs_and_experiments(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    shutil.copytree(repo_root / "configs", tmp_path / "configs")
    shutil.copytree(repo_root / "experiments", tmp_path / "experiments")


def test_second_runner_refuses_to_start_while_first_holds_lock(tmp_path, monkeypatch):
    _copy_configs_and_experiments(tmp_path, monkeypatch)
    runs_dir = tmp_path / "runs"
    lock = QueueLock(runs_dir / ".queue.lock")
    assert lock.acquire() is True
    try:
        runner = QueueRunner("experiments/queue_rehearsal.yaml", session_budget_minutes=30)
        raised = False
        try:
            runner.run()
        except RuntimeError as exc:
            raised = "already holds the lock" in str(exc)
        assert raised
    finally:
        lock.release()


def test_lock_released_after_successful_run_allows_a_fresh_runner(tmp_path, monkeypatch):
    _copy_configs_and_experiments(tmp_path, monkeypatch)
    runner1 = QueueRunner("experiments/queue_rehearsal.yaml", session_budget_minutes=30)
    runner1.run()
    assert not (tmp_path / "runs" / ".queue.lock").exists()

    # a second runner over the SAME already-completed queue must not raise
    # (lock is free, and every job is skipped_done)
    runner2 = QueueRunner("experiments/queue_rehearsal.yaml", session_budget_minutes=30)
    results2 = runner2.run()
    assert all(r.status == "skipped_done" for r in results2)


def test_current_job_status_written_during_run(tmp_path, monkeypatch):
    _copy_configs_and_experiments(tmp_path, monkeypatch)
    runner = QueueRunner("experiments/queue_rehearsal.yaml", session_budget_minutes=30)
    runner.run()

    current_job_path = tmp_path / "runs" / "current_job.json"
    assert current_job_path.exists()
    data = json.loads(current_job_path.read_text())
    assert "name" in data and "tier" in data and "started_at" in data
