import json

import pytest
import yaml

from lrmc.cli.main import main
from lrmc.data.mock import generate_mock_malimg


def test_cli_verify_data_malimg_ok(tmp_path, capsys):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=10, image_size=32, seed=0)
    # exits 1 by design (mock data mismatches the real 25/9339 counts) --
    # exercise that the report is printed and SystemExit(1) is raised, same
    # as it would be for a real, broken layout.
    with pytest.raises(SystemExit) as exc:
        main(["verify-data", "--dataset", "malimg", "--root", str(root)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "mismatches" in report
    assert report["mismatches"]


def test_cli_status_with_no_runs_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["status", "--runs-dir", "runs"])
    out = capsys.readouterr().out
    assert "No job currently recorded" in out
    assert "Free disk" in out


def test_cli_status_after_queue_run_shows_progress(tmp_path, monkeypatch, capsys):
    import shutil
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    shutil.copytree(repo_root / "configs", tmp_path / "configs")
    shutil.copytree(repo_root / "experiments", tmp_path / "experiments")

    main(["run-queue", "experiments/queue_rehearsal.yaml"])
    capsys.readouterr()  # discard run-queue's own output

    main(["status"])
    out = capsys.readouterr().out
    assert "Current job:" in out
    assert "Queue progress:" in out
    assert "Free disk:" in out


def test_cli_benchmark_rewrites_queue(tmp_path, monkeypatch, capsys):
    import shutil
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    shutil.copytree(repo_root / "configs", tmp_path / "configs")
    shutil.copytree(repo_root / "experiments", tmp_path / "experiments")

    main(
        [
            "benchmark",
            "--queue",
            "experiments/queue_rehearsal.yaml",
            "--device",
            "cpu",
            "--seeds",
            "2",
            "--no-expand-seeds",
        ]
    )
    with open("experiments/queue_rehearsal.yaml") as fh:
        queue = yaml.safe_load(fh)
    any_job = queue["tiers"]["A"]["jobs"][0]
    assert any_job["estimated"] is False
    assert (tmp_path / "experiments" / "queue_rehearsal.yaml.pre_benchmark_backup").exists()
