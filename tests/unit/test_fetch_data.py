import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "fetch_data.py"


def _load_fetch_data_module():
    spec = importlib.util.spec_from_file_location("fetch_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_disk_budget_passes_when_enough_space(tmp_path):
    mod = _load_fetch_data_module()
    mod.check_disk_budget(tmp_path, required_bytes=1024)  # trivially small; should not raise


def test_check_disk_budget_raises_when_insufficient(tmp_path):
    mod = _load_fetch_data_module()
    huge = 10**18  # 1 exabyte -- no real disk has this much free
    with pytest.raises(RuntimeError, match="Refusing to start download"):
        mod.check_disk_budget(tmp_path, required_bytes=huge)


def test_should_skip_download_false_when_missing(tmp_path):
    mod = _load_fetch_data_module()
    assert mod.should_skip_download(tmp_path / "missing.zip", expected_size_bytes=123) is False


def test_should_skip_download_true_when_size_matches(tmp_path):
    mod = _load_fetch_data_module()
    p = tmp_path / "data.zip"
    p.write_bytes(b"x" * 100)
    assert mod.should_skip_download(p, expected_size_bytes=100) is True


def test_should_skip_download_false_when_size_mismatches(tmp_path):
    mod = _load_fetch_data_module()
    p = tmp_path / "data.zip"
    p.write_bytes(b"x" * 50)
    assert mod.should_skip_download(p, expected_size_bytes=100) is False


def test_should_skip_download_true_when_no_expected_size_and_file_exists(tmp_path):
    mod = _load_fetch_data_module()
    p = tmp_path / "data.zip"
    p.write_bytes(b"x")
    assert mod.should_skip_download(p, expected_size_bytes=None) is True


def test_get_kaggle_api_raises_helpful_error_without_kaggle_package():
    mod = _load_fetch_data_module()
    try:
        import kaggle  # noqa: F401

        pytest.skip("kaggle package is installed in this environment; nothing to test here")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="pip install kaggle"):
        mod._get_kaggle_api()
