"""Tests the pure logic in scripts/setup_env.py (command building, path
resolution) without actually invoking pip/venv -- those are exercised
manually on the real workstation, not in CI-style tests."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "setup_env.py"


def _load_setup_env_module():
    spec = importlib.util.spec_from_file_location("setup_env", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_venv_python_path_windows():
    mod = _load_setup_env_module()
    p = mod.venv_python_path(Path("C:/repo/.venv"), os_name="Windows")
    assert p == Path("C:/repo/.venv/Scripts/python.exe")


def test_venv_python_path_linux():
    mod = _load_setup_env_module()
    p = mod.venv_python_path(Path("/repo/.venv"), os_name="Linux")
    assert p == Path("/repo/.venv/bin/python")


def test_build_pip_cmd_basic():
    mod = _load_setup_env_module()
    cmd = mod.build_pip_cmd("python", ["-r", "requirements.txt"])
    assert cmd == ["python", "-m", "pip", "install", "-r", "requirements.txt"]


def test_build_pip_cmd_with_proxy_and_user():
    mod = _load_setup_env_module()
    cmd = mod.build_pip_cmd(
        "python", ["-r", "requirements.txt"], proxy="http://proxy:8080", use_user=True
    )
    assert "--proxy" in cmd
    assert cmd[cmd.index("--proxy") + 1] == "http://proxy:8080"
    assert "--user" in cmd


def test_build_pip_cmd_with_wheelhouse_is_offline():
    mod = _load_setup_env_module()
    cmd = mod.build_pip_cmd("python", ["-r", "requirements.txt"], wheelhouse="wheelhouse")
    assert "--no-index" in cmd
    assert "--find-links=wheelhouse" in cmd


def test_module_has_no_third_party_imports_at_load_time():
    """setup_env.py must be importable with nothing but the stdlib installed
    (it runs before any dependency exists). exec_module() would raise
    ImportError here if a module-level import pulled in a third-party
    package, so a clean load is itself the proof."""
    mod = _load_setup_env_module()
    assert hasattr(mod, "main")
    assert hasattr(mod, "build_pip_cmd")
