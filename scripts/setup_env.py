#!/usr/bin/env python
"""Cross-platform environment setup for the college GPU workstation.

Deliberately stdlib-only (no torch/yaml/etc. imports at module level) since
it has to run BEFORE any dependency is installed, on a machine whose OS,
admin rights, and network access are all unknown until you connect to it.

Usage:
    python scripts/setup_env.py                       # venv + online install
    python scripts/setup_env.py --dev                 # + pytest/ruff/black
    python scripts/setup_env.py --proxy http://host:port
    python scripts/setup_env.py --offline              # use ./wheelhouse
    python scripts/setup_env.py --wheelhouse D:\\wheels # use an explicit path
    python scripts/setup_env.py --no-venv               # no admin / venv blocked -> pip --user

Building a wheelhouse at home (for a workstation with no/restricted
internet): on a machine with the SAME OS and Python major.minor version as
the workstation,
    pip download -r requirements.txt -d wheelhouse
    pip download -r requirements-dev.txt -d wheelhouse   # if you also want dev tools
then copy the whole `wheelhouse/` folder to the workstation (USB drive,
AnyDesk file transfer, etc.) and run this script with --wheelhouse pointing
at it.

No admin rights: venv creation and pip installs into a venv do NOT require
admin on Windows or Linux in the normal case. If `python -m venv` is blocked
by a locked-down corporate image, use --no-venv, which installs with
`pip install --user` into whatever Python you invoked this script with
instead of creating an isolated environment.

Proxy: pass --proxy http://user:pass@host:port, or simply have
HTTP_PROXY/HTTPS_PROXY set in your shell environment before running this
script -- pip picks those up automatically either way.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def venv_python_path(venv_dir: Path, os_name: str | None = None) -> Path:
    os_name = (os_name or platform.system()).lower()
    if os_name == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def build_pip_cmd(
    python_exe: str | Path,
    pip_args: list[str],
    proxy: str | None = None,
    use_user: bool = False,
    wheelhouse: str | None = None,
) -> list[str]:
    """Pure command-builder (no subprocess call) so this logic is unit-testable
    without actually invoking pip."""
    cmd = [str(python_exe), "-m", "pip", "install"]
    if wheelhouse:
        cmd += ["--no-index", f"--find-links={wheelhouse}"]
    if proxy:
        cmd += ["--proxy", proxy]
    if use_user:
        cmd += ["--user"]
    cmd += pip_args
    return cmd


VERIFY_SCRIPT = """
import platform, shutil
try:
    import torch
except ImportError as e:
    print("IMPORT_FAILED: torch did not import:", e)
    raise SystemExit(1)

print("Python:", platform.python_version())
print("Platform:", platform.platform())
print("Torch:", torch.__version__)
cuda_ok = torch.cuda.is_available()
print("CUDA available:", cuda_ok)
print("CUDA version (torch build):", torch.version.cuda)
if cuda_ok:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name}, {p.total_memory / 1e9:.1f} GB VRAM")
else:
    print("No CUDA GPU detected -- will run on CPU (fine for testing, not for real training).")
free = shutil.disk_usage(".").free
print(f"Free disk (cwd): {free / 1e9:.1f} GB")
"""


def create_venv(venv_dir: Path) -> bool:
    """Returns True on success, False if venv creation failed (caller should
    fall back to --no-venv-style --user installs)."""
    if venv_dir.exists():
        print(f"venv already exists at {venv_dir}, reusing it.")
        return True
    try:
        print(f"Creating venv at {venv_dir} ...")
        venv.create(venv_dir, with_pip=True)
        return True
    except Exception as exc:  # noqa: BLE001 - any venv-creation failure should degrade, not crash
        print(f"WARNING: venv creation failed ({exc}).")
        print("This can happen without admin rights on some locked-down Windows images.")
        print("Falling back to --user installs into the current Python instead.")
        return False


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def install_requirements(
    python_exe: str | Path,
    requirements_path: Path,
    proxy: str | None,
    wheelhouse: str | None,
    use_user: bool,
) -> None:
    if wheelhouse:
        run(
            build_pip_cmd(
                python_exe, ["-r", str(requirements_path)], use_user=use_user, wheelhouse=wheelhouse
            )
        )
        return
    try:
        run(
            build_pip_cmd(
                python_exe, ["-r", str(requirements_path)], proxy=proxy, use_user=use_user
            )
        )
    except subprocess.CalledProcessError:
        print("\nOnline pip install failed.")
        auto_wheelhouse = REPO_ROOT / "wheelhouse"
        if auto_wheelhouse.exists():
            print(f"Found {auto_wheelhouse} -- retrying as an offline install from it.")
            run(
                build_pip_cmd(
                    python_exe,
                    ["-r", str(requirements_path)],
                    use_user=use_user,
                    wheelhouse=str(auto_wheelhouse),
                )
            )
        else:
            print(
                "No ./wheelhouse found either. Build one at home (same OS/Python "
                "version) with:\n"
                "    pip download -r requirements.txt -d wheelhouse\n"
                "copy the wheelhouse/ folder here, then re-run with --wheelhouse wheelhouse"
            )
            raise


def verify_install(python_exe: str | Path) -> None:
    print("\n--- Verifying install ---")
    result = subprocess.run([str(python_exe), "-c", VERIFY_SCRIPT], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("Verification failed -- see output above.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--venv-dir", default=".venv")
    parser.add_argument("--dev", action="store_true", help="also install requirements-dev.txt")
    parser.add_argument("--proxy", default=None, help="e.g. http://user:pass@host:port")
    parser.add_argument("--wheelhouse", default=None, help="path to an offline wheel directory")
    parser.add_argument(
        "--offline", action="store_true", help="shorthand for --wheelhouse ./wheelhouse"
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="skip venv creation; pip install --user into the current Python instead",
    )
    args = parser.parse_args(argv)

    wheelhouse = args.wheelhouse or (str(REPO_ROOT / "wheelhouse") if args.offline else None)

    if args.no_venv:
        python_exe: str | Path = sys.executable
        use_user = True
        print(f"--no-venv: installing with --user into {python_exe}")
    else:
        venv_dir = REPO_ROOT / args.venv_dir
        if create_venv(venv_dir):
            python_exe = venv_python_path(venv_dir)
            use_user = False
        else:
            python_exe = sys.executable
            use_user = True

    try:
        run(build_pip_cmd(python_exe, ["--upgrade", "pip"], proxy=args.proxy, use_user=use_user))
    except subprocess.CalledProcessError:
        print("pip self-upgrade failed (non-fatal) -- continuing with the existing pip.")

    req_file = "requirements-dev.txt" if args.dev else "requirements.txt"
    install_requirements(python_exe, REPO_ROOT / req_file, args.proxy, wheelhouse, use_user)

    print("\nInstalling lrmc itself (editable) ...")
    run(
        build_pip_cmd(
            python_exe,
            ["-e", str(REPO_ROOT)],
            proxy=args.proxy,
            use_user=use_user,
            wheelhouse=wheelhouse,
        )
    )

    verify_install(python_exe)

    print("\nSetup complete.")
    if not args.no_venv:
        activate_hint = (
            f"{args.venv_dir}\\Scripts\\activate"
            if platform.system().lower() == "windows"
            else f"source {args.venv_dir}/bin/activate"
        )
        print(f"Activate with: {activate_hint}")
    print("Next: see HANDOFF.md (fetch data, verify-data, benchmark, smoke run, launch Tier A).")


if __name__ == "__main__":
    main()
