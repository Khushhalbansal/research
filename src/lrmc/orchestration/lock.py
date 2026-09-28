"""PID-based lock file so two `lrmc run-queue` processes never run at once
(e.g. one launched via run_detached and a second started by accident after
reconnecting over AnyDesk and forgetting the first is still going).

A stale lock (the PID it names is no longer alive -- the previous run
crashed or the machine rebooted without a clean shutdown) is reclaimed
automatically rather than requiring manual deletion. This has a small
TOCTOU race between checking liveness and writing the new lock; acceptable
for a single-operator workstation, not intended for a multi-tenant cluster.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import psutil


class QueueLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._acquired = False

    def _held_by_live_process(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            pid = data.get("pid")
        except (json.JSONDecodeError, OSError, KeyError):
            return False
        return pid is not None and psutil.pid_exists(pid)

    def acquire(self) -> bool:
        if self._held_by_live_process():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "acquired_at": time.time()}, fh)
        self._acquired = True
        return True

    def release(self) -> None:
        if self._acquired and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
        self._acquired = False

    def __enter__(self) -> QueueLock:
        if not self.acquire():
            raise RuntimeError(
                f"Another queue runner already holds the lock at {self.path} "
                "(a live process owns it). If you're sure no run-queue is "
                "actually in progress, delete that file and retry."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
