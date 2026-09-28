import json
import os

from lrmc.orchestration.lock import QueueLock


def test_acquire_and_release(tmp_path):
    lock = QueueLock(tmp_path / "queue.lock")
    assert lock.acquire() is True
    assert (tmp_path / "queue.lock").exists()
    lock.release()
    assert not (tmp_path / "queue.lock").exists()


def test_second_acquire_by_live_process_fails(tmp_path):
    path = tmp_path / "queue.lock"
    lock1 = QueueLock(path)
    assert lock1.acquire() is True

    lock2 = QueueLock(path)
    assert lock2.acquire() is False  # held by this (live) process
    lock1.release()


def test_stale_lock_is_reclaimed(tmp_path):
    path = tmp_path / "queue.lock"
    # a PID that (almost certainly) does not exist
    dead_pid = 999_999
    with open(path, "w") as fh:
        json.dump({"pid": dead_pid, "acquired_at": 0}, fh)

    lock = QueueLock(path)
    assert lock.acquire() is True
    with open(path) as fh:
        data = json.load(fh)
    assert data["pid"] == os.getpid()
    lock.release()


def test_context_manager_raises_when_locked(tmp_path):
    path = tmp_path / "queue.lock"
    lock1 = QueueLock(path)
    lock1.acquire()
    try:
        raised = False
        try:
            with QueueLock(path):
                pass
        except RuntimeError:
            raised = True
        assert raised
    finally:
        lock1.release()
