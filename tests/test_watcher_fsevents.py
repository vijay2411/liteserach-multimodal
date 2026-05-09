"""FSEventsWatcher integration test against tmp dirs.

These tests use the real watchdog Observer (FSEvents on macOS). They're
fast — events fire within ~100ms of fs ops. Wait windows are conservative.
"""
import os
import time
from pathlib import Path
import pytest
from semanticsd.watcher.events import DirtyPathQueue
from semanticsd.watcher.fsevents_watcher import FSEventsWatcher


WAIT_S = 1.5  # generous; FSEvents typically delivers in <300ms


def _wait_for(queue: DirtyPathQueue, predicate, timeout_s: float = WAIT_S) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate(queue):
            return True
        time.sleep(0.05)
    return False


def test_create_event_is_marked(tmp_path):
    q = DirtyPathQueue(debounce_s=0.0)
    w = FSEventsWatcher(q)
    w.start([tmp_path])
    try:
        new_file = tmp_path / "new.txt"
        new_file.write_text("hi")
        assert _wait_for(q, lambda qq: qq.pending_count() >= 1)
        out = q.take_all()
        paths = [e.path for e in out]
        assert any(new_file == p or new_file.resolve() == p.resolve() for p in paths)
        assert all(e.deleted is False for e in out if e.path == new_file)
    finally:
        w.stop()


def test_modify_event_is_marked(tmp_path):
    p = tmp_path / "existing.txt"
    p.write_text("v1")
    q = DirtyPathQueue(debounce_s=0.0)
    w = FSEventsWatcher(q)
    w.start([tmp_path])
    try:
        time.sleep(0.2)  # give the watcher a beat to settle
        q.take_all()  # clear any startup events
        p.write_text("v2-modified")
        os.utime(p, None)
        assert _wait_for(q, lambda qq: qq.pending_count() >= 1)
    finally:
        w.stop()


def test_delete_event_is_marked_deleted(tmp_path):
    p = tmp_path / "doomed.txt"
    p.write_text("bye")
    q = DirtyPathQueue(debounce_s=0.0)
    w = FSEventsWatcher(q)
    w.start([tmp_path])
    try:
        time.sleep(0.2)
        q.take_all()
        p.unlink()
        assert _wait_for(q, lambda qq: qq.pending_count() >= 1)
        out = q.take_all()
        deleted_entries = [e for e in out if e.deleted]
        assert any(e.path.name == "doomed.txt" for e in deleted_entries)
    finally:
        w.stop()


def test_start_with_no_valid_dirs_is_idle(tmp_path):
    q = DirtyPathQueue()
    w = FSEventsWatcher(q)
    w.start([tmp_path / "does-not-exist"])
    try:
        assert w.is_running is False
        assert w.roots == []
    finally:
        w.stop()


def test_stop_is_idempotent():
    q = DirtyPathQueue()
    w = FSEventsWatcher(q)
    w.stop()  # never started
    w.stop()  # double stop


def test_restart_picks_up_new_dirs(tmp_path):
    """Calling start() while running should restart with the new set."""
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    q = DirtyPathQueue(debounce_s=0.0)
    w = FSEventsWatcher(q)
    w.start([a])
    try:
        assert w.is_running
        assert w.roots == [a]
        w.start([b])  # should restart
        assert w.is_running
        assert w.roots == [b]
    finally:
        w.stop()
