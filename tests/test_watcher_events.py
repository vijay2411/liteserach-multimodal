"""DirtyPathQueue — debounced, thread-safe coalescer."""
import time
from pathlib import Path
from semanticsd.watcher.events import DirtyPathQueue


def test_mark_dedups_same_path():
    q = DirtyPathQueue(debounce_s=0.0)
    p = Path("/x/a")
    q.mark(p)
    q.mark(p)
    q.mark(p)
    out = q.take_all()
    assert len(out) == 1
    assert out[0].path == p
    assert out[0].deleted is False


def test_mark_deleted_overrides_mark_in_window():
    q = DirtyPathQueue(debounce_s=0.0)
    p = Path("/x/a")
    q.mark(p)
    q.mark_deleted(p)
    out = q.take_all()
    assert len(out) == 1
    assert out[0].deleted is True


def test_recreate_after_delete_reverts_to_create():
    q = DirtyPathQueue(debounce_s=0.0)
    p = Path("/x/a")
    q.mark_deleted(p)
    q.mark(p)
    out = q.take_all()
    assert len(out) == 1
    assert out[0].deleted is False


def test_take_quiet_respects_debounce():
    q = DirtyPathQueue(debounce_s=0.2)
    q.mark(Path("/x/a"))
    # Immediately after marking, not yet quiet.
    assert q.take_quiet() == []
    time.sleep(0.25)
    out = q.take_quiet()
    assert len(out) == 1


def test_take_quiet_returns_empty_when_no_dirty():
    q = DirtyPathQueue(debounce_s=0.0)
    assert q.take_quiet() == []
    assert q.is_quiet() is False


def test_take_all_ignores_debounce():
    q = DirtyPathQueue(debounce_s=10.0)
    q.mark(Path("/x/a"))
    out = q.take_all()
    assert len(out) == 1


def test_concurrent_marks_are_safe():
    """Hammer the queue from many threads; final count matches unique paths."""
    import threading
    q = DirtyPathQueue(debounce_s=0.0)
    paths = [Path(f"/x/{i}") for i in range(50)]

    def worker(p):
        for _ in range(20):
            q.mark(p)

    threads = [threading.Thread(target=worker, args=(p,)) for p in paths]
    for t in threads: t.start()
    for t in threads: t.join()
    out = q.take_all()
    assert len(out) == 50
    assert {e.path for e in out} == set(paths)


def test_pending_count():
    q = DirtyPathQueue(debounce_s=0.0)
    assert q.pending_count() == 0
    q.mark(Path("/x/a"))
    q.mark(Path("/x/b"))
    assert q.pending_count() == 2
