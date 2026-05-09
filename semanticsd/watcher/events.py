"""Thread-safe debounced queue of dirty filesystem paths."""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Default time after the last event before we consider the queue "quiet"
# enough to drain. 1.0s is forgiving enough for git checkout / editor
# atomic-write bursts without making interactive search feel stale.
DEFAULT_DEBOUNCE_S = 1.0


@dataclass(frozen=True)
class DirtyEntry:
    path: Path
    deleted: bool  # True iff the FS event was a delete or move-out


class DirtyPathQueue:
    """Coalesces filesystem events into a per-path "what is the latest state"
    so we don't index a file 5 times during a single editor save.

    Threading: the watcher thread calls `mark()` and `mark_deleted()`. The
    drain task (asyncio loop) calls `take_quiet()` periodically. Internal
    `_lock` protects shared state.
    """

    def __init__(self, debounce_s: float = DEFAULT_DEBOUNCE_S):
        self.debounce_s = debounce_s
        self._lock = threading.Lock()
        self._dirty: dict[Path, bool] = {}  # path -> deleted?
        self._last_event_at: float = 0.0

    def mark(self, path: Path) -> None:
        """Mark `path` as needing reindex. A subsequent mark_deleted overrides."""
        with self._lock:
            # If we previously marked it deleted in the same window and now it
            # came back, the create/modify wins.
            self._dirty[path] = False
            self._last_event_at = time.monotonic()

    def mark_deleted(self, path: Path) -> None:
        with self._lock:
            self._dirty[path] = True
            self._last_event_at = time.monotonic()

    def is_quiet(self, now: float | None = None) -> bool:
        """Has the debounce window elapsed since the last event?"""
        with self._lock:
            if not self._dirty:
                return False  # nothing to drain anyway
            now = now if now is not None else time.monotonic()
            return (now - self._last_event_at) >= self.debounce_s

    def take_quiet(self) -> list[DirtyEntry]:
        """If the queue has been quiet for `debounce_s`, atomically swap and
        return all pending entries. Otherwise return []."""
        with self._lock:
            if not self._dirty:
                return []
            if (time.monotonic() - self._last_event_at) < self.debounce_s:
                return []
            entries = [DirtyEntry(p, deleted) for p, deleted in self._dirty.items()]
            self._dirty.clear()
            return entries

    def take_all(self) -> list[DirtyEntry]:
        """Drain immediately, ignoring debounce. Used on shutdown / forced flush."""
        with self._lock:
            entries = [DirtyEntry(p, deleted) for p, deleted in self._dirty.items()]
            self._dirty.clear()
            return entries

    def pending_count(self) -> int:
        with self._lock:
            return len(self._dirty)
