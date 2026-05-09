"""watchdog-based filesystem watcher.

On macOS, watchdog uses FSEvents under the hood (kqueue on BSD, inotify
on Linux). This module hides those details behind a clean start/stop
contract that drops events into a DirtyPathQueue.
"""
from __future__ import annotations
import logging
from pathlib import Path
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileDeletedEvent,
    DirCreatedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    DirDeletedEvent,
)
from watchdog.observers import Observer

from semanticsd.watcher.events import DirtyPathQueue

log = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    """Routes watchdog events into a DirtyPathQueue.

    We don't filter on extension/size here — that's the indexer's job
    via the existing IgnoreMatcher and extractor registry. Better to be
    forgiving at the watcher layer.
    """

    def __init__(self, queue: DirtyPathQueue):
        super().__init__()
        self.queue = queue

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            self.queue.mark(Path(event.src_path))

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            self.queue.mark(Path(event.src_path))

    def on_deleted(self, event):
        if isinstance(event, (FileDeletedEvent, DirDeletedEvent)):
            self.queue.mark_deleted(Path(event.src_path))

    def on_moved(self, event):
        if isinstance(event, (FileMovedEvent, DirMovedEvent)):
            self.queue.mark_deleted(Path(event.src_path))
            self.queue.mark(Path(event.dest_path))


class FSEventsWatcher:
    """Wraps watchdog's Observer with start/stop and target-dir tracking."""

    def __init__(self, queue: DirtyPathQueue):
        self.queue = queue
        self._observer: Observer | None = None
        self._roots: list[Path] = []

    @property
    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def start(self, directories: list[Path]) -> None:
        """Begin watching the given directories recursively. Idempotent: if
        already running, restart with the new dir set."""
        if self.is_running:
            self.stop()

        directories = [Path(d).expanduser() for d in directories]
        valid = [d for d in directories if d.exists() and d.is_dir()]
        if not valid:
            log.info("no valid watch directories — watcher idle")
            self._observer = None
            self._roots = []
            return

        observer = Observer()
        handler = _Handler(self.queue)
        for d in valid:
            observer.schedule(handler, str(d), recursive=True)
            log.info("watching %s", d)
        observer.start()
        self._observer = observer
        self._roots = valid

    def stop(self, timeout_s: float = 2.0) -> None:
        if self._observer is None:
            return
        try:
            self._observer.stop()
            self._observer.join(timeout=timeout_s)
        except Exception as e:
            log.warning("error stopping observer: %s", e)
        finally:
            self._observer = None
            self._roots = []
