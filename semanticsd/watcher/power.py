"""PowerController — orchestrates watcher / periodic-sweep / battery loops
based on the active mode (active | saver) plus auto-saver-on-battery.

Lifecycle:
    pc = PowerController(cfg, indexer, worker)
    await pc.startup()    # initial sweep + start whichever mode applies
    ...
    await pc.shutdown()   # stop watcher, drain, cancel tasks

Mode transitions (active <-> saver) are atomic and idempotent. Callers from
the HTTP/CLI layer use `await pc.set_mode(...)`.
"""
from __future__ import annotations
import asyncio
import logging
import time
from pathlib import Path
from typing import Literal

from semanticsd.config import Config, parse_interval
from semanticsd.pipeline.indexer import Indexer
from semanticsd.pipeline.worker import Worker
from semanticsd.watcher.battery import is_on_battery, power_source
from semanticsd.watcher.events import DirtyPathQueue
from semanticsd.watcher.fsevents_watcher import FSEventsWatcher
from semanticsd.watcher.sweep import sweep_directories

log = logging.getLogger(__name__)

Mode = Literal["active", "saver"]

# How often the dirty-queue drain loop wakes up to check whether the
# debounce window has elapsed. Smaller = faster reaction; bigger = less CPU.
DRAIN_TICK_S = 0.2

# Battery polling cadence.
BATTERY_TICK_S = 30.0


class PowerController:
    def __init__(
        self,
        cfg: Config,
        indexer: Indexer,
        worker: Worker,
        debounce_s: float = 1.0,
    ):
        self.cfg = cfg
        self.indexer = indexer
        self.worker = worker
        self.dirty = DirtyPathQueue(debounce_s=debounce_s)
        self.watcher = FSEventsWatcher(self.dirty)

        # Configured mode is the target; actual mode is None until startup()
        # drives the first transition. This guarantees the initial entry
        # actually starts the watcher / saver task even when configured mode
        # equals the field default.
        self._configured_mode: Mode = (
            cfg.power.mode if cfg.power.mode in ("active", "saver") else "active"
        )
        self.mode: Mode | None = None
        self.last_sweep_at: float | None = None
        self._stopping = False

        # Background tasks (created in startup, cancelled in shutdown):
        self._drain_task: asyncio.Task | None = None
        self._worker_task: asyncio.Task | None = None
        self._saver_task: asyncio.Task | None = None
        self._battery_task: asyncio.Task | None = None
        self._mode_lock = asyncio.Lock()

    @property
    def directories(self) -> list[Path]:
        return [Path(d).expanduser() for d in self.cfg.watch.directories]

    @property
    def saver_interval_s(self) -> int:
        try:
            return parse_interval(self.cfg.power.saver_reindex_interval)
        except Exception as e:
            log.warning("invalid saver_reindex_interval=%r (%s); falling back to 1h",
                        self.cfg.power.saver_reindex_interval, e)
            return 3600

    # ------------------------------------------------------------------ status

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "auto_saver_on_battery": bool(self.cfg.power.auto_saver_on_battery),
            "power_source": power_source(),
            "directories": [str(d) for d in self.directories],
            "watcher_running": self.watcher.is_running,
            "dirty_pending": self.dirty.pending_count(),
            "last_sweep_at": self.last_sweep_at,
            "saver_interval_s": self.saver_interval_s,
        }

    # ------------------------------------------------------------------ lifecycle

    async def startup(self) -> None:
        """Run initial sweep + start whichever mode applies + worker drain loop."""
        await self._initial_sweep()
        # Always run the worker drain loop in background so embed jobs flow.
        self._worker_task = asyncio.create_task(self._worker_loop(), name="worker-drain")
        # Apply battery preference before activating the chosen mode, so a
        # daemon launched on battery starts in saver if configured.
        target = self._mode_for_battery_state(self._configured_mode)
        await self._enter_mode(target, reason="startup")
        # Start the battery poll only if auto_saver_on_battery is enabled.
        if self.cfg.power.auto_saver_on_battery:
            self._battery_task = asyncio.create_task(self._battery_loop(), name="battery-poll")

    async def shutdown(self) -> None:
        self._stopping = True
        self.watcher.stop()
        for t in (self._drain_task, self._saver_task, self._battery_task, self._worker_task):
            if t is not None:
                t.cancel()
        # Final flush of pending dirty entries — best-effort, may be partial.
        await self._drain_dirty(force=True)

    # ------------------------------------------------------------------ control

    async def set_mode(self, target: Mode, reason: str = "manual") -> None:
        async with self._mode_lock:
            if target == self.mode:
                return
            await self._enter_mode(target, reason=reason)

    async def force_sweep(self) -> dict:
        """Trigger an immediate full re-walk regardless of mode. Returns sweep stats."""
        return await self._run_sweep(reason="manual")

    # ------------------------------------------------------------------ internals

    async def _initial_sweep(self) -> None:
        if not self.directories:
            log.info("no watch directories configured; skipping initial sweep")
            return
        await self._run_sweep(reason="initial")

    async def _run_sweep(self, reason: str) -> dict:
        log.info("starting %s sweep over %d dirs", reason, len(self.directories))
        # Indexer is sync-CPU-light but does sqlite writes; run in default executor
        # to avoid blocking the asyncio loop on big dirs.
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(
            None, sweep_directories, self.indexer, self.directories,
        )
        self.last_sweep_at = time.time()
        return stats

    async def _enter_mode(self, target: Mode, reason: str) -> None:
        if target not in ("active", "saver"):
            raise ValueError(f"unknown mode: {target!r}")
        prev = self.mode
        if prev == target:
            return
        log.info("power: %s -> %s (reason=%s)", prev, target, reason)

        if target == "active":
            # Stop saver task, start watcher + drain loop.
            if self._saver_task is not None:
                self._saver_task.cancel()
                self._saver_task = None
            self.watcher.start(self.directories)
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(self._drain_loop(), name="dirty-drain")
        else:  # saver
            if self._drain_task is not None:
                self._drain_task.cancel()
                self._drain_task = None
            self.watcher.stop()
            # Drain anything pending before stopping reactive indexing.
            await self._drain_dirty(force=True)
            if self._saver_task is None or self._saver_task.done():
                self._saver_task = asyncio.create_task(self._saver_loop(), name="saver-sweep")

        self.mode = target

    # ----------------------------- background loops --------------------------

    async def _drain_loop(self) -> None:
        """In active mode: every DRAIN_TICK_S, drain quiet dirty paths."""
        try:
            while not self._stopping:
                await asyncio.sleep(DRAIN_TICK_S)
                await self._drain_dirty(force=False)
        except asyncio.CancelledError:
            return

    async def _drain_dirty(self, force: bool) -> int:
        entries = self.dirty.take_all() if force else self.dirty.take_quiet()
        if not entries:
            return 0
        loop = asyncio.get_running_loop()
        n = 0
        for entry in entries:
            try:
                if entry.deleted:
                    n += await loop.run_in_executor(None, self.indexer.unindex_path, entry.path)
                else:
                    res = await loop.run_in_executor(None, self.indexer.index_path, entry.path)
                    n += int(res.get("files_indexed", 0))
            except Exception as e:
                log.warning("dirty drain failed for %s: %s", entry.path, e)
        if entries:
            log.debug("drained %d dirty entries (force=%s)", len(entries), force)
        return n

    async def _saver_loop(self) -> None:
        """In saver mode: full re-walk every saver_interval_s."""
        try:
            while not self._stopping:
                await asyncio.sleep(self.saver_interval_s)
                if self._stopping:
                    return
                try:
                    await self._run_sweep(reason="saver")
                except Exception as e:
                    log.warning("saver sweep failed: %s", e)
        except asyncio.CancelledError:
            return

    async def _worker_loop(self) -> None:
        """Drain the embed-job queue forever."""
        try:
            self.worker.reset_stale()
            while not self._stopping:
                try:
                    n = self.worker.drain_once()
                except Exception as e:
                    log.error("worker drain crashed: %s", e)
                    n = 0
                if n == 0:
                    await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            return

    async def _battery_loop(self) -> None:
        try:
            while not self._stopping:
                await asyncio.sleep(BATTERY_TICK_S)
                if not self.cfg.power.auto_saver_on_battery:
                    continue
                target = self._mode_for_battery_state(self.mode)
                if target != self.mode:
                    await self.set_mode(target, reason="battery")
        except asyncio.CancelledError:
            return

    def _mode_for_battery_state(self, current: Mode) -> Mode:
        if not self.cfg.power.auto_saver_on_battery:
            return current
        return "saver" if is_on_battery() else "active"
