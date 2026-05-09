# FSEvents Watcher + Power Modes for SemanticsD

**Status:** approved, pending implementation
**Date:** 2026-05-09
**Builds on:** all prior plans (Foundation, Embedders, Pipeline, Multimodal, Search, Robustness)

## Goal

Make the daemon keep its index fresh **automatically**: when the user creates,
edits, or deletes a file under a configured `[watch].directories` path, the
daemon notices and reindexes (or removes) it without manual `ssearch --index`.

Two power modes:
- **active**: FSEvents watcher running; changes propagate within seconds.
- **saver**: watcher off; periodic full-walk reindex every
  `saver_reindex_interval` (default 1h). Lowest power impact.

A third "auto-saver-on-battery" toggle flips between modes based on AC power
state.

## Non-goals

- Cross-platform watching (Linux inotify / Windows ReadDirectoryChangesW). Out
  of scope; FSEvents-only via `watchdog`.
- Per-directory mode (e.g. one dir active, another saver). Single global mode.
- Real-time push notifications to clients. The daemon stays freshly-indexed;
  HTTP clients see the new state on their next query.

## Architecture

### Components

```
semanticsd/watcher/
    __init__.py
    events.py              — DirtyPathQueue (debounced + thread-safe)
    fsevents_watcher.py    — Watchdog Observer wrapper
    sweep.py               — initial / periodic full-walk indexer
    power.py               — PowerController (active ↔ saver state machine)
    battery.py             — psutil-based AC/battery detection
```

`PowerController` is the orchestrator. It owns:
- a `DirtyPathQueue` (small in-process queue of changed paths)
- the `FSEventsWatcher` (started/stopped on mode transitions)
- a periodic-reindex `asyncio.Task` (started in saver, stopped in active)
- a battery-poll `asyncio.Task` (when `auto_saver_on_battery=true`)

It exposes coroutines that the daemon's main loop awaits, plus methods the
HTTP/CLI control surface calls into.

### Event flow (active mode)

```
fs change ─► watchdog Observer ─► DirtyPathQueue.mark(path)
                                       │
                                       │  debounce 1s after last event
                                       ▼
                            asyncio task drains the queue
                                       │
                                       ▼
                                Indexer.index_path(path)
                                       │
                                       ▼
                                 worker drains jobs
```

### Mode transitions

```
active   ─POST /v1/power {mode: "saver"}─►  saver
            (stop watcher, drain dirty queue,
             start periodic reindex timer)

saver    ─POST /v1/power {mode: "active"}─►  active
            (stop periodic timer,
             start watcher,
             optional one-shot full sweep to catch up)

auto-saver: every 30s, poll battery
   on AC: ensure mode == active   (config.auto_saver_on_battery)
   on bat: ensure mode == saver
```

### Initial sweep (chosen behavior)

On daemon startup, **before** anything else:
1. For each path in `[watch].directories`, run `Indexer.index_path(path)`.
2. The existing mtime+size unchanged check makes re-runs cheap on warm caches.
3. After the sweep, transition to the configured power mode.

This means: user adds a directory to config → restart daemon → it's indexed
and watched. No separate `ssearch --index` step needed.

### Debouncing

Watchdog emits one event per filesystem operation. A single editor save can
fire {created, modified, modified} within milliseconds. We debounce by:
1. On every event, set `last_change_time = now` and `dirty.add(path)`.
2. A drain loop wakes every 200ms and checks: if `now - last_change_time >= 1s`
   AND `dirty` non-empty, atomically swap `dirty` → local set, process those.
3. Quiet periods → drain immediately. Burst periods → batched.

This handles `git checkout` (1000 events in 5s) cleanly: one batched reindex.

### Delete / move / rename

Watchdog reports `FileDeletedEvent`, `FileMovedEvent`, `DirDeletedEvent`,
`DirMovedEvent`. Handling:

- **Delete**: remove file row + cascading chunk + vec rows. New
  `Indexer.unindex_path(path)` method.
- **Move**: handle as delete-old + create-new. Indexer.unindex(old) then
  index(new). Even if content didn't change, paths and the FTS path-index
  need updating; content-hash dedup ensures the embedder isn't called again.
- **Dir delete/move**: walk all files under the (now-gone) path in our DB,
  unindex each.

`Indexer.unindex_path(path)`:
```python
def unindex_path(self, path: Path) -> int:
    """Remove the file (and all its chunks/embeddings/jobs) from the index.
    Returns # of files removed (handles dir-deletes via prefix match)."""
    s = str(path)
    rows = self.conn.execute(
        "SELECT id FROM files WHERE path = ? OR path LIKE ?",
        (s, s + "/%"),
    ).fetchall()
    for (file_id,) in rows:
        # Cascading deletes via FK on chunks; we manually clear FTS + vec rows
        # since those are virtual tables.
        chunk_ids = [r[0] for r in self.conn.execute(
            "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
        )]
        if chunk_ids:
            ph = ",".join("?" for _ in chunk_ids)
            self.conn.execute(f"DELETE FROM fts_chunks WHERE rowid IN ({ph})", chunk_ids)
            for vec_table in self._existing_vec_tables():
                self.conn.execute(f"DELETE FROM {vec_table} WHERE rowid IN ({ph})", chunk_ids)
        self.conn.execute("DELETE FROM fts_paths WHERE rowid = ?", (file_id,))
        self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))  # cascade chunks/jobs
    return len(rows)
```

### Power state persistence

Power mode is config-driven (`[power].mode`) and **not** persisted in the DB.
A runtime POST changes the in-memory mode but doesn't rewrite config — so
restarting the daemon reverts to the configured mode. This is intentional:
config is the source of truth.

### Battery detection

Via `psutil.sensors_battery()`. On macOS this returns a `sbattery` namedtuple
with `power_plugged` field. None means no battery (e.g., desktop). Polled
every 30s by a dedicated task; transitions are atomic.

```python
def battery_loop(self):
    while True:
        if self.config.auto_saver_on_battery:
            try:
                bat = psutil.sensors_battery()
                if bat is not None:
                    target = "active" if bat.power_plugged else "saver"
                    if target != self.mode:
                        await self.transition_to(target, reason="battery")
            except Exception as e:
                log.warning("battery poll failed: %s", e)
        await asyncio.sleep(30)
```

## HTTP API

```
GET  /v1/watch              — status: mode, dirty count, last sweep time
POST /v1/watch/sweep        — trigger an immediate full re-walk of all dirs
GET  /v1/power              — { mode, auto_saver_on_battery, on_battery }
POST /v1/power              — body: {"mode": "active|saver"}
```

## CLI

```
ssearch watch                    # status
ssearch watch --sweep            # force full re-walk now
ssearch power                    # current mode
ssearch power active             # switch mode
ssearch power saver
```

## Config

Already in the spec from Plan 1, no schema change needed:

```toml
[watch]
directories = ["/Users/me/Documents", "/Users/me/Code"]   # user-chosen
ignore_patterns = [...]                                    # already exists
max_file_size_mb = 50

[power]
mode = "active"                       # active | saver
saver_reindex_interval = "1h"         # parsed: 30m, 1h, 6h, etc.
saver_pause_watcher = true            # vestigial — always true now
auto_saver_on_battery = true
```

`saver_reindex_interval` parser: support `Nm`, `Nh`, `Nd` strings + raw int (seconds).

## Daemon lifecycle integration

Modify `cli.py serve()`:

```python
def serve():
    cfg = config.load()
    logging_setup.configure(level=cfg.daemon.log_level, to_file=True)
    paths.ensure_dirs()

    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    router = embedders.get_router()

    indexer = Indexer(conn=conn, max_file_size_mb=cfg.watch.max_file_size_mb,
                     ignore_patterns=cfg.watch.ignore_patterns)
    worker  = Worker(conn=conn, router=router, batch_size=cfg.embedding.text.batch_size)

    power = PowerController(cfg, indexer, worker, conn)
    app = create_app(power)  # injected so HTTP endpoints can call into it

    async def lifespan(_app):
        await power.startup()       # initial sweep + start mode
        # background tasks: worker drain loop, dirty queue drain, periodic
        # reindex (if saver), battery poll (if enabled)
        yield
        await power.shutdown()

    app.router.lifespan_context = lifespan
    uvicorn.run(app, host=..., port=...)
```

## Tests

**Unit:**
- `DirtyPathQueue`: debounce window, dedup, drain semantics
- `parse_interval("1h")` etc.
- `Indexer.unindex_path()` against a seeded DB
- `PowerController.transition_to()` — start/stop the right tasks
- Battery loop — mocked `psutil.sensors_battery`

**Integration (no slow):**
- Watcher with a tmp dir: create file → expect mark; modify → expect re-mark;
  delete → expect unmark.
- Full power cycle: start in active, file change observed; transition to
  saver, file change NOT observed; transition back, observed again.

**E2E (slow):**
- Real launchd-style daemon startup against the stress corpus dir; create a
  new .md file; within 5s, search for its content surfaces it.
- Power-cycle while indexing: no jobs lost, no zombie watchers.

## Out of scope (deferred)

- Recursive symlink loops — watchdog handles this internally; we trust it.
- Network mounts (NFS/SMB) — FSEvents may not fire reliably; document as
  known limitation, fall back to periodic sweep.
- Indexing under `~/Documents` automatically — user must explicitly add it
  to config (no surprise indexing principle from Plan 1).

## Files added

```
semanticsd/watcher/__init__.py
semanticsd/watcher/events.py
semanticsd/watcher/fsevents_watcher.py
semanticsd/watcher/sweep.py
semanticsd/watcher/power.py
semanticsd/watcher/battery.py
semanticsd/server/routes/watch.py
semanticsd/server/routes/power.py
tests/test_watcher_events.py
tests/test_watcher_fsevents.py
tests/test_watcher_sweep.py
tests/test_watcher_power.py
tests/test_watcher_battery.py
tests/test_server_watch.py
tests/test_server_power.py
tests/test_e2e_watch.py
```

## Files modified

```
semanticsd/cli.py                    — serve() builds & runs PowerController
semanticsd/server/app.py             — register watch/power routers, accept power
semanticsd/pipeline/indexer.py       — add unindex_path()
semanticsd/config.py                 — saver_reindex_interval parser
requirements.txt                      — watchdog, psutil
```
