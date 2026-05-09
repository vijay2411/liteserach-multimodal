"""PowerController state machine + lifecycle tests.

Uses a real Indexer + Worker against a tmp DB but stubs the embedder
(via FakeText that returns deterministic 768-d vectors) so we don't
hit Ollama or Gemini in unit tests.
"""
import asyncio
import time
from pathlib import Path
from unittest.mock import patch
import pytest

from semanticsd.config import Config, TextEmbeddingConfig, EmbeddingConfig, WatchConfig, PowerConfig
from semanticsd.db import connection, migrations
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.router import EmbedderRouter
from semanticsd.pipeline.indexer import Indexer
from semanticsd.pipeline.worker import Worker
from semanticsd.watcher.power import PowerController


class _FakeText(Embedder):
    provider_id = "fake"; model_id = "f"; dim = 768
    supports_kind = False; cost_per_million_input_tokens_usd = 0.0
    def embed(self, texts, kind):
        return EmbedResult(vectors=[[0.1] * 768 for _ in texts], input_tokens=1)
    def health_check(self): return (True, "ok")
    def estimate_tokens(self, texts): return 1


def _build_cfg(directories: list[Path], mode="active",
               saver_interval="1h", auto_saver=False) -> Config:
    return Config(
        watch=WatchConfig(directories=[str(d) for d in directories]),
        embedding=EmbeddingConfig(text=TextEmbeddingConfig(preset="local")),
        power=PowerConfig(
            mode=mode,
            saver_reindex_interval=saver_interval,
            auto_saver_on_battery=auto_saver,
        ),
    )


def _build_pc(tmp_path: Path, directories: list[Path], **cfg_kwargs) -> tuple:
    db = tmp_path / "pc.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    indexer = Indexer(conn=conn, max_file_size_mb=10)
    router = EmbedderRouter(text=_FakeText())
    worker = Worker(conn=conn, router=router, batch_size=64)
    cfg = _build_cfg(directories, **cfg_kwargs)
    pc = PowerController(cfg, indexer, worker, debounce_s=0.2)
    return pc, conn


@pytest.mark.asyncio
async def test_initial_sweep_indexes_existing_files(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    (src / "alpha.md").write_text("UNIQUEMARK_alpha some content")
    (src / "beta.md").write_text("UNIQUEMARK_beta different content")

    pc, conn = _build_pc(tmp_path, [src])
    try:
        await pc.startup()
        # Give the worker loop a beat to drain
        await asyncio.sleep(0.5)
        n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        assert n_files == 2
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_reactive_indexing_in_active_mode(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    pc, conn = _build_pc(tmp_path, [src])
    try:
        await pc.startup()
        await asyncio.sleep(0.3)  # let watcher attach
        new = src / "new.md"
        new.write_text("UNIQUEMARK_new this is a fresh document")
        # Wait long enough for: FSEvents → debounce 0.2s → drain tick 0.2s → index
        await asyncio.sleep(1.5)
        n_match = conn.execute(
            "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH 'UNIQUEMARK_new'"
        ).fetchone()[0]
        assert n_match >= 1
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_delete_event_unindexes(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    p = src / "doomed.md"
    p.write_text("UNIQUEMARK_doomed about to be deleted")
    pc, conn = _build_pc(tmp_path, [src])
    try:
        await pc.startup()
        await asyncio.sleep(0.3)
        assert conn.execute(
            "SELECT COUNT(*) FROM files WHERE path = ?", (str(p.resolve()),)
        ).fetchone()[0] == 1

        p.unlink()
        await asyncio.sleep(1.5)
        assert conn.execute(
            "SELECT COUNT(*) FROM files WHERE path = ?", (str(p.resolve()),)
        ).fetchone()[0] == 0
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_mode_transition_active_to_saver(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    pc, _ = _build_pc(tmp_path, [src])
    try:
        await pc.startup()
        assert pc.mode == "active"
        assert pc.watcher.is_running

        await pc.set_mode("saver")
        assert pc.mode == "saver"
        assert pc.watcher.is_running is False
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_mode_transition_saver_to_active(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    pc, _ = _build_pc(tmp_path, [src], mode="saver")
    try:
        await pc.startup()
        assert pc.mode == "saver"
        assert pc.watcher.is_running is False

        await pc.set_mode("active")
        assert pc.mode == "active"
        # Give the watcher a beat to start its observer thread
        await asyncio.sleep(0.2)
        assert pc.watcher.is_running
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_set_mode_idempotent(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    pc, _ = _build_pc(tmp_path, [src])
    try:
        await pc.startup()
        await pc.set_mode("active")  # no-op
        await pc.set_mode("active")  # no-op
        assert pc.mode == "active"
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_force_sweep_returns_stats(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    (src / "x.md").write_text("UNIQUEMARK_force_sweep content here yes")
    pc, _ = _build_pc(tmp_path, [src])
    try:
        await pc.startup()
        stats = await pc.force_sweep()
        assert "files_indexed" in stats
        assert "elapsed_s" in stats
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_status_payload_shape(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    pc, _ = _build_pc(tmp_path, [src])
    try:
        await pc.startup()
        s = pc.status()
        assert s["mode"] == "active"
        assert s["watcher_running"] is True
        assert isinstance(s["dirty_pending"], int)
        assert s["saver_interval_s"] == 3600
        assert s["directories"] == [str(src)]
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_battery_auto_switch_to_saver(tmp_path):
    """Daemon launched on battery + auto_saver enabled → starts in saver."""
    src = tmp_path / "src"; src.mkdir()
    pc, _ = _build_pc(tmp_path, [src], mode="active", auto_saver=True)

    fake_bat = type("Bat", (), {"power_plugged": False})()
    with patch("psutil.sensors_battery", return_value=fake_bat):
        try:
            await pc.startup()
            # Even though config.power.mode == "active", on-battery should flip us.
            assert pc.mode == "saver"
        finally:
            await pc.shutdown()


@pytest.mark.asyncio
async def test_battery_auto_switch_to_active(tmp_path):
    """Daemon launched on AC + auto_saver enabled → stays/becomes active."""
    src = tmp_path / "src"; src.mkdir()
    pc, _ = _build_pc(tmp_path, [src], mode="saver", auto_saver=True)

    fake_bat = type("Bat", (), {"power_plugged": True})()
    with patch("psutil.sensors_battery", return_value=fake_bat):
        try:
            await pc.startup()
            # Even though config says saver, on AC + auto should activate.
            assert pc.mode == "active"
        finally:
            await pc.shutdown()
