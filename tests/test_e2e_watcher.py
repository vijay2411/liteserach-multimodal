"""End-to-end: full daemon lifecycle with FSEvents watcher.

Spins up a real PowerController in active mode against a tmp watch dir,
drops a new file in, and verifies it becomes searchable through the
real engine within a few seconds. Stubs out the embedder with a
deterministic fake so this test runs without Ollama/Gemini.
"""
import asyncio
import time
from pathlib import Path
import pytest

from semanticsd.config import Config, EmbeddingConfig, TextEmbeddingConfig, WatchConfig, PowerConfig
from semanticsd.db import connection, migrations
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.router import EmbedderRouter
from semanticsd.pipeline.indexer import Indexer
from semanticsd.pipeline.worker import Worker
from semanticsd.search.engine import Engine
from semanticsd.search.types import SearchOptions
from semanticsd.watcher.power import PowerController


class _DeterministicText(Embedder):
    """Returns a unique vector per text content (hash of len+first chars).
    Different texts → different vectors → semantic search returns the right one."""
    provider_id = "fake_det"; model_id = "f"; dim = 768
    supports_kind = False; cost_per_million_input_tokens_usd = 0.0

    def embed(self, texts, kind):
        vectors = []
        for t in texts:
            # Embed by spreading hash bits into the first 32 dims.
            h = hash(t) & 0xFFFFFFFF
            v = [0.0] * 768
            for i in range(32):
                v[i] = 1.0 if (h >> i) & 1 else 0.0
            # L2-normalize so cosine works
            mag = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / mag for x in v])
        return EmbedResult(vectors=vectors, input_tokens=len(texts))

    def health_check(self): return (True, "ok")
    def estimate_tokens(self, texts): return len(texts)


@pytest.mark.asyncio
async def test_watcher_indexes_new_file_and_makes_it_searchable(tmp_path):
    src = tmp_path / "watched"
    src.mkdir()

    db = tmp_path / "e2e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    indexer = Indexer(conn=conn, max_file_size_mb=10)
    router = EmbedderRouter(text=_DeterministicText())
    worker = Worker(conn=conn, router=router, batch_size=64)
    cfg = Config(
        watch=WatchConfig(directories=[str(src)]),
        embedding=EmbeddingConfig(text=TextEmbeddingConfig(preset="local")),
        power=PowerConfig(mode="active", auto_saver_on_battery=False),
    )
    pc = PowerController(cfg, indexer, worker, debounce_s=0.2)
    engine = Engine(conn, router)

    try:
        await pc.startup()
        # Initial state: no files
        assert engine.search("ZZUNIQUEMARKZZ_alpha", SearchOptions(all=True, mode="grep")) == []

        # Add a file under the watch dir
        new_file = src / "alpha.md"
        new_file.write_text("ZZUNIQUEMARKZZ_alpha contents that are searchable")

        # Wait for: FSEvent → debounce → drain → indexer → fts/vec → worker
        deadline = time.monotonic() + 5.0
        found = False
        while time.monotonic() < deadline:
            results = engine.search(
                "ZZUNIQUEMARKZZ_alpha",
                SearchOptions(all=True, mode="grep", limit=5),
            )
            if results and any("alpha.md" in r.path for r in results):
                found = True
                break
            await asyncio.sleep(0.2)

        assert found, "new file did not become searchable within 5s"
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_watcher_unindexes_deleted_file(tmp_path):
    src = tmp_path / "watched"; src.mkdir()
    p = src / "doomed.md"
    p.write_text("ZZUNIQUEMARKZZ_doomed about to be removed")

    db = tmp_path / "e2e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    indexer = Indexer(conn=conn, max_file_size_mb=10)
    router = EmbedderRouter(text=_DeterministicText())
    worker = Worker(conn=conn, router=router, batch_size=64)
    cfg = Config(
        watch=WatchConfig(directories=[str(src)]),
        embedding=EmbeddingConfig(text=TextEmbeddingConfig(preset="local")),
        power=PowerConfig(mode="active", auto_saver_on_battery=False),
    )
    pc = PowerController(cfg, indexer, worker, debounce_s=0.2)
    engine = Engine(conn, router)

    try:
        await pc.startup()
        # The file should be in the index from the initial sweep
        await asyncio.sleep(0.5)
        before = engine.search("ZZUNIQUEMARKZZ_doomed",
                               SearchOptions(all=True, mode="grep", limit=5))
        assert before, "initial sweep should have indexed doomed.md"

        # Now delete and wait
        p.unlink()
        deadline = time.monotonic() + 5.0
        gone = False
        while time.monotonic() < deadline:
            results = engine.search("ZZUNIQUEMARKZZ_doomed",
                                    SearchOptions(all=True, mode="grep", limit=5))
            if not results:
                gone = True
                break
            await asyncio.sleep(0.2)
        assert gone, "deleted file remained searchable after 5s"
    finally:
        await pc.shutdown()


@pytest.mark.asyncio
async def test_saver_mode_does_not_react_to_changes_immediately(tmp_path):
    """In saver mode, FSEvents is OFF — a new file should NOT appear until the
    next periodic sweep. We use a 60s saver interval so the periodic sweep
    won't fire during this test, and verify the file stays un-indexed."""
    src = tmp_path / "watched"; src.mkdir()

    db = tmp_path / "e2e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    indexer = Indexer(conn=conn, max_file_size_mb=10)
    router = EmbedderRouter(text=_DeterministicText())
    worker = Worker(conn=conn, router=router, batch_size=64)
    cfg = Config(
        watch=WatchConfig(directories=[str(src)]),
        embedding=EmbeddingConfig(text=TextEmbeddingConfig(preset="local")),
        power=PowerConfig(mode="saver", saver_reindex_interval="60s",
                          auto_saver_on_battery=False),
    )
    pc = PowerController(cfg, indexer, worker, debounce_s=0.2)
    engine = Engine(conn, router)

    try:
        await pc.startup()
        assert pc.mode == "saver"
        assert pc.watcher.is_running is False

        # Drop a file — watcher is off, so it should NOT be picked up.
        new_file = src / "lazy.md"
        new_file.write_text("ZZUNIQUEMARKZZ_lazy file dropped during saver mode")
        await asyncio.sleep(2.0)
        results = engine.search("ZZUNIQUEMARKZZ_lazy",
                                SearchOptions(all=True, mode="grep", limit=5))
        assert results == [], "saver mode should not have reactively indexed the file"

        # But a manual sweep should pick it up.
        await pc.force_sweep()
        await asyncio.sleep(0.5)
        results = engine.search("ZZUNIQUEMARKZZ_lazy",
                                SearchOptions(all=True, mode="grep", limit=5))
        assert results, "force_sweep should have indexed the file"
    finally:
        await pc.shutdown()
