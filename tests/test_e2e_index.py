"""End-to-end: real OllamaEmbedder + indexer + worker against ./sandbox/.

Slow-marked because it requires Ollama running locally with the
embeddinggemma model pulled. Skipped otherwise.
"""
import socket
import pytest
from pathlib import Path
from semanticsd.db import connection, migrations
from semanticsd.embedders.ollama import OllamaEmbedder
from semanticsd.embedders.router import EmbedderRouter
from semanticsd.pipeline.indexer import Indexer
from semanticsd.pipeline.worker import Worker


def _ollama_up() -> bool:
    s = socket.socket()
    try:
        s.settimeout(0.5)
        s.connect(("localhost", 11434))
        return True
    except OSError:
        return False
    finally:
        s.close()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _ollama_up(), reason="ollama not running"),
]


def test_index_sandbox_end_to_end(tmp_path):
    """Run the full pipeline against the sandbox seed corpus."""
    sandbox = Path(__file__).resolve().parents[1] / "sandbox"
    assert sandbox.exists(), "sandbox/ should exist"

    db = tmp_path / "e2e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    idx = Indexer(conn=conn, max_file_size_mb=50)
    stats = idx.index_path(sandbox)
    assert stats["files_indexed"] >= 4
    assert stats["chunks_created"] >= stats["files_indexed"]
    assert stats["jobs_queued"] == stats["chunks_created"]

    embedder = OllamaEmbedder(model="embeddinggemma")
    router = EmbedderRouter(text=embedder)
    worker = Worker(conn=conn, router=router, batch_size=64)

    total_drained = 0
    while True:
        n = worker.drain_once()
        total_drained += n
        if n == 0:
            break
    assert total_drained == stats["jobs_queued"]

    emb_count = conn.execute("SELECT count(*) FROM vec_text_embeddings").fetchone()[0]
    assert emb_count == stats["chunks_created"]
    pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert pending == 0
