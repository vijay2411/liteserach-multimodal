"""End-to-end: real LocalEmbedder + indexer + worker against ./sandbox/.

Slow-marked because it loads the real bge-small-en-v1.5 model and embeds
multiple sandbox files.
"""
import pytest
from pathlib import Path
from semanticsd.db import connection, migrations
from semanticsd.embedders.local import LocalEmbedder
from semanticsd.pipeline.indexer import Indexer
from semanticsd.pipeline.worker import Worker


pytestmark = pytest.mark.slow


def test_index_sandbox_end_to_end(tmp_path):
    """Run the full pipeline against the sandbox seed corpus."""
    sandbox = Path(__file__).resolve().parents[1] / "sandbox"
    assert sandbox.exists(), "sandbox/ should exist (Plan 2 Task 2 fixture)"

    db = tmp_path / "e2e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    idx = Indexer(conn=conn, max_file_size_mb=50)
    stats = idx.index_path(sandbox)
    assert stats["files_indexed"] >= 4
    assert stats["chunks_created"] >= stats["files_indexed"]
    assert stats["jobs_queued"] == stats["chunks_created"]

    embedder = LocalEmbedder()
    worker = Worker(conn=conn, embedder=embedder, batch_size=128)

    total_drained = 0
    while True:
        n = worker.drain_once()
        total_drained += n
        if n == 0:
            break
    assert total_drained == stats["jobs_queued"]

    emb_count = conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0]
    assert emb_count == stats["chunks_created"]
    pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert pending == 0
