import struct
from semanticsd.db import connection, migrations
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.pipeline.worker import Worker
from semanticsd.pipeline.indexer import Indexer
from tests._fixtures import make_text


class FakeEmbedder(Embedder):
    provider_id = "fake"
    model_id = "fake-1"
    dim = 384
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0

    def __init__(self):
        self.calls = 0

    def embed(self, texts, kind):
        self.calls += 1
        return EmbedResult(
            vectors=[[float(i) / 1000.0] * 384 for i in range(len(texts))],
            input_tokens=sum(len(t) // 4 for t in texts),
        )

    def health_check(self):
        return (True, "ok")

    def estimate_tokens(self, texts):
        return sum(len(t) // 4 for t in texts)


def _index_one(tmp_path):
    db = tmp_path / "w.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    src = tmp_path / "src"
    src.mkdir()
    make_text(src, body="Some content to embed.")
    Indexer(conn=conn, max_file_size_mb=50).index_path(src)
    return conn


def test_drain_once_processes_pending_jobs(tmp_path):
    conn = _index_one(tmp_path)
    emb = FakeEmbedder()
    w = Worker(conn=conn, embedder=emb, batch_size=10)
    n = w.drain_once()
    assert n >= 1
    pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert pending == 0
    emb_count = conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0]
    assert emb_count >= 1


def test_dedup_skips_redundant_embed_calls(tmp_path):
    conn = _index_one(tmp_path)
    emb = FakeEmbedder()
    w = Worker(conn=conn, embedder=emb, batch_size=10)
    w.drain_once()
    calls_after_first = emb.calls

    src2 = tmp_path / "src2"
    src2.mkdir()
    make_text(src2, name="dupe.txt", body="Some content to embed.")
    Indexer(conn=conn, max_file_size_mb=50).index_path(src2)
    w.drain_once()

    assert emb.calls == calls_after_first


def test_reset_stale_resets_in_flight(tmp_path):
    conn = _index_one(tmp_path)
    conn.execute("UPDATE jobs SET status='in_flight'")
    Worker(conn=conn, embedder=FakeEmbedder()).reset_stale()
    in_flight = conn.execute("SELECT count(*) FROM jobs WHERE status='in_flight'").fetchone()[0]
    assert in_flight == 0
    pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert pending >= 1


def test_drain_once_returns_zero_when_no_jobs(tmp_path):
    db = tmp_path / "empty.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    w = Worker(conn=conn, embedder=FakeEmbedder())
    assert w.drain_once() == 0
