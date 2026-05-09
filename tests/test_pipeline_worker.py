"""Worker — modality-aware routing via EmbedderRouter."""
from semanticsd.db import connection, migrations
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.vision_base import VisionEmbedder
from semanticsd.embedders.router import EmbedderRouter
from semanticsd.pipeline.worker import Worker
from semanticsd.pipeline.indexer import Indexer
from tests._fixtures import make_text


class FakeTextEmbedder(Embedder):
    provider_id = "fake"
    model_id = "fake-1"
    dim = 768
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0

    def __init__(self):
        self.calls = 0

    def embed(self, texts, kind):
        self.calls += 1
        return EmbedResult(
            vectors=[[float(i) / 1000.0] * 768 for i in range(len(texts))],
            input_tokens=sum(len(t) // 4 for t in texts),
        )

    def health_check(self):
        return (True, "ok")

    def estimate_tokens(self, texts):
        return sum(len(t) // 4 for t in texts)


class FakeVisionEmbedder(VisionEmbedder):
    provider_id = "fakev"
    model_id = "fake-vision-1"
    dim = 3072

    def __init__(self):
        self.calls = 0

    def embed_images(self, images, kind="doc"):
        self.calls += 1
        return EmbedResult(
            vectors=[[0.1] * 3072 for _ in images],
            input_tokens=len(images),
        )

    def health_check(self):
        return (True, "ok")

    def estimate_image_tokens(self, images):
        return len(images)


def _router_text_only():
    return EmbedderRouter(text=FakeTextEmbedder())


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
    router = _router_text_only()
    w = Worker(conn=conn, router=router, batch_size=10)
    n = w.drain_once()
    assert n >= 1
    pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert pending == 0
    emb_count = conn.execute("SELECT count(*) FROM vec_text_embeddings").fetchone()[0]
    assert emb_count >= 1


def test_dedup_skips_redundant_embed_calls(tmp_path):
    conn = _index_one(tmp_path)
    text_em = FakeTextEmbedder()
    router = EmbedderRouter(text=text_em)
    w = Worker(conn=conn, router=router, batch_size=10)
    w.drain_once()
    calls_after_first = text_em.calls

    src2 = tmp_path / "src2"
    src2.mkdir()
    make_text(src2, name="dupe.txt", body="Some content to embed.")
    Indexer(conn=conn, max_file_size_mb=50).index_path(src2)
    w.drain_once()

    assert text_em.calls == calls_after_first


def test_reset_stale_resets_in_flight(tmp_path):
    conn = _index_one(tmp_path)
    conn.execute("UPDATE jobs SET status='in_flight'")
    Worker(conn=conn, router=_router_text_only()).reset_stale()
    in_flight = conn.execute("SELECT count(*) FROM jobs WHERE status='in_flight'").fetchone()[0]
    assert in_flight == 0
    pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert pending >= 1


def test_drain_once_returns_zero_when_no_jobs(tmp_path):
    db = tmp_path / "empty.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    w = Worker(conn=conn, router=_router_text_only())
    assert w.drain_once() == 0


def test_worker_routes_vision_to_vision_embedder(tmp_path):
    """Mixed batch: text chunks go to text embedder, vision chunks to vision embedder."""
    db = tmp_path / "mm.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    conn.execute(
        "INSERT INTO files(path, modified_at, size, file_type, indexed_at) "
        "VALUES('/x', 1, 1, 'inline', 1)"
    )
    fid = int(conn.execute("SELECT id FROM files").fetchone()[0])
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, content_hash, byte_start, byte_end, modality) "
        "VALUES (?, 0, 'hi', 'h_text_1', 0, 2, 'text')",
        (fid,),
    )
    cid_t = int(conn.execute("SELECT id FROM chunks WHERE chunk_index=0").fetchone()[0])
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, content_hash, byte_start, byte_end, modality, image_blob) "
        "VALUES (?, 1, '<img>', 'h_vis_1', 0, 5, 'vision', ?)",
        (fid, b"\x89PNG_FAKE"),
    )
    cid_v = int(conn.execute("SELECT id FROM chunks WHERE chunk_index=1").fetchone()[0])
    conn.execute("INSERT INTO jobs(chunk_id,status,attempts,created_at,updated_at) VALUES(?,'pending',0,1,1)", (cid_t,))
    conn.execute("INSERT INTO jobs(chunk_id,status,attempts,created_at,updated_at) VALUES(?,'pending',0,1,1)", (cid_v,))

    text_em = FakeTextEmbedder()
    vis_em = FakeVisionEmbedder()
    router = EmbedderRouter(text=text_em, vision=vis_em)
    w = Worker(conn, router=router)
    processed = w.drain_once()
    assert processed == 2

    assert text_em.calls == 1
    assert vis_em.calls == 1

    assert conn.execute("SELECT COUNT(*) FROM vec_text_embeddings").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM vec_vision_embeddings").fetchone()[0] == 1

    # embedding_meta carries modality
    rows = conn.execute("SELECT modality FROM embedding_meta ORDER BY chunk_id").fetchall()
    assert {r[0] for r in rows} == {"text", "vision"}


def test_worker_skips_modality_with_no_embedder(tmp_path):
    """Vision job with text-only router fails the job, leaves text untouched."""
    db = tmp_path / "skip.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    conn.execute(
        "INSERT INTO files(path, modified_at, size, file_type, indexed_at) "
        "VALUES('/y', 1, 1, 'inline', 1)"
    )
    fid = int(conn.execute("SELECT id FROM files").fetchone()[0])
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, content_hash, byte_start, byte_end, modality, image_blob) "
        "VALUES (?, 0, '<img>', 'h_v', 0, 5, 'vision', ?)",
        (fid, b"\x89PNG"),
    )
    cid = int(conn.execute("SELECT id FROM chunks").fetchone()[0])
    conn.execute("INSERT INTO jobs(chunk_id,status,attempts,created_at,updated_at) VALUES(?,'pending',0,1,1)", (cid,))

    router = EmbedderRouter(text=FakeTextEmbedder(), vision=None)
    w = Worker(conn, router=router)
    w.drain_once()

    err = conn.execute("SELECT last_error FROM jobs").fetchone()[0]
    assert err and "no_embedder_for_modality" in err


# --- usage recording + budget gate ---

class _PaidText(Embedder):
    """A deterministic 'paid' embedder that costs $1/M input tokens."""
    provider_id = "paid"; model_id = "p1"; dim = 768
    supports_kind = False
    cost_per_million_input_tokens_usd = 1.0  # $1 per million tokens

    def embed(self, texts, kind):
        # Pretend each text is 1M tokens — so each call costs $1.
        return EmbedResult(
            vectors=[[0.5] * 768 for _ in texts],
            input_tokens=1_000_000 * len(texts),
        )

    def health_check(self): return (True, "ok")
    def estimate_tokens(self, texts): return 1_000_000 * len(texts)


def test_worker_writes_usage_row(tmp_path):
    """A successful embed batch should insert exactly one row into usage."""
    conn = _index_one(tmp_path)
    text_em = _PaidText()
    router = EmbedderRouter(text=text_em)
    w = Worker(conn=conn, router=router, batch_size=10)
    w.drain_once()

    rows = conn.execute(
        "SELECT provider_id, model_id, operation, cost_usd, chunk_count "
        "FROM usage"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "paid"
    assert rows[0][2] == "text_embed"
    assert rows[0][3] > 0.0  # paid provider should record positive cost
    assert rows[0][4] >= 1


def test_worker_respects_budget_gate(tmp_path):
    """When the gate refuses a batch, jobs should fail with budget_exceeded."""
    from semanticsd.usage.budget import BudgetGate

    conn = _index_one(tmp_path)
    # Pre-charge the month's usage to $4.00, with a $5 cap. Next call would
    # add another $1+ and push us over.
    import time as _t
    conn.execute(
        "INSERT INTO usage(timestamp, provider_id, model_id, operation, "
        "input_tokens, cost_usd, chunk_count, duration_ms) "
        "VALUES (?, 'paid', 'p1', 'text_embed', 1, 4.5, 1, 100)",
        (int(_t.time()),),
    )

    gate = BudgetGate(conn, monthly_limit_usd=5.0)
    text_em = _PaidText()
    router = EmbedderRouter(text=text_em)
    w = Worker(conn=conn, router=router, batch_size=10, budget_gate=gate)
    w.drain_once()

    failed = conn.execute(
        "SELECT last_error FROM jobs WHERE status='failed'"
    ).fetchone()
    pending = conn.execute(
        "SELECT last_error FROM jobs WHERE status='pending'"
    ).fetchone()
    # The job either landed in failed (max_attempts=1) or was bumped back to
    # pending with the budget_exceeded error message.
    err = (failed and failed[0]) or (pending and pending[0])
    assert err and "budget_exceeded" in err


def test_worker_local_provider_bypasses_budget(tmp_path):
    """Free providers (rate=0) must not be blocked by the gate."""
    from semanticsd.usage.budget import BudgetGate

    conn = _index_one(tmp_path)
    # Already over cap.
    import time as _t
    conn.execute(
        "INSERT INTO usage(timestamp, provider_id, model_id, operation, "
        "input_tokens, cost_usd, chunk_count, duration_ms) "
        "VALUES (?, 'paid', 'p1', 'text_embed', 1, 100.0, 1, 100)",
        (int(_t.time()),),
    )
    gate = BudgetGate(conn, monthly_limit_usd=5.0)
    # FakeTextEmbedder has cost_per_million=0 (free)
    router = EmbedderRouter(text=FakeTextEmbedder())
    w = Worker(conn=conn, router=router, batch_size=10, budget_gate=gate)
    n = w.drain_once()
    assert n >= 1  # processed despite over-cap
