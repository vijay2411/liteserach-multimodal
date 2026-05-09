"""End-to-end: index → usage rows accumulate → switch provider → reembed →
new vectors land in the new (provider, model, dim) bucket.

Stubs both embedders so this test runs without external services.
"""
import time
from semanticsd.db import connection, migrations
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.router import EmbedderRouter
from semanticsd.pipeline.indexer import Indexer
from semanticsd.pipeline.worker import Worker
from semanticsd.reembed import queue_reembed
from semanticsd.usage.budget import BudgetGate
from semanticsd.usage.reports import totals
from tests._fixtures import make_text


class _ProviderA(Embedder):
    """768-d, $1/M tokens. Pretends each call uses 100k tokens."""
    provider_id = "providerA"; model_id = "model-A"; dim = 768
    supports_kind = False
    cost_per_million_input_tokens_usd = 1.0
    def embed(self, texts, kind):
        return EmbedResult(
            vectors=[[0.1] * 768 for _ in texts],
            input_tokens=100_000 * len(texts),
        )
    def health_check(self): return (True, "ok")
    def estimate_tokens(self, texts): return 100_000 * len(texts)


class _ProviderB(Embedder):
    """1024-d, free. Different (provider, model, dim) from A."""
    provider_id = "providerB"; model_id = "model-B"; dim = 1024
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0
    def embed(self, texts, kind):
        return EmbedResult(
            vectors=[[0.2] * 1024 for _ in texts],
            input_tokens=50_000 * len(texts),
        )
    def health_check(self): return (True, "ok")
    def estimate_tokens(self, texts): return 50_000 * len(texts)


def test_full_lifecycle_index_usage_switch_reembed(tmp_path):
    db = tmp_path / "lc.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    src = tmp_path / "corpus"; src.mkdir()
    for name, body in [
        ("alpha.md", "Alpha contents about apples and oranges"),
        ("beta.md",  "Beta talks about cats and dogs"),
        ("gamma.md", "Gamma describes mountains and rivers"),
    ]:
        make_text(src, name=name, body=body)

    # 1. Index with provider A
    Indexer(conn=conn, max_file_size_mb=10).index_path(src)
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert n_chunks == 3

    router_a = EmbedderRouter(text=_ProviderA())
    Worker(conn=conn, router=router_a, batch_size=10).drain_once()

    # All 3 chunks should now have ProviderA embeddings.
    n_meta_a = conn.execute(
        "SELECT COUNT(*) FROM embedding_meta WHERE provider_id='providerA'"
    ).fetchone()[0]
    assert n_meta_a == 3

    # Usage row(s) should have been written. ProviderA at $1/M with
    # 100k tokens per chunk = $0.10 per chunk; we ran ONE batch with
    # 3 chunks → $0.30 single row.
    t = totals(conn)
    assert t.calls == 1
    assert t.chunks == 3
    assert abs(t.cost_usd - 0.3) < 1e-9

    # 2. Now "switch provider" to B and run reembed.
    router_b = EmbedderRouter(text=_ProviderB())
    counts = queue_reembed(conn, router_b, modality="text")
    assert counts["text"] == 3   # all chunks lack ProviderB vectors

    # Drain — worker uses router_b, so jobs land in ProviderB's table.
    n_drained = Worker(conn=conn, router=router_b, batch_size=10).drain_once()
    assert n_drained == 3

    # Both providers' embeddings now coexist.
    n_meta_a = conn.execute(
        "SELECT COUNT(*) FROM embedding_meta WHERE provider_id='providerA'"
    ).fetchone()[0]
    n_meta_b = conn.execute(
        "SELECT COUNT(*) FROM embedding_meta WHERE provider_id='providerB'"
    ).fetchone()[0]
    # Note: embedding_meta has chunk_id PK, so the latest UPSERT replaces.
    # ProviderB's INSERT OR REPLACE wins for each chunk → 3 rows for B, 0 for A.
    # But the vec_text_embeddings table still has A's vectors (rowid != PK there).
    assert n_meta_b == 3
    # Both vec tables are populated: 768-d (canonical) for A, 1024-d (suffixed) for B.
    n_vec_a = conn.execute("SELECT COUNT(*) FROM vec_text_embeddings").fetchone()[0]
    n_vec_b = conn.execute("SELECT COUNT(*) FROM vec_text_embeddings_1024").fetchone()[0]
    assert n_vec_a == 3
    assert n_vec_b == 3

    # 3. Re-running reembed should be a no-op (all chunks now have B).
    counts2 = queue_reembed(conn, router_b, modality="text")
    assert counts2["text"] == 0


def test_budget_gate_short_circuits_pricey_batch(tmp_path):
    db = tmp_path / "bg.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    src = tmp_path / "corpus"; src.mkdir()
    for i in range(3):
        make_text(src, name=f"f{i}.md",
                  body=f"sample content for file {i} with some words to embed.")
    Indexer(conn=conn, max_file_size_mb=10).index_path(src)

    # Pre-charge to within $0.10 of cap. Each ProviderA call costs $0.10/chunk
    # × 3 chunks = $0.30 — so the next call would breach.
    conn.execute(
        "INSERT INTO usage(timestamp, provider_id, model_id, operation, "
        "input_tokens, cost_usd, chunk_count, duration_ms) "
        "VALUES (?, 'providerA', 'model-A', 'text_embed', 1, 0.95, 1, 50)",
        (int(time.time()),),
    )
    gate = BudgetGate(conn, monthly_limit_usd=1.0)
    router = EmbedderRouter(text=_ProviderA())
    Worker(conn=conn, router=router, batch_size=10, budget_gate=gate).drain_once()

    # Jobs should be marked failed/pending with budget_exceeded, no embeds done.
    n_meta = conn.execute(
        "SELECT COUNT(*) FROM embedding_meta WHERE provider_id='providerA'"
    ).fetchone()[0]
    assert n_meta == 0
    err_rows = conn.execute(
        "SELECT last_error FROM jobs WHERE last_error LIKE '%budget_exceeded%'"
    ).fetchall()
    assert len(err_rows) >= 1
