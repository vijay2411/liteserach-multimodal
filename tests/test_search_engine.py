"""Search Engine — orchestrates per-mode + applies CWD + fuses."""
import struct
import pytest
from pathlib import Path
from semanticsd.db import connection, migrations
from semanticsd.search.engine import Engine
from semanticsd.search.types import SearchOptions
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.router import EmbedderRouter


class FakeText(Embedder):
    provider_id = "fake_t"
    model_id = "ft"
    dim = 768
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0

    def embed(self, texts, kind):
        # Simple bag-of-chars vector: position 0 = len, rest zero — deterministic
        return EmbedResult(
            vectors=[[float(len(t))] + [0.0] * 767 for t in texts],
            input_tokens=1,
        )

    def health_check(self):
        return (True, "ok")

    def estimate_tokens(self, texts):
        return 1


def _vec_blob(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _seed(conn):
    files = [
        (1, "/cwd/notes/alpha.md", "alpha contents about FOXes"),
        (2, "/cwd/notes/beta.md",  "beta talks about cats and dogs"),
        (3, "/other/zeta.md",      "fox content not in cwd"),
    ]
    for fid, path, _ in files:
        conn.execute(
            "INSERT INTO files(id, path, modified_at, size, file_type, indexed_at) "
            "VALUES (?, ?, 1, 1, 'text', 1)",
            (fid, path),
        )
        conn.execute("INSERT INTO fts_paths(rowid, path) VALUES (?, ?)", (fid, path))

    for chunk_id, (fid, _, text) in enumerate(files, start=1):
        conn.execute(
            "INSERT INTO chunks(id, file_id, chunk_index, text, content_hash, "
            "byte_start, byte_end, modality) VALUES (?, ?, 0, ?, ?, 0, ?, 'text')",
            (chunk_id, fid, text, f"h{chunk_id}", len(text)),
        )
        conn.execute("INSERT INTO fts_chunks(rowid, text) VALUES (?, ?)", (chunk_id, text))
        # Vector: position 0 = len(text), rest zero — same as FakeText embed
        vec = [float(len(text))] + [0.0] * 767
        conn.execute(
            "INSERT INTO vec_text_embeddings(rowid, embedding) VALUES (?, ?)",
            (chunk_id, _vec_blob(vec)),
        )


def test_engine_hybrid_returns_results(tmp_path):
    db = tmp_path / "e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn)

    router = EmbedderRouter(text=FakeText())
    engine = Engine(conn, router)
    results = engine.search("alpha contents about FOXes", SearchOptions(all=True, mode="hybrid"))
    assert len(results) >= 1
    # The chunk whose text matches BOTH semantically (same vector via len match)
    # and lexically (FTS) should rank top.
    assert results[0].file_id == 1


def test_engine_cwd_filter(tmp_path):
    db = tmp_path / "e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn)

    router = EmbedderRouter(text=FakeText())
    engine = Engine(conn, router)
    results = engine.search("fox", SearchOptions(cwd=Path("/cwd"), mode="grep"))
    paths = {r.path for r in results}
    assert "/cwd/notes/alpha.md" in paths
    assert "/other/zeta.md" not in paths


def test_engine_filename_only(tmp_path):
    db = tmp_path / "e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn)

    router = EmbedderRouter(text=FakeText())
    engine = Engine(conn, router)
    results = engine.search("alpha", SearchOptions(all=True, mode="filename"))
    assert len(results) == 1
    assert results[0].mode == "filename"


def test_engine_empty_query(tmp_path):
    db = tmp_path / "e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn)

    router = EmbedderRouter(text=FakeText())
    engine = Engine(conn, router)
    assert engine.search("", SearchOptions()) == []


def test_engine_caches_query_embedding(tmp_path):
    """Repeated searches reuse the cached query embedding."""
    db = tmp_path / "e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn)

    text_em = FakeText()
    call_count = {"n": 0}

    original_embed = text_em.embed

    def counting_embed(texts, kind):
        call_count["n"] += 1
        return original_embed(texts, kind)

    text_em.embed = counting_embed
    router = EmbedderRouter(text=text_em)
    engine = Engine(conn, router)

    engine.search("alpha", SearchOptions(all=True, mode="semantic"))
    n1 = call_count["n"]
    engine.search("alpha", SearchOptions(all=True, mode="semantic"))
    assert call_count["n"] == n1, "second call should hit cache"


def test_engine_records_query_usage(tmp_path):
    """Each query that touches a paid text embedder should write a usage row."""
    db = tmp_path / "e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn)

    class _Paid(FakeText):
        provider_id = "paid_q"; model_id = "pq"
        cost_per_million_input_tokens_usd = 0.5

    text_em = _Paid()
    router = EmbedderRouter(text=text_em)
    engine = Engine(conn, router)
    engine.search("alpha contents about FOXes",
                  SearchOptions(all=True, mode="semantic"))

    rows = conn.execute(
        "SELECT provider_id, operation, chunk_count FROM usage"
    ).fetchall()
    assert len(rows) >= 1
    assert any(r[0] == "paid_q" and r[1] == "query_embed" for r in rows)
