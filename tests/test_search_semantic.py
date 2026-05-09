"""Semantic search over per-modality vec0 tables."""
import struct
from semanticsd.db import connection, migrations
from semanticsd.search.semantic import (
    search_semantic_text,
    search_semantic_vision,
    _vision_vec_tables_for_dim,
)


def _vec_blob(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _seed_text_chunk(conn, chunk_id, file_id, path, text, vector: list[float]):
    conn.execute(
        "INSERT OR IGNORE INTO files(id, path, modified_at, size, file_type, indexed_at) "
        "VALUES (?, ?, 1, 1, 'text', 1)",
        (file_id, path),
    )
    conn.execute(
        "INSERT INTO chunks(id, file_id, chunk_index, text, content_hash, "
        "byte_start, byte_end, modality) VALUES (?, ?, 0, ?, 'h', 0, ?, 'text')",
        (chunk_id, file_id, text, len(text)),
    )
    conn.execute(
        "INSERT INTO vec_text_embeddings(rowid, embedding) VALUES (?, ?)",
        (chunk_id, _vec_blob(vector)),
    )


def test_semantic_text_returns_nearest(tmp_path):
    db = tmp_path / "s.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    # Three chunks; nearest to query [1,0,...] is chunk 1 (also [1,0,...]).
    base = [0.0] * 768
    v_close = [1.0] + [0.0] * 767
    v_far = [-1.0] + [0.0] * 767
    v_other = [0.0, 1.0] + [0.0] * 766

    _seed_text_chunk(conn, 1, 1, "/x/a.md", "alpha contents about foxes", v_close)
    _seed_text_chunk(conn, 2, 2, "/x/b.md", "beta contents about cats and dogs", v_far)
    _seed_text_chunk(conn, 3, 3, "/x/c.md", "gamma contents about lazy mornings", v_other)

    results = search_semantic_text(conn, query_vec=v_close, limit=3)
    assert len(results) == 3
    assert results[0].chunk_id == 1
    assert results[0].mode == "semantic"
    assert results[0].modality == "text"


def test_semantic_text_empty_query(tmp_path):
    db = tmp_path / "s.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    assert search_semantic_text(conn, query_vec=[], limit=5) == []


def test_vision_table_resolution(tmp_path):
    db = tmp_path / "s.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    # default vec_vision_embeddings (3072-d) exists from migration
    assert "vec_vision_embeddings" in _vision_vec_tables_for_dim(conn, 3072)
    # 2048-d table doesn't exist yet -> empty
    assert _vision_vec_tables_for_dim(conn, 2048) == []
    # Create one and re-check.
    conn.execute("CREATE VIRTUAL TABLE vec_vision_embeddings_2048 USING vec0(embedding FLOAT[2048])")
    assert _vision_vec_tables_for_dim(conn, 2048) == ["vec_vision_embeddings_2048"]


def test_semantic_vision_returns_results(tmp_path):
    db = tmp_path / "s.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    # Seed a 2048-d vision chunk
    conn.execute("CREATE VIRTUAL TABLE vec_vision_embeddings_2048 USING vec0(embedding FLOAT[2048])")
    conn.execute(
        "INSERT INTO files(id, path, modified_at, size, file_type, indexed_at) "
        "VALUES (1, '/x/img.png', 1, 1, 'image', 1)"
    )
    conn.execute(
        "INSERT INTO chunks(id, file_id, chunk_index, text, content_hash, "
        "byte_start, byte_end, modality, image_blob) "
        "VALUES (1, 1, 0, '<image: img.png>', 'hh', 0, 16, 'vision', X'89504e47')"
    )
    v = [1.0] + [0.0] * 2047
    conn.execute(
        "INSERT INTO vec_vision_embeddings_2048(rowid, embedding) VALUES (1, ?)",
        (_vec_blob(v),),
    )

    results = search_semantic_vision(conn, query_vec=v, dim=2048, limit=5)
    assert len(results) == 1
    assert results[0].modality == "vision"
    assert results[0].metadata["vec_table"] == "vec_vision_embeddings_2048"


def test_semantic_drops_short_chunks(tmp_path):
    """Chunks below MIN_TEXT_LEN should not surface even when vec MATCH ranks them top."""
    db = tmp_path / "s.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    v = [1.0] + [0.0] * 767
    _seed_text_chunk(conn, 1, 1, "/x/empty.json", "{}", v)  # 2 chars
    _seed_text_chunk(conn, 2, 2, "/x/real.md", "this is a real document about something interesting", v)

    results = search_semantic_text(conn, query_vec=v, limit=5)
    paths = [r.path for r in results]
    assert "/x/empty.json" not in paths
    assert "/x/real.md" in paths


def test_semantic_score_is_cosine_in_zero_to_one(tmp_path):
    """For unit-normalized vectors, score should be in [0, 1] not the
    legacy 1 - L2 (which can go negative)."""
    db = tmp_path / "s.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    v_a = [1.0] + [0.0] * 767
    v_b = [0.0, 1.0] + [0.0] * 766
    _seed_text_chunk(conn, 1, 1, "/x/a.md", "first document about quantum mechanics", v_a)
    _seed_text_chunk(conn, 2, 2, "/x/b.md", "second document about classical music", v_b)

    results = search_semantic_text(conn, query_vec=v_a, limit=5)
    assert len(results) == 2
    # Top result is identical -> cos_sim = 1.0
    assert results[0].score > 0.99
    # Second is orthogonal -> cos_sim ≈ 0.0
    assert -0.01 < results[1].score < 0.01
