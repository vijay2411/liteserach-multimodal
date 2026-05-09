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

    _seed_text_chunk(conn, 1, 1, "/x/a.md", "alpha", v_close)
    _seed_text_chunk(conn, 2, 2, "/x/b.md", "beta", v_far)
    _seed_text_chunk(conn, 3, 3, "/x/c.md", "gamma", v_other)

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
