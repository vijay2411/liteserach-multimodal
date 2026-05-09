from semanticsd.pipeline.hasher import sha256_hex, normalize_for_hash, find_existing_embedding
from semanticsd.db import connection, migrations


def test_sha256_hex_is_deterministic():
    assert sha256_hex("hello") == sha256_hex("hello")


def test_sha256_hex_changes_with_content():
    assert sha256_hex("hello") != sha256_hex("world")


def test_normalize_collapses_whitespace_and_lowercases():
    a = normalize_for_hash("  Hello\nWorld  \t")
    b = normalize_for_hash("hello world")
    assert a == b


def test_find_existing_embedding_returns_none_when_absent(tmp_path):
    db = tmp_path / "h.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    res = find_existing_embedding(conn, content_hash="abc", provider_id="local",
                                  model_id="m", dim=384)
    assert res is None


def test_find_existing_embedding_returns_chunk_id_when_present(tmp_path):
    """Insert a fake chunk + embedding_meta + vec_embeddings, then look it up."""
    import struct
    db = tmp_path / "h.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    conn.execute(
        "INSERT INTO files(id, path, modified_at, size, file_type, indexed_at) "
        "VALUES (1, '/x', 0, 0, 'text', 0)"
    )
    conn.execute(
        "INSERT INTO chunks(id, file_id, chunk_index, text, content_hash, byte_start, byte_end) "
        "VALUES (1, 1, 0, 'hello', 'abc', 0, 5)"
    )
    blob = struct.pack("384f", *([0.1] * 384))
    conn.execute("INSERT INTO vec_embeddings(rowid, embedding) VALUES (1, ?)", (blob,))
    conn.execute(
        "INSERT INTO embedding_meta(chunk_id, provider_id, model_id, dim, content_hash) "
        "VALUES (1, 'local', 'm', 384, 'abc')"
    )
    res = find_existing_embedding(conn, content_hash="abc", provider_id="local",
                                  model_id="m", dim=384)
    assert res == 1
