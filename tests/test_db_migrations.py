"""Schema migration tests."""
from semanticsd.db import connection, migrations
from semanticsd import paths


def test_v3_adds_modality_and_vec_tables(tmp_app_support):
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)

    cols_chunks = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
    assert "modality" in cols_chunks
    assert "image_blob" in cols_chunks

    cols_meta = {r[1] for r in conn.execute("PRAGMA table_info(embedding_meta)")}
    assert "modality" in cols_meta

    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','virtual') OR sql LIKE '%VIRTUAL TABLE%'"
    )}
    # sqlite stores virtual tables in sqlite_master with type='table'
    all_tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert "vec_text_embeddings" in all_tables
    assert "vec_vision_embeddings" in all_tables


def test_migrations_idempotent(tmp_app_support):
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    migrations.apply(conn)  # should be no-op
    migrations.apply(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
    assert "modality" in cols


def test_schema_version_recorded(tmp_app_support):
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    from semanticsd.db.schema import SCHEMA_VERSION
    assert int(row[0]) == SCHEMA_VERSION
