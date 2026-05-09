import sqlite3
from semanticsd.db import connection, migrations, schema


def test_apply_migrations_creates_tables(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    for required in ("files", "chunks", "jobs", "usage", "meta", "embedding_meta"):
        assert required in tables, f"missing {required}"


def test_fts_virtual_tables_created(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'fts_%'")
    names = {r[0] for r in cur.fetchall()}
    assert "fts_chunks" in names
    assert "fts_paths" in names


def test_schema_version_recorded(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert v is not None
    assert int(v[0]) == schema.SCHEMA_VERSION


def test_apply_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    migrations.apply(conn)  # second call must not raise
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert int(v[0]) == schema.SCHEMA_VERSION


def test_wal_mode_enabled(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert mode == "wal"


def test_vec_embeddings_table_exists(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE name='vec_embeddings'")
    assert cur.fetchone() is not None


def test_vec_embeddings_round_trip(tmp_path):
    """Insert a 384-dim vector, query it back."""
    import struct
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    vec = [0.1] * 384
    blob = struct.pack(f"{len(vec)}f", *vec)
    conn.execute("INSERT INTO vec_embeddings(rowid, embedding) VALUES (1, ?)", (blob,))
    row = conn.execute(
        "SELECT rowid, distance FROM vec_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
        (blob,),
    ).fetchone()
    assert row is not None
    assert row[0] == 1
    # distance to self ~ 0
    assert row[1] < 1e-3


def test_schema_version_is_current(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    from semanticsd.db.schema import SCHEMA_VERSION
    assert int(v[0]) == SCHEMA_VERSION
