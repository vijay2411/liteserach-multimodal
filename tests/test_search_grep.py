"""Grep FTS search."""
from semanticsd.db import connection, migrations
from semanticsd.search.grep import search_grep


def _seed_chunk(conn, chunk_id, file_id, text, modality="text"):
    # Use chunk_id as chunk_index to keep the unique constraint satisfied.
    conn.execute(
        "INSERT INTO chunks(id, file_id, chunk_index, text, content_hash, "
        "byte_start, byte_end, modality) VALUES (?, ?, ?, ?, 'h' || ?, 0, ?, ?)",
        (chunk_id, file_id, chunk_id, text, chunk_id, len(text), modality),
    )
    if modality == "text":
        conn.execute(
            "INSERT INTO fts_chunks(rowid, text) VALUES (?, ?)",
            (chunk_id, text),
        )


def _seed_file(conn, file_id, path):
    conn.execute(
        "INSERT INTO files(id, path, modified_at, size, file_type, indexed_at) "
        "VALUES (?, ?, 1, 1, 'text', 1)",
        (file_id, path),
    )


def test_grep_finds_text_chunks(tmp_path):
    db = tmp_path / "g.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed_file(conn, 1, "/x/notes.md")
    _seed_chunk(conn, 1, 1, "the quick brown fox jumped")
    _seed_chunk(conn, 2, 1, "the lazy dog sleeps quietly")

    results = search_grep(conn, "fox", limit=5)
    assert len(results) == 1
    assert results[0].chunk_id == 1
    assert "fox" in results[0].snippet
    assert results[0].mode == "grep"


def test_grep_excludes_vision_chunks(tmp_path):
    db = tmp_path / "g.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed_file(conn, 1, "/x/img.png")
    # Vision chunk: exists in chunks but not in fts_chunks (indexer skipped it).
    _seed_chunk(conn, 1, 1, "<image: foo.png>", modality="vision")
    # ... but if some bug ever inserted it, the modality filter still excludes:
    conn.execute(
        "INSERT INTO fts_chunks(rowid, text) VALUES (1, '<image: foo.png>')"
    )

    results = search_grep(conn, "image", limit=5)
    # Even though FTS would match, grep filters modality='text'.
    assert results == []
