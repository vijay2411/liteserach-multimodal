"""Filename FTS search."""
from semanticsd.db import connection, migrations
from semanticsd.search.filename import search_filename


def _seed(conn, files: list[tuple[int, str]]):
    for fid, path in files:
        conn.execute(
            "INSERT INTO files(id, path, modified_at, size, file_type, indexed_at) "
            "VALUES (?, ?, 1, 1, 'text', 1)",
            (fid, path),
        )
        conn.execute("INSERT INTO fts_paths(rowid, path) VALUES (?, ?)", (fid, path))


def test_filename_match_returns_top(tmp_path):
    db = tmp_path / "x.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn, [
        (1, "/x/notes/alpha.md"),
        (2, "/x/notes/beta.md"),
        (3, "/x/code/foo.py"),
    ])

    results = search_filename(conn, "alpha", limit=5)
    assert len(results) == 1
    assert results[0].path == "/x/notes/alpha.md"
    assert results[0].mode == "filename"
    assert results[0].file_id == 1


def test_filename_no_match(tmp_path):
    db = tmp_path / "x.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn, [(1, "/x/a.md")])
    assert search_filename(conn, "zzz_nothing") == []
