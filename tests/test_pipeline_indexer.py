from pathlib import Path
import time
from semanticsd.db import connection, migrations
from semanticsd.pipeline.indexer import Indexer
from tests._fixtures import make_text, make_markdown


def _fresh_db(tmp_path: Path):
    db = tmp_path / "x.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    return conn


def test_index_path_creates_files_and_chunks_and_jobs(tmp_path):
    conn = _fresh_db(tmp_path)
    src = tmp_path / "corpus"
    src.mkdir()
    make_text(src, body="Hello world. Second sentence.")

    idx = Indexer(conn=conn, max_file_size_mb=50)
    stats = idx.index_path(src)
    assert stats["files_indexed"] == 1
    assert stats["chunks_created"] >= 1
    assert stats["jobs_queued"] == stats["chunks_created"]

    rows = conn.execute("SELECT count(*) FROM files").fetchone()
    assert rows[0] == 1
    rows = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert rows[0] >= 1
    rows = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()
    assert rows[0] >= 1


def test_index_path_skips_unsupported(tmp_path):
    conn = _fresh_db(tmp_path)
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "binary.dat").write_bytes(b"\x00\x01" * 10)
    make_markdown(src)

    idx = Indexer(conn=conn, max_file_size_mb=50)
    stats = idx.index_path(src)
    assert stats["files_indexed"] == 1
    assert stats["files_skipped_unsupported"] == 1


def test_re_index_same_content_does_not_duplicate(tmp_path):
    conn = _fresh_db(tmp_path)
    src = tmp_path / "corpus"
    src.mkdir()
    f = make_text(src, body="Stable content.")

    idx = Indexer(conn=conn, max_file_size_mb=50)
    s1 = idx.index_path(src)
    s2 = idx.index_path(src)

    files_count = conn.execute("SELECT count(*) FROM files").fetchone()[0]
    assert files_count == 1
    assert s2["jobs_queued"] == 0


def test_index_inline_creates_synthetic_path(tmp_path):
    conn = _fresh_db(tmp_path)
    idx = Indexer(conn=conn, max_file_size_mb=50)
    stats = idx.index_inline(source="conversation://1", content="hello inline", metadata={"role": "user"})
    assert stats["chunks_created"] >= 1
    assert stats["jobs_queued"] >= 1
    row = conn.execute("SELECT path, file_type FROM files").fetchone()
    assert row[0] == "conversation://1"
    assert row[1] == "inline"


def test_indexer_persists_vision_chunk_with_blob(tmp_app_support, tmp_path):
    from semanticsd.db import connection, migrations
    from semanticsd.pipeline.indexer import Indexer
    from semanticsd import paths
    from tests._fixtures import make_image_with_text

    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)

    img_path = make_image_with_text(tmp_path, text="hi")
    indexer = Indexer(conn)
    stats = indexer.index_path(img_path)

    assert stats["files_indexed"] == 1
    rows = conn.execute(
        "SELECT modality, image_blob, content_hash FROM chunks WHERE modality='vision'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "vision"
    assert rows[0][1] is not None
    assert rows[0][1][:8].startswith(b"\x89PNG")
    assert len(rows[0][2]) == 64  # sha256 hex
