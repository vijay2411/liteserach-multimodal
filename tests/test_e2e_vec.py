"""End-to-end: real LocalEmbedder + sqlite-vec round-trip.

Downloads BAAI/bge-small-en-v1.5 on first run (~30MB).
Marked slow so a developer can skip via `pytest -m "not slow"`.
"""
import struct
import pytest
from semanticsd.db import connection, migrations
from semanticsd.embedders.local import LocalEmbedder


pytestmark = pytest.mark.slow


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def test_local_embedder_round_trip_through_vec_embeddings(tmp_path):
    """Embed real text, store it via sqlite-vec, query nearest, retrieve self."""
    db = tmp_path / "vec.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    embedder = LocalEmbedder()
    docs = [
        "The alpha protocol authenticates two parties via a shared secret.",
        "Pasta carbonara is a Roman dish with egg and pancetta.",
        "The beta release improved indexing throughput via batching.",
    ]
    out = embedder.embed(docs, kind="doc")
    assert len(out.vectors) == 3
    assert all(len(v) == 384 for v in out.vectors)

    for i, v in enumerate(out.vectors, start=1):
        conn.execute(
            "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
            (i, _vec_to_blob(v)),
        )

    query_vec = embedder.embed(["how does authentication work"], kind="query").vectors[0]
    rows = conn.execute(
        "SELECT rowid, distance FROM vec_embeddings WHERE embedding MATCH ? "
        "ORDER BY distance LIMIT 3",
        (_vec_to_blob(query_vec),),
    ).fetchall()
    assert len(rows) == 3
    # Nearest neighbour must be doc #1 (the auth one).
    nearest_id = rows[0][0]
    assert nearest_id == 1, f"expected nearest=1 (auth doc), got {nearest_id}"
