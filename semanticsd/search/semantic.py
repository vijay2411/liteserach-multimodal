"""Semantic search — vector similarity over per-modality vec0 tables."""
from __future__ import annotations
import logging
import sqlite3
import struct
from semanticsd.search.types import SearchResult

log = logging.getLogger(__name__)


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _vision_vec_tables_for_dim(conn: sqlite3.Connection, dim: int) -> list[str]:
    """Return all vec_vision_* tables matching `dim`.

    Default canonical name is vec_vision_embeddings (3072-d). Other dims live in
    vec_vision_embeddings_<dim>. We pick whichever ones match the active vision
    embedder's dim — non-matching tables are stale data from another provider.
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_vision_embeddings%'"
    ).fetchall()
    candidates = [r[0] for r in rows if not any(
        r[0].endswith(suffix)
        for suffix in ("_info", "_chunks", "_rowids", "_vector_chunks00")
    )]
    if dim == 3072:
        return [t for t in candidates if t == "vec_vision_embeddings"]
    suffix = f"_{dim}"
    return [t for t in candidates if t.endswith(suffix)]


def search_semantic_text(
    conn: sqlite3.Connection,
    query_vec: list[float],
    limit: int = 20,
) -> list[SearchResult]:
    """Top-K nearest text chunks by cosine distance."""
    if not query_vec:
        return []
    blob = _vec_to_blob(query_vec)
    rows = conn.execute(
        """
        SELECT v.rowid, v.distance, c.file_id, c.text, c.byte_start, c.byte_end,
               c.modality, f.path, f.file_type
        FROM vec_text_embeddings v
        JOIN chunks c ON c.id = v.rowid
        JOIN files  f ON f.id = c.file_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (blob, limit),
    ).fetchall()
    return [
        SearchResult(
            path=str(row[7]),
            modality="text",
            mode="semantic",
            score=1.0 - float(row[1]),  # convert L2 distance to similarity-ish
            chunk_id=int(row[0]),
            file_id=int(row[2]),
            snippet=str(row[3]),
            byte_start=int(row[4]),
            byte_end=int(row[5]),
            metadata={"file_type": row[8]},
        )
        for row in rows
    ]


def search_semantic_vision(
    conn: sqlite3.Connection,
    query_vec: list[float],
    dim: int,
    limit: int = 20,
) -> list[SearchResult]:
    """Top-K nearest vision chunks across all vec_vision_* tables matching dim.

    With cross-modal embedders (Gemini Embedding 2, Qwen3-VL), passing a TEXT
    query embedded by the vision embedder lets you find images that match
    the description.
    """
    if not query_vec:
        return []
    tables = _vision_vec_tables_for_dim(conn, dim)
    if not tables:
        return []
    blob = _vec_to_blob(query_vec)
    out: list[SearchResult] = []
    for table in tables:
        try:
            rows = conn.execute(
                f"""
                SELECT v.rowid, v.distance, c.file_id, c.text, c.byte_start, c.byte_end,
                       f.path, f.file_type
                FROM {table} v
                JOIN chunks c ON c.id = v.rowid
                JOIN files  f ON f.id = c.file_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (blob, limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            log.warning("vision search on %s failed: %s", table, e)
            continue
        for row in rows:
            out.append(SearchResult(
                path=str(row[6]),
                modality="vision",
                mode="semantic",
                score=1.0 - float(row[1]),
                chunk_id=int(row[0]),
                file_id=int(row[2]),
                snippet=str(row[3]),
                byte_start=int(row[4]),
                byte_end=int(row[5]),
                metadata={"file_type": row[7], "vec_table": table},
            ))
    out.sort(key=lambda r: -r.score)
    return out[:limit]
