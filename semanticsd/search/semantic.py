"""Semantic search — vector similarity over per-modality vec0 tables."""
from __future__ import annotations
import logging
import sqlite3
import struct
from semanticsd.search.types import SearchResult

log = logging.getLogger(__name__)

# Chunks shorter than this (after stripping) aren't worth ranking — empty
# JSON files, single-line dotfiles, etc. dominate semantic results because
# their near-zero vectors are close to almost any query.
MIN_TEXT_LEN = 20


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _l2_to_cosine(l2_dist: float) -> float:
    """Convert sqlite-vec's L2 distance to cosine similarity, assuming the
    stored vectors are unit-normalized (Ollama embeddinggemma, Gemini
    Embedding 2, Qwen3-VL-Embedding all normalize their outputs).

    For unit vectors |a-b|² = 2 - 2·cos(a,b), so cos = 1 - L2²/2. Result
    is in [-1, 1] in theory; for embeddings of real-world text it's
    typically in [0.4, 1.0].
    """
    return 1.0 - (l2_dist * l2_dist) / 2.0


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
    # Over-fetch since we'll drop short / near-empty rows below.
    rows = conn.execute(
        """
        SELECT v.rowid, v.distance, c.file_id, c.text, c.byte_start, c.byte_end,
               c.modality, f.path, f.file_type
        FROM vec_text_embeddings v
        JOIN chunks c ON c.id = v.rowid
        JOIN files  f ON f.id = c.file_id
        WHERE v.embedding MATCH ? AND k = ?
          AND length(trim(c.text)) >= ?
        ORDER BY v.distance
        """,
        (blob, limit * 2, MIN_TEXT_LEN),
    ).fetchall()
    return [
        SearchResult(
            path=str(row[7]),
            modality="text",
            mode="semantic",
            score=_l2_to_cosine(float(row[1])),
            chunk_id=int(row[0]),
            file_id=int(row[2]),
            snippet=str(row[3]),
            byte_start=int(row[4]),
            byte_end=int(row[5]),
            metadata={"file_type": row[8]},
        )
        for row in rows[:limit]
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
                score=_l2_to_cosine(float(row[1])),
                chunk_id=int(row[0]),
                file_id=int(row[2]),
                snippet=str(row[3]),
                byte_start=int(row[4]),
                byte_end=int(row[5]),
                metadata={"file_type": row[7], "vec_table": table},
            ))
    out.sort(key=lambda r: -r.score)
    return out[:limit]
