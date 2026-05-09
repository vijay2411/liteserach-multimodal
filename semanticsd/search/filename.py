"""Filename FTS — search file paths via fts_paths."""
from __future__ import annotations
import sqlite3
from semanticsd.search.grep import _build_fts_query
from semanticsd.search.types import SearchResult


def search_filename(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """Return file-level matches by FTS over the path column.

    BM25 rank ascending == better match in FTS5; we invert via -rank so
    score is "higher = better". Queries are OR-rewritten so multi-word
    queries match files containing any of the tokens.
    """
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []
    try:
        rows = conn.execute(
            """
            SELECT f.id, f.path, f.file_type, fts_paths.rank
            FROM fts_paths
            JOIN files f ON f.id = fts_paths.rowid
            WHERE fts_paths MATCH ?
            ORDER BY fts_paths.rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        SearchResult(
            path=str(row[1]),
            modality="text",
            mode="filename",
            score=float(-row[3]),  # invert BM25 so higher == better
            file_id=int(row[0]),
            chunk_id=None,
            metadata={"file_type": row[2]},
        )
        for row in rows
    ]
