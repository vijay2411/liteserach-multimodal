"""Filename FTS — search file paths via fts_paths."""
from __future__ import annotations
import sqlite3
from semanticsd.search.types import SearchResult


def search_filename(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """Return file-level matches by FTS over the path column.

    BM25 rank ascending == better match in FTS5; we invert via -rank so
    score is "higher = better".
    """
    rows = conn.execute(
        """
        SELECT f.id, f.path, f.file_type, fts_paths.rank
        FROM fts_paths
        JOIN files f ON f.id = fts_paths.rowid
        WHERE fts_paths MATCH ?
        ORDER BY fts_paths.rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
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
