"""Grep FTS — search chunk text via fts_chunks."""
from __future__ import annotations
import sqlite3
from semanticsd.search.types import SearchResult


def search_grep(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """Return chunk-level matches via FTS over chunk text.

    Vision chunks are excluded — their text field is just a synthetic
    "<image: ...>" descriptor, useless for grep-style search.
    """
    rows = conn.execute(
        """
        SELECT c.id, c.file_id, c.text, c.byte_start, c.byte_end, c.modality,
               f.path, f.file_type, fts_chunks.rank
        FROM fts_chunks
        JOIN chunks c ON c.id = fts_chunks.rowid
        JOIN files  f ON f.id = c.file_id
        WHERE fts_chunks MATCH ? AND c.modality = 'text'
        ORDER BY fts_chunks.rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return [
        SearchResult(
            path=str(row[6]),
            modality="text",
            mode="grep",
            score=float(-row[8]),
            chunk_id=int(row[0]),
            file_id=int(row[1]),
            snippet=str(row[2]),
            byte_start=int(row[3]),
            byte_end=int(row[4]),
            metadata={"file_type": row[7]},
        )
        for row in rows
    ]
