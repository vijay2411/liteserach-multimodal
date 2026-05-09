"""Grep FTS — search chunk text via fts_chunks."""
from __future__ import annotations
import re
import sqlite3
from semanticsd.search.types import SearchResult

MIN_TEXT_LEN = 3  # mirror semantic.py — drop trivial chunks (empty, `{}`, etc.)


# FTS5 reserved/special characters that need quoting to be safe in a query.
# Letters, digits, underscore are safe; anything else (incl. CJK punctuation,
# hyphens, apostrophes) is quoted as a phrase to avoid syntax errors.
_SAFE_TOKEN_RE = re.compile(r"^[\w]+$", re.UNICODE)

# Minimal English stop-word list. Without this, queries like "spooky action at
# a distance" or "orange cat with whiskers" surface every file containing "at"
# or "with" — which is most of them. We keep this short and focused on the
# words that genuinely add no signal; semantic mode handles paraphrasing
# anyway, so missing edge-case stops here is fine.
_STOPWORDS = frozenset({
    "the", "and", "but", "for", "nor", "yet", "with", "without", "from",
    "into", "onto", "upon", "over", "under", "than", "that", "this",
    "these", "those", "are", "was", "were", "been", "being", "have",
    "has", "had", "does", "did", "doing", "will", "would", "should",
    "could", "may", "might", "can", "you", "your", "yours", "our",
    "ours", "they", "them", "their", "theirs", "his", "her", "hers",
    "him", "she", "all", "some", "any", "each", "few", "more", "most",
    "other", "such", "very",
})


def _build_fts_query(raw: str) -> str:
    """Rewrite a free-form query as an FTS5 OR-of-tokens search.

    SQLite FTS5 defaults to AND between bare tokens, which means
    "password hashing function" requires all three words in one chunk —
    too strict for natural language queries. We OR them so partial
    matches still contribute to BM25 rank.

    Bare safe tokens go through unquoted so the porter stemmer applies
    (lets "fox" match "foxes", "transactions" match "transaction").
    Tokens with hyphens/apostrophes/CJK punctuation get quoted as phrases
    to avoid FTS5 syntax errors. Tokens shorter than 2 chars are dropped.
    """
    raw = (raw or "").strip()
    if not raw:
        return raw
    tokens = re.findall(r"[\w'-]+", raw, flags=re.UNICODE)
    # Drop very short tokens (FTS5 ignores 1-char terms anyway) and English
    # stop words that contribute noise without signal.
    tokens = [
        t for t in tokens
        if len(t) >= 3 and t.lower() not in _STOPWORDS
    ]
    if not tokens:
        return f'"{raw}"'
    parts: list[str] = []
    for t in tokens:
        if _SAFE_TOKEN_RE.match(t):
            parts.append(t)  # bare → stemmer applies
        else:
            parts.append(f'"{t}"')  # quoted phrase, no special-char surprises
    return " OR ".join(parts)


def search_grep(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """Return chunk-level matches via FTS over chunk text.

    Vision chunks are excluded — their text field is just a synthetic
    "<image: ...>" descriptor, useless for grep-style search. Trivially
    short chunks (e.g. `{}`) are also dropped. The query is rewritten as
    OR-of-tokens so partial matches contribute to ranking.
    """
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []
    try:
        rows = conn.execute(
            """
            SELECT c.id, c.file_id, c.text, c.byte_start, c.byte_end, c.modality,
                   f.path, f.file_type, fts_chunks.rank
            FROM fts_chunks
            JOIN chunks c ON c.id = fts_chunks.rowid
            JOIN files  f ON f.id = c.file_id
            WHERE fts_chunks MATCH ?
              AND c.modality = 'text'
              AND length(trim(c.text)) >= ?
            ORDER BY fts_chunks.rank
            LIMIT ?
            """,
            (fts_query, MIN_TEXT_LEN, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # FTS5 may reject malformed queries — fall back to empty result rather than crash.
        return []
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
