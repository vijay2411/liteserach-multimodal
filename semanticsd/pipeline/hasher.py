"""Content hashing + dedup lookup against (content_hash, provider, model, dim)."""
from __future__ import annotations
import hashlib
import re
import sqlite3


_WS_RE = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    """Lowercase + collapse whitespace so trivial variants dedup."""
    return _WS_RE.sub(" ", text.strip().lower())


def sha256_hex(text: str) -> str:
    """SHA-256 of normalized text, hex-encoded."""
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex of raw bytes (e.g. image data, no normalization)."""
    return hashlib.sha256(data).hexdigest()


def find_existing_embedding(
    conn: sqlite3.Connection,
    content_hash: str,
    provider_id: str,
    model_id: str,
    dim: int,
) -> int | None:
    """Return the chunk_id of an existing embedding for this triplet+hash, or None."""
    row = conn.execute(
        "SELECT chunk_id FROM embedding_meta "
        "WHERE content_hash = ? AND provider_id = ? AND model_id = ? AND dim = ? "
        "LIMIT 1",
        (content_hash, provider_id, model_id, dim),
    ).fetchone()
    return int(row[0]) if row else None
