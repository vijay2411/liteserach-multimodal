"""Queue re-embedding jobs for chunks lacking the current embedder's vectors.

When the user switches `[embedding.text].preset` (or model/provider for
vision), existing chunks still carry their previous embeddings. The
per-(modality, dim) vec tables introduced in Plan 4.5 mean those old
vectors stay queryable; this module just queues fresh jobs for chunks
that don't yet have a vector from the current router.
"""
from __future__ import annotations
import logging
import sqlite3
import time
from typing import Literal
from semanticsd.embedders.router import EmbedderRouter

log = logging.getLogger(__name__)

Modality = Literal["text", "vision", "all"]


def _queue_for_modality(
    conn: sqlite3.Connection,
    modality: str,
    embedder,
) -> int:
    """Find every chunk of `modality` that does NOT have an embedding from
    (embedder.provider_id, embedder.model_id, embedder.dim) and queue a
    pending job for it. Returns the number of jobs queued."""
    rows = conn.execute(
        """
        SELECT c.id FROM chunks c
        WHERE c.modality = ?
          AND NOT EXISTS (
            SELECT 1 FROM embedding_meta em
            WHERE em.chunk_id = c.id
              AND em.provider_id = ?
              AND em.model_id = ?
              AND em.dim = ?
          )
        """,
        (modality, embedder.provider_id, embedder.model_id, embedder.dim),
    ).fetchall()
    if not rows:
        return 0
    now = int(time.time())
    conn.executemany(
        "INSERT INTO jobs(chunk_id, status, attempts, created_at, updated_at) "
        "VALUES (?, 'pending', 0, ?, ?)",
        [(int(r[0]), now, now) for r in rows],
    )
    return len(rows)


def queue_reembed(
    conn: sqlite3.Connection,
    router: EmbedderRouter,
    modality: Modality = "all",
) -> dict[str, int]:
    """Queue jobs for every chunk lacking the current embedder's embedding.

    Returns: {"text": N, "vision": M} — counts of jobs queued per modality.
    Modalities with no configured embedder are skipped.
    """
    out = {"text": 0, "vision": 0}
    if modality in ("text", "all") and router.text is not None:
        out["text"] = _queue_for_modality(conn, "text", router.text)
        log.info("queued %d text re-embed jobs (provider=%s model=%s dim=%s)",
                 out["text"], router.text.provider_id, router.text.model_id,
                 router.text.dim)
    if modality in ("vision", "all") and router.vision is not None:
        out["vision"] = _queue_for_modality(conn, "vision", router.vision)
        log.info("queued %d vision re-embed jobs (provider=%s model=%s dim=%s)",
                 out["vision"], router.vision.provider_id, router.vision.model_id,
                 router.vision.dim)
    return out
