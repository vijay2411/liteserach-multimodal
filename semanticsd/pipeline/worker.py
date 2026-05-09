"""Job-queue worker."""
from __future__ import annotations
import asyncio
import logging
import sqlite3
import struct
import time
from semanticsd.embedders.base import Embedder
from semanticsd.pipeline.hasher import find_existing_embedding

log = logging.getLogger(__name__)


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class Worker:
    def __init__(self, conn: sqlite3.Connection, embedder: Embedder, batch_size: int = 128, max_attempts: int = 5):
        self.conn = conn
        self.embedder = embedder
        self.batch_size = batch_size
        self.max_attempts = max_attempts

    def reset_stale(self) -> None:
        self.conn.execute("UPDATE jobs SET status='pending' WHERE status='in_flight'")

    def drain_once(self) -> int:
        rows = self.conn.execute(
            "SELECT j.id, j.chunk_id, c.text, c.content_hash "
            "FROM jobs j JOIN chunks c ON c.id = j.chunk_id "
            "WHERE j.status = 'pending' "
            "ORDER BY j.id LIMIT ?",
            (self.batch_size,),
        ).fetchall()
        if not rows:
            return 0

        job_ids = [int(r[0]) for r in rows]
        placeholders = ",".join("?" for _ in job_ids)
        self.conn.execute(
            f"UPDATE jobs SET status='in_flight', updated_at=? WHERE id IN ({placeholders})",
            [int(time.time()), *job_ids],
        )

        to_embed: list[tuple[int, int, str, str]] = []
        cached: list[tuple[int, int, int]] = []
        for jid, cid, text, chash in rows:
            existing_chunk_id = find_existing_embedding(
                self.conn,
                content_hash=chash,
                provider_id=self.embedder.provider_id,
                model_id=self.embedder.model_id,
                dim=self.embedder.dim,
            )
            if existing_chunk_id is not None and int(existing_chunk_id) != int(cid):
                cached.append((int(jid), int(cid), int(existing_chunk_id)))
            else:
                to_embed.append((int(jid), int(cid), text, chash))

        if to_embed:
            try:
                texts = [t[2] for t in to_embed]
                result = self.embedder.embed(texts, kind="doc")
            except Exception as e:
                log.warning("embedder failed: %s", e)
                self._mark_failed(job_ids, str(e))
                return 0
            for (jid, cid, _t, chash), vec in zip(to_embed, result.vectors):
                self.conn.execute(
                    "INSERT OR REPLACE INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                    (cid, _vec_to_blob(list(vec))),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO embedding_meta(chunk_id, provider_id, model_id, dim, content_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (cid, self.embedder.provider_id, self.embedder.model_id, self.embedder.dim, chash),
                )

        for jid, target_cid, source_cid in cached:
            row = self.conn.execute(
                "SELECT embedding FROM vec_embeddings WHERE rowid = ?", (source_cid,)
            ).fetchone()
            if row is None:
                continue
            self.conn.execute(
                "INSERT OR REPLACE INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                (target_cid, row[0]),
            )
            row2 = self.conn.execute(
                "SELECT content_hash FROM embedding_meta WHERE chunk_id = ?", (source_cid,)
            ).fetchone()
            chash = row2[0] if row2 else ""
            self.conn.execute(
                "INSERT OR REPLACE INTO embedding_meta(chunk_id, provider_id, model_id, dim, content_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (target_cid, self.embedder.provider_id, self.embedder.model_id, self.embedder.dim, chash),
            )

        self.conn.execute(
            f"UPDATE jobs SET status='done', updated_at=? WHERE id IN ({placeholders})",
            [int(time.time()), *job_ids],
        )
        return len(job_ids)

    def _mark_failed(self, job_ids: list[int], error: str) -> None:
        placeholders = ",".join("?" for _ in job_ids)
        self.conn.execute(
            f"UPDATE jobs SET status='pending', attempts = attempts + 1, "
            f"last_error = ?, updated_at = ? WHERE id IN ({placeholders})",
            [error, int(time.time()), *job_ids],
        )
        self.conn.execute(
            "UPDATE jobs SET status='failed' WHERE attempts >= ?",
            (self.max_attempts,),
        )

    async def run_forever(self, poll_interval_s: float = 2.0) -> None:
        self.reset_stale()
        while True:
            try:
                processed = self.drain_once()
            except Exception as e:
                log.error("worker drain crashed: %s", e)
                processed = 0
            if processed == 0:
                await asyncio.sleep(poll_interval_s)
