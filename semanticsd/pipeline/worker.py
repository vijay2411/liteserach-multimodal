"""Job-queue worker — modality-aware routing via EmbedderRouter."""
from __future__ import annotations
import asyncio
import logging
import sqlite3
import struct
import time
from collections import defaultdict
from semanticsd.embedders.router import EmbedderRouter
from semanticsd.pipeline.hasher import find_existing_embedding
from semanticsd.usage.recorder import UsageEvent, record_usage, compute_cost
from semanticsd.usage.budget import BudgetGate

log = logging.getLogger(__name__)

def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _vec_table_for(modality: str, dim: int) -> str:
    """Vec table name keyed by (modality, dim). Multiple providers per modality
    coexist in their own dim-keyed tables; switching provider is non-destructive
    if the new dim differs."""
    if modality == "text":
        # 768-d default text providers (ollama embeddinggemma, nomic) share the
        # canonical table; non-default dims get their own table.
        return "vec_text_embeddings" if dim == 768 else f"vec_text_embeddings_{dim}"
    if modality == "vision":
        return "vec_vision_embeddings" if dim == 3072 else f"vec_vision_embeddings_{dim}"
    raise ValueError(f"unknown modality: {modality!r}")


def _ensure_vec_table(conn, table: str, dim: int) -> None:
    """Idempotent CREATE for a per-(modality, dim) vec0 table."""
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(embedding FLOAT[{dim}])"
    )


class Worker:
    def __init__(
        self,
        conn: sqlite3.Connection,
        router: EmbedderRouter,
        batch_size: int = 128,
        max_attempts: int = 5,
        budget_gate: BudgetGate | None = None,
    ):
        self.conn = conn
        self.router = router
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        # When None, no budget enforcement (test/legacy path). The daemon
        # passes a real gate built from cfg.budget.
        self.budget_gate = budget_gate

    def reset_stale(self) -> None:
        self.conn.execute("UPDATE jobs SET status='pending' WHERE status='in_flight'")

    def drain_once(self) -> int:
        rows = self.conn.execute(
            "SELECT j.id, j.chunk_id, c.text, c.content_hash, c.modality, c.image_blob "
            "FROM jobs j JOIN chunks c ON c.id = j.chunk_id "
            "WHERE j.status='pending' "
            "ORDER BY j.id LIMIT ?",
            (self.batch_size,),
        ).fetchall()
        if not rows:
            return 0

        all_job_ids = [int(r[0]) for r in rows]
        ph_all = ",".join("?" for _ in all_job_ids)
        self.conn.execute(
            f"UPDATE jobs SET status='in_flight', updated_at=? WHERE id IN ({ph_all})",
            [int(time.time()), *all_job_ids],
        )

        groups: dict[str, list] = defaultdict(list)
        for r in rows:
            modality = r[4] or "text"
            groups[modality].append(r)

        processed = 0
        for modality, group in groups.items():
            embedder = self.router.get(modality)
            if embedder is None:
                ids = [int(r[0]) for r in group]
                log.warning(
                    "no embedder configured for modality=%s; %d jobs skipped",
                    modality, len(ids),
                )
                self._mark_failed(ids, f"no_embedder_for_modality:{modality}")
                continue
            try:
                processed += self._process_group(modality, embedder, group)
            except Exception as e:
                log.warning("group %s failed: %s", modality, e)
                self._mark_failed([int(r[0]) for r in group], str(e))

        return processed

    def _process_group(self, modality: str, embedder, group) -> int:
        vec_table = _vec_table_for(modality, embedder.dim)
        _ensure_vec_table(self.conn, vec_table, embedder.dim)
        to_embed: list[tuple[int, int, str, str, bytes | None]] = []
        cached: list[tuple[int, int, int]] = []

        for jid, cid, text, chash, _m, blob in group:
            existing_cid = find_existing_embedding(
                self.conn,
                content_hash=chash,
                provider_id=embedder.provider_id,
                model_id=embedder.model_id,
                dim=embedder.dim,
            )
            if existing_cid is not None and int(existing_cid) != int(cid):
                cached.append((int(jid), int(cid), int(existing_cid)))
            else:
                to_embed.append((int(jid), int(cid), text, chash, blob))

        if to_embed:
            # Estimate this batch's cost up-front for the budget gate.
            rate = float(getattr(embedder, "cost_per_million_input_tokens_usd",
                                 getattr(embedder, "cost_per_million_image_tokens_usd", 0.0)))
            if self.budget_gate is not None and rate > 0.0:
                # Use the embedder's token estimator so we don't double-count.
                if modality == "text":
                    est_tokens = embedder.estimate_tokens([t[2] for t in to_embed])
                else:
                    est_tokens = embedder.estimate_image_tokens([t[4] for t in to_embed])
                est_cost = compute_cost(int(est_tokens), rate)
                if not self.budget_gate.can_spend(est_cost):
                    job_ids = [int(t[0]) for t in to_embed]
                    log.warning(
                        "budget exceeded — refusing %d %s jobs (est $%.4f)",
                        len(job_ids), modality, est_cost,
                    )
                    self._mark_failed(job_ids, "budget_exceeded")
                    return 0
            t0 = time.monotonic()
            if modality == "text":
                inputs = [t[2] for t in to_embed]
                result = embedder.embed(inputs, kind="doc")
            else:  # vision
                inputs = [t[4] for t in to_embed]
                result = embedder.embed_images(inputs, kind="doc")
            duration_ms = int((time.monotonic() - t0) * 1000)

            actual_cost = compute_cost(int(result.input_tokens), rate)
            try:
                record_usage(self.conn, UsageEvent(
                    provider_id=embedder.provider_id,
                    model_id=embedder.model_id,
                    operation=f"{modality}_embed",
                    input_tokens=int(result.input_tokens),
                    chunk_count=len(to_embed),
                    duration_ms=duration_ms,
                    cost_usd=actual_cost,
                ))
            except Exception as e:
                log.warning("failed to record usage row: %s", e)

            for (jid, cid, _t, chash, _b), vec in zip(to_embed, result.vectors):
                self.conn.execute(
                    f"INSERT OR REPLACE INTO {vec_table}(rowid, embedding) VALUES (?, ?)",
                    (cid, _vec_to_blob(list(vec))),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO embedding_meta"
                    "(chunk_id, provider_id, model_id, dim, content_hash, modality) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (cid, embedder.provider_id, embedder.model_id,
                     embedder.dim, chash, modality),
                )

        for jid, target_cid, source_cid in cached:
            row = self.conn.execute(
                f"SELECT embedding FROM {vec_table} WHERE rowid=?", (source_cid,)
            ).fetchone()
            if row is None:
                continue
            self.conn.execute(
                f"INSERT OR REPLACE INTO {vec_table}(rowid, embedding) VALUES (?, ?)",
                (target_cid, row[0]),
            )
            row2 = self.conn.execute(
                "SELECT content_hash FROM embedding_meta WHERE chunk_id=?",
                (source_cid,),
            ).fetchone()
            chash = row2[0] if row2 else ""
            self.conn.execute(
                "INSERT OR REPLACE INTO embedding_meta"
                "(chunk_id, provider_id, model_id, dim, content_hash, modality) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (target_cid, embedder.provider_id, embedder.model_id,
                 embedder.dim, chash, modality),
            )

        ids = [int(t[0]) for t in to_embed] + [int(c[0]) for c in cached]
        if ids:
            ph = ",".join("?" for _ in ids)
            self.conn.execute(
                f"UPDATE jobs SET status='done', updated_at=? WHERE id IN ({ph})",
                [int(time.time()), *ids],
            )
        return len(ids)

    def _mark_failed(self, job_ids: list[int], error: str) -> None:
        if not job_ids:
            return
        ph = ",".join("?" for _ in job_ids)
        self.conn.execute(
            f"UPDATE jobs SET status='pending', attempts=attempts+1, "
            f"last_error=?, updated_at=? WHERE id IN ({ph})",
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
