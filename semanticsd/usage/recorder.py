"""Write usage rows to the DB.

One row per embedder call (batched). The cost is computed up-front
from the embedder's per-million-token rate, so reports don't need to
look at provider class.
"""
from __future__ import annotations
import sqlite3
import time
from dataclasses import dataclass


@dataclass
class UsageEvent:
    provider_id: str
    model_id: str
    operation: str            # "text_embed" | "vision_embed" | "query_embed"
    input_tokens: int
    chunk_count: int
    duration_ms: int
    cost_usd: float


def compute_cost(input_tokens: int, cost_per_million_usd: float) -> float:
    """Cost = tokens / 1e6 * rate. Rate of 0 means free (local provider)."""
    if cost_per_million_usd <= 0.0:
        return 0.0
    return (input_tokens / 1_000_000.0) * cost_per_million_usd


def record_usage(conn: sqlite3.Connection, event: UsageEvent) -> None:
    """Insert one row into the `usage` table. Caller commits separately
    (worker keeps the same transaction as the chunk/job updates)."""
    conn.execute(
        "INSERT INTO usage(timestamp, provider_id, model_id, operation, "
        "input_tokens, cost_usd, chunk_count, duration_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            int(time.time()),
            event.provider_id,
            event.model_id,
            event.operation,
            int(event.input_tokens),
            float(event.cost_usd),
            int(event.chunk_count),
            int(event.duration_ms),
        ),
    )
