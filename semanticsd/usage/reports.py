"""Read-side queries: aggregates for the `ssearch usage` CLI and HTTP."""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from semanticsd.usage.budget import month_start_unix


@dataclass
class UsageRow:
    provider_id: str
    model_id: str
    operation: str
    calls: int
    chunks: int
    input_tokens: int
    cost_usd: float
    duration_ms: int


@dataclass
class UsageTotals:
    since_unix: int
    until_unix: int | None
    calls: int
    chunks: int
    input_tokens: int
    cost_usd: float
    by_provider: list[UsageRow]


def aggregate_by_provider(
    conn: sqlite3.Connection,
    since_unix: int | None = None,
    until_unix: int | None = None,
    provider: str | None = None,
) -> list[UsageRow]:
    where = ["1=1"]
    params: list = []
    if since_unix is not None:
        where.append("timestamp >= ?")
        params.append(int(since_unix))
    if until_unix is not None:
        where.append("timestamp <= ?")
        params.append(int(until_unix))
    if provider:
        where.append("provider_id = ?")
        params.append(provider)
    sql = (
        "SELECT provider_id, model_id, operation, COUNT(*) AS calls, "
        "SUM(chunk_count) AS chunks, SUM(input_tokens) AS tokens, "
        "SUM(cost_usd) AS cost, SUM(duration_ms) AS dur "
        f"FROM usage WHERE {' AND '.join(where)} "
        "GROUP BY provider_id, model_id, operation "
        "ORDER BY cost DESC, calls DESC"
    )
    rows = conn.execute(sql, params).fetchall()
    return [
        UsageRow(
            provider_id=r[0], model_id=r[1], operation=r[2],
            calls=int(r[3] or 0), chunks=int(r[4] or 0),
            input_tokens=int(r[5] or 0), cost_usd=float(r[6] or 0.0),
            duration_ms=int(r[7] or 0),
        )
        for r in rows
    ]


def totals(
    conn: sqlite3.Connection,
    since_unix: int | None = None,
    until_unix: int | None = None,
    provider: str | None = None,
) -> UsageTotals:
    """High-level summary used by the health endpoint and `ssearch usage`."""
    if since_unix is None:
        since_unix = month_start_unix()
    rows = aggregate_by_provider(conn, since_unix, until_unix, provider)
    return UsageTotals(
        since_unix=since_unix,
        until_unix=until_unix,
        calls=sum(r.calls for r in rows),
        chunks=sum(r.chunks for r in rows),
        input_tokens=sum(r.input_tokens for r in rows),
        cost_usd=round(sum(r.cost_usd for r in rows), 6),
        by_provider=rows,
    )
