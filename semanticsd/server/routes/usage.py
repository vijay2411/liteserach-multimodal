"""GET /v1/usage — cost & volume report."""
from __future__ import annotations
from datetime import datetime, timezone
import calendar
from fastapi import APIRouter, Depends, HTTPException, Query
from semanticsd.server.auth import require_token
from semanticsd.db import connection
from semanticsd import paths
from semanticsd.usage.reports import totals
from semanticsd.usage.budget import month_start_unix
from dataclasses import asdict

router = APIRouter()


def _parse_iso_date(s: str) -> int:
    """YYYY-MM-DD → unix timestamp at 00:00:00 UTC."""
    try:
        d = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"bad date {s!r}: {e}")
    return calendar.timegm(d.timetuple())


@router.get("/usage", dependencies=[Depends(require_token)])
def get_usage(
    since: str | None = Query(None, description="YYYY-MM-DD; default = start of month"),
    until: str | None = Query(None, description="YYYY-MM-DD; default = now"),
    provider: str | None = Query(None, description="provider_id filter, e.g. 'gemini'"),
):
    conn = connection.get_connection(paths.db_path())
    since_unix = _parse_iso_date(since) if since else month_start_unix()
    until_unix = _parse_iso_date(until) if until else None

    t = totals(conn, since_unix=since_unix, until_unix=until_unix, provider=provider)
    return {
        "since_unix": t.since_unix,
        "until_unix": t.until_unix,
        "calls": t.calls,
        "chunks": t.chunks,
        "input_tokens": t.input_tokens,
        "cost_usd": t.cost_usd,
        "by_provider": [asdict(r) for r in t.by_provider],
    }
