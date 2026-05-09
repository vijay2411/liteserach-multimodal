"""Health endpoint."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from semanticsd import __version__
from semanticsd.server.auth import require_token
from semanticsd.db import connection
from semanticsd import paths

router = APIRouter()


@router.get("/health", dependencies=[Depends(require_token)])
def health() -> dict:
    db_ok = True
    doc_count = 0
    try:
        conn = connection.get_connection(paths.db_path())
        row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
        doc_count = int(row[0]) if row else 0
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "version": __version__,
        "doc_count": doc_count,
        "vector_store": {"ok": db_ok},
        "embedder": {"ok": True, "message": "not configured (Plan 2)"},
    }
