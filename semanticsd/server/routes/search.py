"""GET /v1/search — semantic + filename + grep + hybrid."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query
from semanticsd.server.auth import require_token
from semanticsd.db import connection
from semanticsd import paths
from semanticsd import embedders as emb_pkg
from semanticsd.search.engine import Engine
from semanticsd.search.types import SearchOptions

router = APIRouter()


@router.get("/search", dependencies=[Depends(require_token)])
def search(
    q: str = Query(..., min_length=1, description="Query string"),
    mode: str = Query("hybrid", description="semantic | filename | grep | hybrid"),
    limit: int = Query(20, ge=1, le=200),
    cwd: str | None = Query(None, description="Restrict to files under this dir"),
    all: bool = Query(False, description="Disable CWD filter"),
    vision: bool = Query(True, description="Include cross-modal vision results"),
    collapse: bool = Query(True, description="Collapse to one result per file"),
) -> dict[str, Any]:
    if mode not in ("semantic", "filename", "grep", "hybrid"):
        raise HTTPException(status_code=400, detail=f"unknown mode: {mode}")

    conn = connection.get_connection(paths.db_path())
    router_obj = emb_pkg.get_router()
    engine = Engine(conn, router_obj)

    opts = SearchOptions(
        mode=mode,  # type: ignore[arg-type]
        limit=limit,
        cwd=Path(cwd) if cwd else None,
        all=all,
        vision=vision,
        collapse=collapse,
    )
    t0 = time.time()
    results = engine.search(q, opts)
    return {
        "results": [r.model_dump() for r in results],
        "took_ms": int((time.time() - t0) * 1000),
    }
