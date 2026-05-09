"""POST /v1/index — manual indexing trigger."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from semanticsd.server.auth import require_token
from semanticsd.db import connection, migrations
from semanticsd import paths
from semanticsd import embedders as emb_pkg
from semanticsd.pipeline.indexer import Indexer
from semanticsd.pipeline.worker import Worker

router = APIRouter()


class IndexRequest(BaseModel):
    path: str | None = None
    source: str | None = None
    content: str | None = None
    metadata: dict[str, Any] | None = None
    drain: bool = False


@router.post("/index", dependencies=[Depends(require_token)])
def index(req: IndexRequest) -> dict[str, Any]:
    if not req.path and not (req.source and req.content is not None):
        raise HTTPException(status_code=400, detail="provide either 'path' or both 'source' and 'content'")

    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)

    idx = Indexer(conn=conn, max_file_size_mb=50)
    if req.path:
        stats = idx.index_path(Path(req.path))
    else:
        stats = idx.index_inline(source=req.source or "", content=req.content or "", metadata=req.metadata)

    drained = 0
    if req.drain:
        router_obj = emb_pkg.get_router()
        if router_obj.text is None:
            raise HTTPException(status_code=503, detail="no text embedder configured")
        worker = Worker(conn=conn, router=router_obj)
        worker.reset_stale()
        while True:
            n = worker.drain_once()
            drained += n
            if n == 0:
                break

    return {**stats, "drained": drained}
