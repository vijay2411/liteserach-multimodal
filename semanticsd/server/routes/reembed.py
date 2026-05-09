"""POST /v1/reembed — queue re-embed jobs for chunks lacking current vectors."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from semanticsd.server.auth import require_token
from semanticsd.db import connection
from semanticsd import paths
from semanticsd import embedders as emb_pkg
from semanticsd.reembed import queue_reembed

router = APIRouter()


class ReembedRequest(BaseModel):
    modality: str = "all"   # text | vision | all


@router.post("/reembed", dependencies=[Depends(require_token)])
def reembed(req: ReembedRequest):
    if req.modality not in ("text", "vision", "all"):
        raise HTTPException(status_code=400,
                            detail=f"modality must be text|vision|all, got {req.modality!r}")
    conn = connection.get_connection(paths.db_path())
    router_obj = emb_pkg.get_router()
    counts = queue_reembed(conn, router_obj, modality=req.modality)
    conn.commit()
    return {
        "ok": True,
        "queued": counts,
        "total": counts["text"] + counts["vision"],
    }
