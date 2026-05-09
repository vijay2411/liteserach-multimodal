"""POST /v1/embedder/test — round-trip a probe embed against a candidate config.

Used by frontends to render a 'Test Connection' button. Never persists state.
"""
from __future__ import annotations
import time
from typing import Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from semanticsd.server.auth import require_token
from semanticsd.embedders.registry import build_embedder

router = APIRouter()


class TestRequest(BaseModel):
    preset: str
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    dim: int | None = None
    dimensions: int | None = None


@router.post("/embedder/test", dependencies=[Depends(require_token)])
def embedder_test(req: TestRequest) -> dict[str, Any]:
    config: dict[str, Any] = {
        "base_url": req.base_url or "",
        "api_key": req.api_key or "",
        "model": req.model or "",
        "dimensions": req.dimensions or 0,
    }
    if req.dim is not None:
        config["dim"] = req.dim

    try:
        embedder = build_embedder(req.preset, config=config)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    start = time.perf_counter()
    try:
        result = embedder.embed(["ping"], kind="query")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "ok": True,
        "provider_id": embedder.provider_id,
        "model_id": embedder.model_id,
        "dim": len(result.vectors[0]) if result.vectors else embedder.dim,
        "latency_ms": elapsed_ms,
    }
