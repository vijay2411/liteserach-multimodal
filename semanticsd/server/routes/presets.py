"""GET /v1/presets — provider registry for frontend dropdowns."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from semanticsd.server.auth import require_token
from semanticsd.embedders.registry import PROVIDER_REGISTRY

router = APIRouter()


@router.get("/presets", dependencies=[Depends(require_token)])
def presets() -> dict:
    return {"presets": PROVIDER_REGISTRY}
