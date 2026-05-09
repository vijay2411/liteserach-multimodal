"""GET /v1/power + POST /v1/power — power-mode inspection + control."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from semanticsd.server.auth import require_token

router = APIRouter()


class PowerRequest(BaseModel):
    mode: str  # "active" | "saver"


def _power_controller(request: Request):
    pc = getattr(request.app.state, "power_controller", None)
    if pc is None:
        raise HTTPException(
            status_code=503, detail="power controller not initialized",
        )
    return pc


@router.get("/power", dependencies=[Depends(require_token)])
def power_get(request: Request):
    pc = _power_controller(request)
    s = pc.status()
    return {
        "mode": s["mode"],
        "auto_saver_on_battery": s["auto_saver_on_battery"],
        "power_source": s["power_source"],
    }


@router.post("/power", dependencies=[Depends(require_token)])
async def power_set(request: Request, body: PowerRequest):
    if body.mode not in ("active", "saver"):
        raise HTTPException(status_code=400, detail=f"unknown mode: {body.mode!r}")
    pc = _power_controller(request)
    await pc.set_mode(body.mode, reason="api")
    return {"ok": True, "mode": pc.mode}
