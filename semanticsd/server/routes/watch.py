"""GET /v1/watch + POST /v1/watch/sweep — watcher status + manual full sweep."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Request
from semanticsd.server.auth import require_token

router = APIRouter()


def _power_controller(request: Request):
    pc = getattr(request.app.state, "power_controller", None)
    if pc is None:
        raise HTTPException(
            status_code=503,
            detail="power controller not initialized (daemon may have started in legacy mode)",
        )
    return pc


@router.get("/watch", dependencies=[Depends(require_token)])
def watch_status(request: Request):
    return _power_controller(request).status()


@router.post("/watch/sweep", dependencies=[Depends(require_token)])
async def watch_sweep(request: Request):
    pc = _power_controller(request)
    stats = await pc.force_sweep()
    return {"ok": True, "stats": stats}
