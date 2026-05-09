"""Health endpoint — surfaces both text and vision embedders."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from semanticsd import __version__
from semanticsd.server.auth import require_token
from semanticsd.db import connection
from semanticsd import paths
from semanticsd import embedders as emb_pkg

router = APIRouter()


def _embedder_section(em, kind_label: str) -> dict:
    if em is None:
        return {"ok": False, "message": f"{kind_label} embedder not configured"}
    try:
        ok, msg = em.health_check()
    except Exception as e:
        return {"ok": False, "message": f"{kind_label} health_check raised: {e}"}
    return {
        "ok": ok,
        "message": msg,
        "provider_id": em.provider_id,
        "model_id": em.model_id,
        "dim": em.dim,
    }


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

    try:
        router_obj = emb_pkg.get_router()
        text_em = router_obj.text
        vision_em = router_obj.vision
    except Exception as e:
        text_em = None
        vision_em = None

    text_section = _embedder_section(text_em, "text")
    # Vision is optional — if not configured, report a benign "disabled" status
    # rather than degraded; only mark degraded when configured-but-broken.
    if vision_em is None:
        vision_section = {"ok": True, "message": "vision disabled (not configured)"}
    else:
        vision_section = _embedder_section(vision_em, "vision")

    # Budget block — best-effort: skip if usage table or config aren't available.
    budget_block: dict = {}
    try:
        from semanticsd import config as cfg_mod
        from semanticsd.usage.budget import BudgetGate
        cfg = cfg_mod.load()
        gate = BudgetGate(
            conn,
            monthly_limit_usd=cfg.budget.monthly_limit_usd,
            warning_threshold=cfg.budget.warning_threshold,
        )
        s = gate.status()
        budget_block = {
            "spent_this_month_usd": s.spent_this_month_usd,
            "limit_usd": s.limit_usd,
            "percent_used": s.percent_used,
            "blocked": s.blocked,
        }
    except Exception as e:
        budget_block = {"error": str(e)}

    overall_ok = db_ok and text_section["ok"] and vision_section["ok"]
    return {
        "status": "ok" if overall_ok else "degraded",
        "version": __version__,
        "doc_count": doc_count,
        "vector_store": {"ok": db_ok},
        "embedders": {
            "text": text_section,
            "vision": vision_section,
        },
        # Back-compat: keep "embedder" pointing at text for older clients.
        "embedder": text_section,
        "budget": budget_block,
    }
