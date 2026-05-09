"""Health endpoint."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from semanticsd import __version__
from semanticsd.server.auth import require_token
from semanticsd.db import connection
from semanticsd import paths
from semanticsd import embedders as emb_pkg

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

    embedder_section: dict
    try:
        embedder = emb_pkg.get_active_embedder()
    except Exception as e:
        embedder = None
        embedder_section = {"ok": False, "message": f"embedder build failed: {e}"}
    else:
        if embedder is None:
            embedder_section = {"ok": False, "message": "embedder not configured"}
        else:
            ok, msg = embedder.health_check()
            embedder_section = {
                "ok": ok,
                "message": msg,
                "provider_id": embedder.provider_id,
                "model_id": embedder.model_id,
                "dim": embedder.dim,
            }

    overall_ok = db_ok and embedder_section["ok"]
    return {
        "status": "ok" if overall_ok else "degraded",
        "version": __version__,
        "doc_count": doc_count,
        "vector_store": {"ok": db_ok},
        "embedder": embedder_section,
    }
