"""GET /v1/stats — aggregated counters for the dashboard.

One endpoint that pulls together everything the monitoring page wants
to show: chunk counts per modality, vec table sizes, job queue states,
file-type histogram, and total DB size on disk.
"""
from __future__ import annotations
import os
from fastapi import APIRouter, Depends
from semanticsd.server.auth import require_token
from semanticsd.db import connection
from semanticsd import paths

router = APIRouter()


@router.get("/stats", dependencies=[Depends(require_token)])
def stats() -> dict:
    db = paths.db_path()
    conn = connection.get_connection(db)

    n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    chunks_by_modality = {
        r[0]: int(r[1])
        for r in conn.execute(
            "SELECT modality, COUNT(*) FROM chunks GROUP BY modality"
        )
    }

    # Job-queue state
    jobs_by_status = {
        r[0]: int(r[1])
        for r in conn.execute(
            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
        )
    }
    for s in ("pending", "in_flight", "done", "failed"):
        jobs_by_status.setdefault(s, 0)

    # File-type histogram (top N)
    file_types = [
        {"file_type": r[0], "count": int(r[1])}
        for r in conn.execute(
            "SELECT file_type, COUNT(*) FROM files "
            "GROUP BY file_type ORDER BY 2 DESC LIMIT 20"
        )
    ]

    # Vec tables and their populated row counts
    vec_tables: list[dict] = []
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name LIKE 'vec_%' "
        "AND name NOT LIKE '%_info' AND name NOT LIKE '%_chunks' "
        "AND name NOT LIKE '%_rowids' AND name NOT LIKE '%_vector_chunks%'"
    ).fetchall()
    for (name,) in rows:
        try:
            cnt = int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            vec_tables.append({"table": name, "rows": cnt})
        except Exception:
            pass
    vec_tables.sort(key=lambda x: -x["rows"])

    # Embedder providers actually present in the corpus
    providers = [
        {"provider_id": r[0], "model_id": r[1], "modality": r[2],
         "dim": int(r[3]), "chunks": int(r[4])}
        for r in conn.execute(
            "SELECT provider_id, model_id, modality, dim, COUNT(*) "
            "FROM embedding_meta GROUP BY 1,2,3,4 ORDER BY 5 DESC"
        )
    ]

    # DB file size on disk (best-effort)
    try:
        db_size_bytes = os.path.getsize(db)
    except OSError:
        db_size_bytes = 0

    return {
        "files_total": n_files,
        "chunks_by_modality": chunks_by_modality,
        "jobs_by_status": jobs_by_status,
        "file_types": file_types,
        "vec_tables": vec_tables,
        "providers": providers,
        "db_size_bytes": db_size_bytes,
    }
