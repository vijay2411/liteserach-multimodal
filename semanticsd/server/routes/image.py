"""GET /v1/image/<chunk_id> — serve the image_blob for a vision chunk.

Used by the web UI to render thumbnails next to vision-modality search
results. Returns 404 if the chunk doesn't exist or has no blob.
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from semanticsd.server.auth import require_token_or_query
from semanticsd.db import connection
from semanticsd import paths

router = APIRouter()


def _detect_mime(data: bytes) -> str:
    if data[:8].startswith(b"\x89PNG\r\n"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


@router.get("/image/{chunk_id}", dependencies=[Depends(require_token_or_query)])
def get_chunk_image(chunk_id: int):
    conn = connection.get_connection(paths.db_path())
    row = conn.execute(
        "SELECT image_blob FROM chunks WHERE id = ? AND modality = 'vision'",
        (chunk_id,),
    ).fetchone()
    if row is None or row[0] is None:
        raise HTTPException(status_code=404, detail="image not found")
    blob = bytes(row[0])
    return Response(
        content=blob,
        media_type=_detect_mime(blob),
        headers={"Cache-Control": "private, max-age=3600"},
    )
