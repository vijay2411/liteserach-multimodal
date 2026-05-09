"""Search types: SearchResult, SearchOptions."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

Mode = Literal["semantic", "filename", "grep", "hybrid"]
Modality = Literal["text", "vision"]


class SearchOptions(BaseModel):
    mode: Mode = "hybrid"
    limit: int = 20
    cwd: Path | None = None
    all: bool = False
    vision: bool = True  # cross-modal search; auto-disabled if no vision embedder

    model_config = {"arbitrary_types_allowed": True}


class SearchResult(BaseModel):
    path: str
    modality: Modality
    mode: str  # actual mode that produced this row (semantic|filename|grep|hybrid)
    score: float
    snippet: str | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    chunk_id: int | None = None  # null for filename matches (file-level)
    file_id: int
    metadata: dict = Field(default_factory=dict)
