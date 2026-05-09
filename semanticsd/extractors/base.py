"""Abstract Extractor base + extracted-document model."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal
from pydantic import BaseModel


class ExtractedSegment(BaseModel):
    """One contiguous chunk of extracted content (text or image).

    For text segments: `text` holds the content; `byte_start`/`byte_end` are
    offsets into the extracted text. `image_data` is None.

    For vision segments: `image_data` holds raw image bytes (PNG/JPEG/WebP).
    `text` holds a synthetic descriptor (e.g. "<image: page=3>") used for
    FTS and result display. `byte_start`/`byte_end` cover the descriptor.
    """
    text: str
    byte_start: int
    byte_end: int
    metadata: dict = {}  # e.g. {"page": 2}, {"slide": 3}
    modality: Literal["text", "vision"] = "text"
    image_data: bytes | None = None


class ExtractedDoc(BaseModel):
    path: str
    file_type: str
    segments: list[ExtractedSegment]
    metadata: dict = {}  # e.g. {"title": "...", "author": "..."}


class Extractor(ABC):
    """One file class per concrete subclass.

    Subclasses set:
      file_type: str         — short identifier for the `files.file_type` column
      extensions: tuple[str] — file extensions handled, lowercase, with dot, e.g. (".pdf",)
    """

    file_type: ClassVar[str] = ""
    extensions: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def extract(self, path: Path) -> ExtractedDoc: ...
