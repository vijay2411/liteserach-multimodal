"""Abstract Extractor base + extracted-document model."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar
from pydantic import BaseModel


class ExtractedSegment(BaseModel):
    """One contiguous chunk of extracted text (e.g. a paragraph, page, or slide).

    `byte_start` and `byte_end` are byte offsets within the *extracted text*
    (not within the original binary). They let us point search results back to
    a region of the doc for highlighting.
    """
    text: str
    byte_start: int
    byte_end: int
    metadata: dict = {}  # e.g. {"page": 2}, {"slide": 3}


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
