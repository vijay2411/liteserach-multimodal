"""PDF extractor — one segment per page, via pypdf."""
from __future__ import annotations
from pathlib import Path
from pypdf import PdfReader
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class PdfExtractor(Extractor):
    file_type = "pdf"
    extensions = (".pdf",)

    def extract(self, path: Path) -> ExtractedDoc:
        reader = PdfReader(str(path))
        segments: list[ExtractedSegment] = []
        cursor = 0
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"page": i},
            ))
            cursor += seg_bytes + 1
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
        )
