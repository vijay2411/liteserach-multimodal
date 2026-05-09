"""DOCX extractor — one segment per paragraph, via python-docx."""
from __future__ import annotations
from pathlib import Path
from docx import Document
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class DocxExtractor(Extractor):
    file_type = "docx"
    extensions = (".docx",)

    def extract(self, path: Path) -> ExtractedDoc:
        doc = Document(str(path))
        segments: list[ExtractedSegment] = []
        cursor = 0
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"paragraph": i},
            ))
            cursor += seg_bytes + 1
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
        )
