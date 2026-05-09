"""RTF extractor — strips control codes, via striprtf."""
from __future__ import annotations
from pathlib import Path
from striprtf.striprtf import rtf_to_text
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class RtfExtractor(Extractor):
    file_type = "rtf"
    extensions = (".rtf",)

    def extract(self, path: Path) -> ExtractedDoc:
        raw = path.read_text(errors="replace")
        text = rtf_to_text(raw).strip()
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=[
                ExtractedSegment(
                    text=text,
                    byte_start=0,
                    byte_end=len(text.encode("utf-8")),
                )
            ],
        )
