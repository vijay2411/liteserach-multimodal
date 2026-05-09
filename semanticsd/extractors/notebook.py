"""Jupyter notebook (.ipynb) extractor — one segment per code/markdown cell."""
from __future__ import annotations
import json
from pathlib import Path
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class NotebookExtractor(Extractor):
    file_type = "notebook"
    extensions = (".ipynb",)

    def extract(self, path: Path) -> ExtractedDoc:
        nb = json.loads(path.read_text())
        segments: list[ExtractedSegment] = []
        cursor = 0
        for i, cell in enumerate(nb.get("cells", [])):
            ctype = cell.get("cell_type", "")
            if ctype not in ("code", "markdown"):
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            text = source.strip()
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"cell_index": i, "cell_type": ctype},
            ))
            cursor += seg_bytes + 1
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
        )
