"""HTML extractor — strips tags, keeps visible text + title."""
from __future__ import annotations
from pathlib import Path
from bs4 import BeautifulSoup
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class HtmlExtractor(Extractor):
    file_type = "html"
    extensions = (".html", ".htm", ".xhtml")

    def extract(self, path: Path) -> ExtractedDoc:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = (soup.title.string.strip() if soup.title and soup.title.string else None)
        text = soup.get_text(separator="\n", strip=True)
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=[ExtractedSegment(text=text, byte_start=0, byte_end=len(text.encode("utf-8")))],
            metadata={"title": title} if title else {},
        )
