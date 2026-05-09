"""EPUB extractor — one segment per chapter (HTML stripped), via ebooklib."""
from __future__ import annotations
from pathlib import Path
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class EpubExtractor(Extractor):
    file_type = "epub"
    extensions = (".epub",)

    def extract(self, path: Path) -> ExtractedDoc:
        book = epub.read_epub(str(path))
        title = book.get_metadata("DC", "title")
        title_str = title[0][0] if title else None

        segments: list[ExtractedSegment] = []
        cursor = 0
        for i, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"chapter": i + 1, "name": item.get_name()},
            ))
            cursor += seg_bytes + 1

        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
            metadata={"title": title_str} if title_str else {},
        )
