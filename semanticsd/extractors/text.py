"""Plain-text-on-disk extractor: text, markdown, code, structured config files."""
from __future__ import annotations
from pathlib import Path
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


_TEXT_EXTS = (
    # Text and prose
    ".txt", ".md", ".rst", ".org", ".log",
    # Code
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java", ".kt",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".swift", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".fish", ".lua", ".pl", ".scala", ".clj",
    # Structured data / config
    ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".xml", ".html",
    ".sql", ".env", ".conf", ".ini", ".gitignore", ".dockerfile",
)


@register
class TextExtractor(Extractor):
    file_type = "text"
    extensions = _TEXT_EXTS

    def extract(self, path: Path) -> ExtractedDoc:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
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
