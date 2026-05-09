"""Image extractor — emits a vision segment (raw bytes) and an OCR text fallback.

The vision segment is always produced (so a configured vision embedder can
embed the image directly). The OCR text segment is best-effort: if Tesseract
isn't installed or OCR fails, only the vision segment remains.
"""
from __future__ import annotations
import logging
from pathlib import Path
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register

log = logging.getLogger(__name__)


@register
class ImageExtractor(Extractor):
    file_type = "image"
    extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic")

    def extract(self, path: Path) -> ExtractedDoc:
        segments: list[ExtractedSegment] = []
        metadata: dict = {}

        try:
            raw = path.read_bytes()
            descriptor = f"<image: {path.name}>"
            segments.append(ExtractedSegment(
                text=descriptor,
                byte_start=0,
                byte_end=len(descriptor.encode("utf-8")),
                modality="vision",
                image_data=raw,
                metadata={"source_path": str(path)},
            ))
        except Exception as e:
            log.warning("failed to read image bytes for %s: %s", path, e)
            metadata["read_error"] = str(e)

        try:
            import pytesseract
            from PIL import Image
            img = Image.open(str(path))
            text = pytesseract.image_to_string(img).strip()
            if text:
                segments.append(ExtractedSegment(
                    text=text,
                    byte_start=0,
                    byte_end=len(text.encode("utf-8")),
                    modality="text",
                    metadata={"source": "ocr"},
                ))
            else:
                metadata["ocr_error"] = "empty_result"
        except Exception as e:
            log.debug("OCR skipped for %s: %s", path, e)
            metadata.setdefault("ocr_error", str(e))

        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
            metadata=metadata,
        )
