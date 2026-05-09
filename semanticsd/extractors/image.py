"""Image extractor — Tesseract OCR via pytesseract.

Graceful degrade: if Tesseract binary is missing, returns a doc with no
segments but an `ocr_error` in metadata.
"""
from __future__ import annotations
import logging
from pathlib import Path
import pytesseract
from PIL import Image
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register

log = logging.getLogger(__name__)


@register
class ImageExtractor(Extractor):
    file_type = "image"
    extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic")

    def extract(self, path: Path) -> ExtractedDoc:
        try:
            img = Image.open(str(path))
            text = pytesseract.image_to_string(img).strip()
        except pytesseract.TesseractNotFoundError as e:
            log.warning("Tesseract not installed; skipping OCR for %s", path)
            return ExtractedDoc(
                path=str(path),
                file_type=self.file_type,
                segments=[],
                metadata={"ocr_error": "tesseract_not_found", "detail": str(e)},
            )
        except Exception as e:
            log.warning("OCR failed for %s: %s", path, e)
            return ExtractedDoc(
                path=str(path),
                file_type=self.file_type,
                segments=[],
                metadata={"ocr_error": "ocr_failure", "detail": str(e)},
            )

        if not text:
            return ExtractedDoc(
                path=str(path),
                file_type=self.file_type,
                segments=[],
                metadata={"ocr_error": "empty_result"},
            )
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=[ExtractedSegment(
                text=text,
                byte_start=0,
                byte_end=len(text.encode("utf-8")),
            )],
        )
