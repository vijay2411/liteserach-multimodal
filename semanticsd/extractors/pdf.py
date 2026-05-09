"""PDF extractor — text segments (pypdf) + vision segments (pypdfium2 page renders)."""
from __future__ import annotations
import io
import logging
from pathlib import Path
from pypdf import PdfReader
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register

log = logging.getLogger(__name__)

RENDER_DPI = 150
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB cap per page


@register
class PdfExtractor(Extractor):
    file_type = "pdf"
    extensions = (".pdf",)

    def extract(self, path: Path) -> ExtractedDoc:
        segments: list[ExtractedSegment] = []
        cursor = 0

        try:
            reader = PdfReader(str(path))
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
                    modality="text",
                ))
                cursor += seg_bytes + 1
        except Exception as e:
            log.warning("pypdf text extraction failed for %s: %s", path, e)

        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(path))
            scale = RENDER_DPI / 72.0
            for i in range(len(pdf)):
                page = pdf[i]
                bitmap = page.render(scale=scale).to_pil()
                buf = io.BytesIO()
                bitmap.save(buf, format="PNG", optimize=True)
                img_bytes = buf.getvalue()
                if len(img_bytes) > MAX_IMAGE_BYTES:
                    log.info(
                        "skipping vision for %s page %d: %d bytes > cap",
                        path, i + 1, len(img_bytes),
                    )
                    continue
                descriptor = f"<image: {path.name} page={i + 1}>"
                segments.append(ExtractedSegment(
                    text=descriptor,
                    byte_start=0,
                    byte_end=len(descriptor.encode("utf-8")),
                    modality="vision",
                    image_data=img_bytes,
                    metadata={"page": i + 1},
                ))
        except Exception as e:
            log.warning("pypdfium2 rendering failed for %s: %s", path, e)

        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
        )
