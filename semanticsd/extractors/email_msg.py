"""Email (.eml) extractor — headers + plain-text body, via stdlib email."""
from __future__ import annotations
from pathlib import Path
from email import message_from_bytes
from email.policy import default as default_policy
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class EmailExtractor(Extractor):
    file_type = "email"
    extensions = (".eml",)

    def extract(self, path: Path) -> ExtractedDoc:
        msg = message_from_bytes(path.read_bytes(), policy=default_policy)
        body_part = msg.get_body(preferencelist=("plain", "html"))
        body_text = ""
        if body_part is not None:
            body_text = body_part.get_content()
            if body_part.get_content_type() == "text/html":
                from bs4 import BeautifulSoup
                body_text = BeautifulSoup(body_text, "html.parser").get_text(separator="\n", strip=True)
        text = body_text.strip()
        meta = {
            "subject": msg.get("Subject", ""),
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "date": msg.get("Date", ""),
        }
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=[ExtractedSegment(text=text, byte_start=0, byte_end=len(text.encode("utf-8")))],
            metadata={k: v for k, v in meta.items() if v},
        )
