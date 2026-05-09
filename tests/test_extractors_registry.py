"""Extractor base + registry shape."""
import pytest
from pathlib import Path
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors import registry


def test_extracted_doc_required_fields():
    seg = ExtractedSegment(text="hello", byte_start=0, byte_end=5)
    doc = ExtractedDoc(path="/tmp/foo.txt", file_type="text", segments=[seg])
    assert doc.path == "/tmp/foo.txt"
    assert doc.file_type == "text"
    assert doc.segments[0].text == "hello"


def test_extractor_is_abstract():
    try:
        Extractor()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("Extractor should be abstract")


def test_concrete_extractor():
    class Stub(Extractor):
        file_type = "stub"
        extensions = (".stub",)
        def extract(self, path: Path) -> ExtractedDoc:
            return ExtractedDoc(path=str(path), file_type="stub",
                                segments=[ExtractedSegment(text="x", byte_start=0, byte_end=1)])
    s = Stub()
    out = s.extract(Path("/tmp/x.stub"))
    assert out.file_type == "stub"


def test_registry_returns_none_for_unknown(tmp_path):
    p = tmp_path / "unknown.xyz"
    p.write_text("")
    assert registry.get_extractor(p) is None
