from semanticsd.extractors.docx import DocxExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_docx


def test_extracts_paragraphs(tmp_path):
    p = make_docx(tmp_path)
    out = DocxExtractor().extract(p)
    assert out.file_type == "docx"
    full = "\n".join(s.text for s in out.segments)
    assert "Title" in full
    assert "First paragraph" in full
    assert "Second paragraph" in full


def test_registry_picks_docx(tmp_path):
    p = make_docx(tmp_path)
    assert isinstance(registry.get_extractor(p), DocxExtractor)
