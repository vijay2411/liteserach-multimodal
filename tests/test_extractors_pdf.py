from semanticsd.extractors.pdf import PdfExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_pdf


def test_extracts_pages_as_segments(tmp_path):
    p = make_pdf(tmp_path)
    out = PdfExtractor().extract(p)
    assert out.file_type == "pdf"
    assert len(out.segments) == 2
    assert out.segments[0].metadata == {"page": 1}
    assert out.segments[1].metadata == {"page": 2}
    assert "Hello" in out.segments[0].text
    assert "Page two" in out.segments[1].text


def test_registry_picks_pdf(tmp_path):
    p = make_pdf(tmp_path)
    assert isinstance(registry.get_extractor(p), PdfExtractor)
