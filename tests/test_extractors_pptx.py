from semanticsd.extractors.pptx import PptxExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_pptx


def test_extracts_slides_as_segments(tmp_path):
    p = make_pptx(tmp_path)
    out = PptxExtractor().extract(p)
    assert out.file_type == "pptx"
    assert len(out.segments) == 2
    assert out.segments[0].metadata == {"slide": 1}
    assert out.segments[1].metadata == {"slide": 2}
    assert "Slide One" in out.segments[0].text
    assert "Slide Two" in out.segments[1].text


def test_registry_picks_pptx(tmp_path):
    p = make_pptx(tmp_path)
    assert isinstance(registry.get_extractor(p), PptxExtractor)
