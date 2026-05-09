from semanticsd.extractors.pdf import PdfExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_pdf


def test_extracts_pages_as_text_segments(tmp_path):
    p = make_pdf(tmp_path)
    out = PdfExtractor().extract(p)
    assert out.file_type == "pdf"
    text_segs = [s for s in out.segments if s.modality == "text"]
    assert len(text_segs) == 2
    assert text_segs[0].metadata == {"page": 1}
    assert text_segs[1].metadata == {"page": 2}
    assert "Hello" in text_segs[0].text
    assert "Page two" in text_segs[1].text


def test_emits_vision_segment_per_page(tmp_path):
    p = make_pdf(tmp_path)
    out = PdfExtractor().extract(p)
    vision_segs = [s for s in out.segments if s.modality == "vision"]
    assert len(vision_segs) == 2
    for v in vision_segs:
        assert v.image_data is not None
        assert v.image_data[:8].startswith(b"\x89PNG")
        assert "page" in v.metadata


def test_registry_picks_pdf(tmp_path):
    p = make_pdf(tmp_path)
    assert isinstance(registry.get_extractor(p), PdfExtractor)
