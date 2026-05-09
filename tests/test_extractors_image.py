import pytest
from semanticsd.extractors.image import ImageExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_image_with_text


def _has_tesseract() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


HAS_TESS = _has_tesseract()


def test_image_extractor_always_emits_vision_segment(tmp_path):
    p = make_image_with_text(tmp_path, text="hello")
    out = ImageExtractor().extract(p)
    assert out.file_type == "image"
    vision_segs = [s for s in out.segments if s.modality == "vision"]
    assert len(vision_segs) == 1
    assert vision_segs[0].image_data is not None
    assert vision_segs[0].image_data[:8].startswith(b"\x89PNG")
    assert vision_segs[0].text.startswith("<image:")


@pytest.mark.skipif(not HAS_TESS, reason="tesseract binary not installed")
def test_ocr_emits_text_segment(tmp_path):
    p = make_image_with_text(tmp_path, text="hello")
    out = ImageExtractor().extract(p)
    text_segs = [s for s in out.segments if s.modality == "text"]
    assert len(text_segs) == 1
    assert "hello" in text_segs[0].text.lower()


def test_graceful_degrade_when_ocr_fails(tmp_path, monkeypatch):
    """OCR failure must not block the vision segment."""
    p = make_image_with_text(tmp_path, text="anything")
    import pytesseract

    def boom(*a, **kw):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", boom)
    out = ImageExtractor().extract(p)
    assert out.file_type == "image"
    vision_segs = [s for s in out.segments if s.modality == "vision"]
    assert len(vision_segs) == 1  # vision still produced
    text_segs = [s for s in out.segments if s.modality == "text"]
    assert len(text_segs) == 0  # OCR fallback failed
    assert "ocr_error" in out.metadata


def test_registry_picks_image(tmp_path):
    p = make_image_with_text(tmp_path)
    assert isinstance(registry.get_extractor(p), ImageExtractor)
