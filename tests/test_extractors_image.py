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


@pytest.mark.skipif(not HAS_TESS, reason="tesseract binary not installed")
def test_ocr_real(tmp_path):
    p = make_image_with_text(tmp_path, text="hello")
    out = ImageExtractor().extract(p)
    assert out.file_type == "image"
    full = " ".join(s.text for s in out.segments)
    assert "hello" in full.lower()


def test_graceful_degrade_when_tesseract_missing(tmp_path, monkeypatch):
    p = make_image_with_text(tmp_path, text="anything")
    import pytesseract
    def boom(*a, **kw):
        raise pytesseract.TesseractNotFoundError()
    monkeypatch.setattr(pytesseract, "image_to_string", boom)
    out = ImageExtractor().extract(p)
    assert out.file_type == "image"
    assert out.segments == []
    assert "ocr_error" in out.metadata


def test_registry_picks_image(tmp_path):
    p = make_image_with_text(tmp_path)
    assert isinstance(registry.get_extractor(p), ImageExtractor)
