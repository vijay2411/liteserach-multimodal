"""Real-Gemini smoke test — requires API key in ~/secrets/gemini_api_key."""
import os
import pathlib
import pytest


def _key():
    p = pathlib.Path.home() / "secrets" / "gemini_api_key"
    if p.exists():
        return p.read_text().strip()
    return os.environ.get("GEMINI_API_KEY")


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _key(), reason="no gemini key available"),
]


def test_gemini_text_real():
    from semanticsd.embedders.gemini import GeminiTextEmbedder
    e = GeminiTextEmbedder(api_key=_key())
    out = e.embed(["semantic search query"], kind="query")
    assert len(out.vectors) == 1
    assert len(out.vectors[0]) == 3072


def test_gemini_vision_real(tmp_path):
    from semanticsd.embedders.gemini_vision import GeminiVisionEmbedder
    from tests._fixtures import make_image_with_text

    img_path = make_image_with_text(tmp_path, text="hello world")
    img_bytes = img_path.read_bytes()

    e = GeminiVisionEmbedder(api_key=_key())
    out = e.embed_images([img_bytes])
    assert len(out.vectors) == 1
    assert len(out.vectors[0]) == 3072
