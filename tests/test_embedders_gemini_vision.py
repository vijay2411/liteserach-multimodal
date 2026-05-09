"""GeminiVisionEmbedder — vision via inline_data parts."""
import base64
import json
import pytest
from semanticsd.embedders.gemini_vision import GeminiVisionEmbedder, _detect_mime


class _FakeResp:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def test_provider_and_dim():
    e = GeminiVisionEmbedder(api_key="k")
    assert e.provider_id == "gemini"
    assert e.model_id == "gemini-embedding-2"
    assert e.dim == 3072


def test_requires_api_key():
    with pytest.raises(ValueError):
        GeminiVisionEmbedder(api_key="")


def test_mime_detection():
    assert _detect_mime(b"\x89PNG\r\n\x1a\n") == "image/png"
    assert _detect_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert _detect_mime(b"RIFF1234WEBPxxxx") == "image/webp"
    assert _detect_mime(b"unknown") == "application/octet-stream"


def test_embed_images_sends_inline_data(monkeypatch):
    e = GeminiVisionEmbedder(api_key="testkey")
    bodies = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        return _FakeResp({"embedding": {"values": [0.1] * 3072}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    img1 = b"\x89PNG\r\n\x1a\nfake_png_data"
    img2 = b"\xff\xd8\xff\xe0fake_jpeg_data"
    out = e.embed_images([img1, img2])

    assert len(out.vectors) == 2
    assert len(bodies) == 2
    part = bodies[0]["content"]["parts"][0]
    assert "inline_data" in part
    assert part["inline_data"]["mime_type"] == "image/png"
    assert base64.b64decode(part["inline_data"]["data"]) == img1
    part2 = bodies[1]["content"]["parts"][0]
    assert part2["inline_data"]["mime_type"] == "image/jpeg"
