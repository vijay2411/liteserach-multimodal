"""GeminiTextEmbedder — uses Gemini Embedding 2 REST API (mocked)."""
import json
import pytest
from semanticsd.embedders.gemini import GeminiTextEmbedder


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


def test_provider_id_is_gemini():
    e = GeminiTextEmbedder(api_key="k")
    assert e.provider_id == "gemini"


def test_default_model_and_dim():
    e = GeminiTextEmbedder(api_key="k")
    assert e.model_id == "gemini-embedding-2"
    assert e.dim == 3072


def test_requires_api_key():
    with pytest.raises(ValueError):
        GeminiTextEmbedder(api_key="")


def test_embed_calls_endpoint(monkeypatch):
    e = GeminiTextEmbedder(api_key="testkey")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return _FakeResp({"embedding": {"values": [0.1] * 3072}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = e.embed(["hello"], kind="doc")
    assert len(out.vectors) == 1
    assert len(out.vectors[0]) == 3072
    assert "gemini-embedding-2:embedContent" in captured["url"]
    assert "testkey" in captured["url"]
    assert captured["body"]["content"]["parts"][0]["text"] == "hello"


def test_embed_handles_batch(monkeypatch):
    e = GeminiTextEmbedder(api_key="k")
    n_calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        n_calls["n"] += 1
        return _FakeResp({"embedding": {"values": [0.1] * 3072}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = e.embed(["a", "b", "c"], kind="doc")
    assert len(out.vectors) == 3
    assert n_calls["n"] == 3


def test_raises_on_empty_response(monkeypatch):
    e = GeminiTextEmbedder(api_key="k")

    def fake_urlopen(req, timeout=None):
        return _FakeResp({"error": "boom"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError):
        e.embed(["x"], kind="doc")
