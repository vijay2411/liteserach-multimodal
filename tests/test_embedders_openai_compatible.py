"""OpenAICompatibleEmbedder — covers any /v1/embeddings server.
The OpenAI Python SDK is monkey-patched; no network calls.
"""
import pytest
from types import SimpleNamespace
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder


class FakeEmbeddings:
    """Replaces openai.OpenAI().embeddings."""
    def __init__(self, dim=384):
        self.dim = dim
        self.last_request = None

    def create(self, model, input, **kwargs):
        self.last_request = {"model": model, "input": input, "kwargs": kwargs}
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * self.dim) for _ in input],
            usage=SimpleNamespace(prompt_tokens=sum(len(t) // 4 for t in input)),
        )


class FakeClient:
    def __init__(self, dim=384):
        self.embeddings = FakeEmbeddings(dim=dim)


@pytest.fixture
def fake_openai(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(
        "semanticsd.embedders.openai_compatible.OpenAI",
        lambda **kwargs: fake,
    )
    return fake


def test_metadata(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="anything",
        model="some-embed-model",
        dim=384,
    )
    assert e.provider_id == "openai_compatible"
    assert e.model_id == "some-embed-model"
    assert e.dim == 384


def test_embed_calls_sdk(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
    )
    out = e.embed(["hello", "world"], kind="doc")
    assert len(out.vectors) == 2
    assert all(len(v) == 384 for v in out.vectors)
    assert fake_openai.embeddings.last_request["model"] == "m"
    assert fake_openai.embeddings.last_request["input"] == ["hello", "world"]


def test_embed_passes_dimensions_when_set(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
        dimensions=512,  # Matryoshka truncation
    )
    e.embed(["x"], kind="doc")
    assert fake_openai.embeddings.last_request["kwargs"].get("dimensions") == 512


def test_embed_does_not_pass_dimensions_when_zero(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
        dimensions=0,
    )
    e.embed(["x"], kind="doc")
    assert "dimensions" not in fake_openai.embeddings.last_request["kwargs"]


def test_input_tokens_from_sdk_usage(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
    )
    out = e.embed(["abcdefgh", "ij"], kind="doc")
    assert out.input_tokens == 2 + 0  # heuristic from FakeEmbeddings


def test_health_check_does_a_short_embed(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
    )
    ok, msg = e.health_check()
    assert ok is True


def test_estimate_tokens_heuristic(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
    )
    assert e.estimate_tokens(["abcdefgh", "ij"]) == 2
