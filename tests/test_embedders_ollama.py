"""OllamaEmbedder — local Ollama via its OpenAI-compatible /v1 endpoint."""
import pytest
from types import SimpleNamespace
from semanticsd.embedders.ollama import OllamaEmbedder


class FakeEmbeddings:
    def create(self, model, input, **kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * 768) for _ in input],
            usage=None,  # Ollama doesn't always populate usage
        )


class FakeClient:
    def __init__(self, **kwargs):
        self.embeddings = FakeEmbeddings()


@pytest.fixture
def fake_openai(monkeypatch):
    last = {}

    def factory(**kwargs):
        last["kwargs"] = kwargs
        return FakeClient(**kwargs)

    monkeypatch.setattr(
        "semanticsd.embedders.openai_compatible.OpenAI",
        factory,
    )
    return last


def test_default_base_url_is_localhost(fake_openai):
    e = OllamaEmbedder()
    assert e.base_url == "http://localhost:11434/v1"


def test_default_model_is_nomic(fake_openai):
    e = OllamaEmbedder()
    assert e.model_id == "nomic-embed-text"
    assert e.dim == 768


def test_provider_id_is_ollama(fake_openai):
    e = OllamaEmbedder()
    assert e.provider_id == "ollama"


def test_no_api_key_required(fake_openai):
    e = OllamaEmbedder()
    # The internal SDK client gets a placeholder; the user passed nothing.
    assert e.api_key == "ollama"


def test_cost_is_zero(fake_openai):
    e = OllamaEmbedder()
    assert e.cost_per_million_input_tokens_usd == 0.0


def test_input_tokens_falls_back_to_heuristic_when_no_usage(fake_openai):
    e = OllamaEmbedder()
    out = e.embed(["abcdefgh"], kind="doc")  # 8 chars => 2 tokens
    assert out.input_tokens == 2
