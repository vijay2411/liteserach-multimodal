"""OpenAIEmbedder — thin subclass of OpenAICompatibleEmbedder."""
import pytest
from types import SimpleNamespace
from semanticsd.embedders.openai import OpenAIEmbedder, COST_PER_MILLION


class FakeEmbeddings:
    def create(self, model, input, **kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input],
            usage=SimpleNamespace(prompt_tokens=10),
        )


class FakeClient:
    def __init__(self, **kwargs):
        self.embeddings = FakeEmbeddings()
        self.kwargs = kwargs


@pytest.fixture
def fake_openai(monkeypatch):
    """Patch the OpenAI symbol used by openai_compatible (its parent module)."""
    last = {}

    def factory(**kwargs):
        last["kwargs"] = kwargs
        c = FakeClient(**kwargs)
        return c

    monkeypatch.setattr(
        "semanticsd.embedders.openai_compatible.OpenAI",
        factory,
    )
    return last


def test_default_base_url_is_openai(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test")
    assert e.base_url == "https://api.openai.com/v1"
    assert fake_openai["kwargs"]["base_url"] == "https://api.openai.com/v1"


def test_default_model_is_3_small(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test")
    assert e.model_id == "text-embedding-3-small"
    assert e.dim == 1536


def test_provider_id_is_openai(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test")
    assert e.provider_id == "openai"


def test_cost_table_for_3_small(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test")
    assert e.cost_per_million_input_tokens_usd == COST_PER_MILLION["text-embedding-3-small"]


def test_cost_table_for_3_large(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-large")
    assert e.cost_per_million_input_tokens_usd == COST_PER_MILLION["text-embedding-3-large"]


def test_cost_zero_for_unknown_model(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test", model="some-future-model")
    assert e.cost_per_million_input_tokens_usd == 0.0
