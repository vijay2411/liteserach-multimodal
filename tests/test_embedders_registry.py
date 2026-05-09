"""Provider registry + build_embedder() factory."""
import pytest
from semanticsd.embedders import registry
from semanticsd.embedders.local import LocalEmbedder
from semanticsd.embedders.openai import OpenAIEmbedder
from semanticsd.embedders.ollama import OllamaEmbedder
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder


def test_registry_has_expected_keys():
    keys = set(registry.PROVIDER_REGISTRY.keys())
    expected = {"local", "openai", "ollama", "openai_compatible", "lmstudio", "vllm", "custom"}
    assert expected.issubset(keys)


def test_registry_entry_shape():
    for preset, entry in registry.PROVIDER_REGISTRY.items():
        assert "class" in entry
        assert "default_model" in entry or entry["default_model"] is None
        assert "needs_api_key" in entry


def test_build_local_embedder(monkeypatch):
    e = registry.build_embedder("local", config={})
    assert isinstance(e, LocalEmbedder)


def test_build_openai_requires_api_key():
    try:
        registry.build_embedder("openai", config={"api_key": ""})
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing OpenAI API key")


def test_build_openai_with_api_key():
    e = registry.build_embedder("openai", config={"api_key": "sk-test"})
    assert isinstance(e, OpenAIEmbedder)


def test_build_ollama_no_key_needed():
    e = registry.build_embedder("ollama", config={})
    assert isinstance(e, OllamaEmbedder)


def test_build_custom_requires_base_url():
    try:
        registry.build_embedder(
            "custom", config={"api_key": "x", "model": "m", "dim": 384}
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing base_url")


def test_build_custom_full():
    e = registry.build_embedder(
        "custom",
        config={
            "base_url": "http://localhost:1234/v1",
            "api_key": "anything",
            "model": "some-model",
            "dim": 384,
        },
    )
    assert isinstance(e, OpenAICompatibleEmbedder)


def test_unknown_preset_raises():
    try:
        registry.build_embedder("nonexistent", config={})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown preset")
