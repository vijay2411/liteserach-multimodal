"""Embedder layer — pluggable providers, one file per provider."""
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.local import LocalEmbedder
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder
from semanticsd.embedders.openai import OpenAIEmbedder
from semanticsd.embedders.ollama import OllamaEmbedder
from semanticsd.embedders.registry import PROVIDER_REGISTRY, build_embedder

__all__ = [
    "Embedder",
    "EmbedResult",
    "LocalEmbedder",
    "OpenAICompatibleEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "PROVIDER_REGISTRY",
    "build_embedder",
]


_active: Embedder | None = None


def get_active_embedder(force_reload: bool = False) -> Embedder | None:
    """Return the configured embedder for the running daemon.

    Reads `[embedding]` from the daemon's config. Caches the instance.
    Returns None if no provider is selected (preset is empty).
    """
    global _active
    if _active is not None and not force_reload:
        return _active

    from semanticsd import config as cfg_mod
    from semanticsd import keychain
    cfg = cfg_mod.load()
    preset = cfg.embedding.preset
    if not preset:
        return None

    api_key = ""
    if PROVIDER_REGISTRY.get(preset, {}).get("needs_api_key"):
        api_key = keychain.get_provider_key(preset) or ""

    config_dict = {
        "api_key": api_key,
        "base_url": cfg.embedding.base_url,
        "model": cfg.embedding.model,
        "dimensions": cfg.embedding.dimensions,
    }
    _active = build_embedder(preset, config_dict)
    return _active


def reset_active_embedder() -> None:
    """Clear the cached embedder (used by tests + after config changes)."""
    global _active
    _active = None
