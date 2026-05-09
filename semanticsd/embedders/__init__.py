"""Embedder layer — pluggable providers, one file per provider."""
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.vision_base import VisionEmbedder
from semanticsd.embedders.local import LocalEmbedder
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder
from semanticsd.embedders.openai import OpenAIEmbedder
from semanticsd.embedders.ollama import OllamaEmbedder
from semanticsd.embedders.gemini import GeminiTextEmbedder
from semanticsd.embedders.gemini_vision import GeminiVisionEmbedder
from semanticsd.embedders.qwen3_vl import LocalQwen3VisionEmbedder
from semanticsd.embedders.registry import (
    PROVIDER_REGISTRY,
    VISION_PROVIDER_REGISTRY,
    build_embedder,
    build_vision_embedder,
)
from semanticsd.embedders.router import EmbedderRouter

__all__ = [
    "Embedder",
    "VisionEmbedder",
    "EmbedResult",
    "LocalEmbedder",
    "OpenAICompatibleEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "GeminiTextEmbedder",
    "GeminiVisionEmbedder",
    "LocalQwen3VisionEmbedder",
    "PROVIDER_REGISTRY",
    "VISION_PROVIDER_REGISTRY",
    "build_embedder",
    "build_vision_embedder",
    "EmbedderRouter",
    "get_router",
    "get_active_embedder",
    "reset_active_embedder",
]


_router: EmbedderRouter | None = None


def get_router(force_reload: bool = False) -> EmbedderRouter:
    """Return the configured router for the running daemon. Cached."""
    global _router
    if _router is not None and not force_reload:
        return _router
    from semanticsd import config as cfg_mod
    cfg = cfg_mod.load()
    _router = EmbedderRouter.from_config(cfg)
    return _router


def get_active_embedder(force_reload: bool = False) -> Embedder | None:
    """Back-compat: returns the text embedder from the router."""
    return get_router(force_reload).text


def reset_active_embedder() -> None:
    """Clear the cached router (used by tests + after config changes)."""
    global _router
    _router = None
