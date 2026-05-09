"""Provider registry + factory.

PROVIDER_REGISTRY maps preset id -> spec dict consumed by both the factory
(server-side embedder construction) and the /v1/presets endpoint
(frontend-side dropdown rendering).
"""
from __future__ import annotations
from typing import Any, Type
from semanticsd.embedders.base import Embedder
from semanticsd.embedders.vision_base import VisionEmbedder
from semanticsd.embedders.local import LocalEmbedder
from semanticsd.embedders.openai import OpenAIEmbedder
from semanticsd.embedders.ollama import OllamaEmbedder
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder
from semanticsd.embedders.gemini import GeminiTextEmbedder
from semanticsd.embedders.gemini_vision import GeminiVisionEmbedder
from semanticsd.embedders.qwen3_vl import LocalQwen3VisionEmbedder


PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "local": {
        "class": "LocalEmbedder",
        "default_model": "BAAI/bge-small-en-v1.5",
        "needs_api_key": False,
        "needs_base_url": False,
    },
    "ollama": {
        "class": "OllamaEmbedder",
        "default_model": "embeddinggemma",
        "needs_api_key": False,
        "needs_base_url": False,
        "default_base_url": "http://localhost:11434/v1",
    },
    "lmstudio": {
        "class": "OpenAICompatibleEmbedder",
        "default_model": None,
        "needs_api_key": False,
        "needs_base_url": True,
        "default_base_url": "http://localhost:1234/v1",
    },
    "vllm": {
        "class": "OpenAICompatibleEmbedder",
        "default_model": None,
        "needs_api_key": False,
        "needs_base_url": True,
        "default_base_url": "http://localhost:8000/v1",
    },
    "openai": {
        "class": "OpenAIEmbedder",
        "default_model": "text-embedding-3-small",
        "needs_api_key": True,
        "needs_base_url": False,
    },
    "gemini": {
        "class": "GeminiTextEmbedder",
        "default_model": "gemini-embedding-2",
        "needs_api_key": True,
        "needs_base_url": False,
    },
    "openai_compatible": {
        "class": "OpenAICompatibleEmbedder",
        "default_model": None,
        "needs_api_key": False,
        "needs_base_url": True,
    },
    "custom": {
        "class": "OpenAICompatibleEmbedder",
        "default_model": None,
        "needs_api_key": False,
        "needs_base_url": True,
    },
}

_CLASS_BY_NAME: dict[str, Type[Embedder]] = {
    "LocalEmbedder": LocalEmbedder,
    "OllamaEmbedder": OllamaEmbedder,
    "OpenAIEmbedder": OpenAIEmbedder,
    "OpenAICompatibleEmbedder": OpenAICompatibleEmbedder,
    "GeminiTextEmbedder": GeminiTextEmbedder,
}


VISION_PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "gemini": {
        "class": "GeminiVisionEmbedder",
        "default_model": "gemini-embedding-2",
        "needs_api_key": True,
        "needs_base_url": False,
    },
    "qwen3_vl_local": {
        "class": "LocalQwen3VisionEmbedder",
        "default_model": "Qwen/Qwen3-VL-Embedding-2B",
        "needs_api_key": False,
        "needs_base_url": False,
    },
}

_VISION_CLASS_BY_NAME: dict[str, Type[VisionEmbedder]] = {
    "GeminiVisionEmbedder": GeminiVisionEmbedder,
    "LocalQwen3VisionEmbedder": LocalQwen3VisionEmbedder,
}


def build_embedder(preset: str, config: dict[str, Any]) -> Embedder:
    """Construct an Embedder instance from a preset id + config dict.

    Required config keys depend on the preset (see PROVIDER_REGISTRY[preset]):
      - needs_api_key: include "api_key"
      - needs_base_url: include "base_url"
    Optional: "model", "dim", "dimensions".

    Raises ValueError on unknown preset or missing required config.
    """
    if preset not in PROVIDER_REGISTRY:
        raise ValueError(f"unknown embedder preset: {preset!r}")
    entry = PROVIDER_REGISTRY[preset]
    cls = _CLASS_BY_NAME[entry["class"]]

    if entry["needs_api_key"] and not config.get("api_key"):
        raise ValueError(f"preset {preset!r} requires an api_key")
    if entry["needs_base_url"] and not config.get("base_url") and not entry.get("default_base_url"):
        raise ValueError(f"preset {preset!r} requires a base_url")

    if cls is LocalEmbedder:
        return LocalEmbedder(model_id=config.get("model") or entry["default_model"])
    if cls is OpenAIEmbedder:
        return OpenAIEmbedder(
            api_key=config["api_key"],
            model=config.get("model") or entry["default_model"],
            dimensions=config.get("dimensions", 0),
        )
    if cls is GeminiTextEmbedder:
        return GeminiTextEmbedder(
            api_key=config["api_key"],
            model=config.get("model") or entry["default_model"],
        )
    if cls is OllamaEmbedder:
        base_url = config.get("base_url") or entry.get("default_base_url")
        return OllamaEmbedder(
            model=config.get("model") or entry["default_model"],
            base_url=base_url,
        )
    if cls is OpenAICompatibleEmbedder:
        base_url = config.get("base_url") or entry.get("default_base_url")
        if not base_url:
            raise ValueError(f"preset {preset!r} requires a base_url")
        if not config.get("model"):
            raise ValueError(f"preset {preset!r} requires a model")
        if not config.get("dim"):
            raise ValueError(f"preset {preset!r} requires a dim")
        return OpenAICompatibleEmbedder(
            base_url=base_url,
            api_key=config.get("api_key", ""),
            model=config["model"],
            dim=int(config["dim"]),
            dimensions=config.get("dimensions", 0),
        )
    raise ValueError(f"no factory branch for class {entry['class']!r}")


def build_vision_embedder(preset: str, config: dict[str, Any]) -> VisionEmbedder:
    """Construct a VisionEmbedder from a preset id + config dict."""
    if preset not in VISION_PROVIDER_REGISTRY:
        raise ValueError(f"unknown vision embedder preset: {preset!r}")
    entry = VISION_PROVIDER_REGISTRY[preset]
    cls = _VISION_CLASS_BY_NAME[entry["class"]]
    if entry["needs_api_key"] and not config.get("api_key"):
        raise ValueError(f"vision preset {preset!r} requires an api_key")
    if cls is GeminiVisionEmbedder:
        return GeminiVisionEmbedder(
            api_key=config["api_key"],
            model=config.get("model") or entry["default_model"],
        )
    if cls is LocalQwen3VisionEmbedder:
        return LocalQwen3VisionEmbedder(
            model=config.get("model") or entry["default_model"],
        )
    raise ValueError(f"no factory branch for vision class {entry['class']!r}")
