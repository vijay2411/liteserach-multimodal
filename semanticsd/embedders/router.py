"""EmbedderRouter — holds one Embedder (text) + one VisionEmbedder.

Lazy-loaded from config. Replaces the old _active singleton.
Each modality is independent: text is mandatory, vision is optional.
"""
from __future__ import annotations
import logging
from typing import Literal
from semanticsd.embedders.base import Embedder
from semanticsd.embedders.vision_base import VisionEmbedder
from semanticsd.embedders.registry import (
    PROVIDER_REGISTRY,
    VISION_PROVIDER_REGISTRY,
    build_embedder,
    build_vision_embedder,
)

log = logging.getLogger(__name__)
Modality = Literal["text", "vision"]


class EmbedderRouter:
    def __init__(
        self,
        text: Embedder | None = None,
        vision: VisionEmbedder | None = None,
    ):
        self.text = text
        self.vision = vision

    def get(self, modality: Modality) -> Embedder | VisionEmbedder | None:
        if modality == "text":
            return self.text
        if modality == "vision":
            return self.vision
        raise ValueError(f"unknown modality: {modality!r}")

    @classmethod
    def from_config(cls, cfg) -> "EmbedderRouter":
        from semanticsd import keychain

        text_em: Embedder | None = None
        vision_em: VisionEmbedder | None = None

        text_cfg = cfg.embedding.text
        if text_cfg and text_cfg.preset:
            api_key = ""
            if PROVIDER_REGISTRY.get(text_cfg.preset, {}).get("needs_api_key"):
                api_key = keychain.get_provider_key(text_cfg.preset) or ""
            try:
                text_em = build_embedder(text_cfg.preset, {
                    "api_key": api_key,
                    "base_url": text_cfg.base_url,
                    "model": text_cfg.model,
                    "dimensions": text_cfg.dimensions,
                })
            except Exception as e:
                log.warning("text embedder init failed: %s", e)
                text_em = None

        vis_cfg = cfg.embedding.vision
        if vis_cfg and vis_cfg.preset:
            api_key = ""
            if VISION_PROVIDER_REGISTRY.get(vis_cfg.preset, {}).get("needs_api_key"):
                api_key = keychain.get_provider_key(vis_cfg.preset) or ""
            try:
                vision_em = build_vision_embedder(vis_cfg.preset, {
                    "api_key": api_key,
                    "base_url": vis_cfg.base_url,
                    "model": vis_cfg.model,
                    "dimensions": vis_cfg.dimensions,
                })
            except Exception as e:
                log.warning("vision embedder init failed: %s", e)
                vision_em = None

        return cls(text=text_em, vision=vision_em)
