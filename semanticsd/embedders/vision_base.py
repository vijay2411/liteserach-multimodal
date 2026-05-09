"""Abstract VisionEmbedder base — embeds raw image bytes.

Parallel to Embedder (text). Each ABC has its own provider files;
a provider supporting both modalities ships as two classes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal
from semanticsd.embedders.base import EmbedResult


class VisionEmbedder(ABC):
    """Pluggable vision embedder. Subclass and register in registry.py.

    Required class attributes:
      provider_id: str  — preset key, e.g. "gemini" / "jina"
      model_id: str
      dim: int
      cost_per_million_image_tokens_usd: float
    """

    provider_id: str = ""
    model_id: str = ""
    dim: int = 0
    cost_per_million_image_tokens_usd: float = 0.0

    @abstractmethod
    def embed_images(
        self,
        images: list[bytes],
        kind: Literal["doc", "query"] = "doc",
    ) -> EmbedResult:
        """Embed raw image bytes (PNG/JPEG/WebP). Provider handles encoding."""
        ...

    @abstractmethod
    def health_check(self) -> tuple[bool, str]: ...

    @abstractmethod
    def estimate_image_tokens(self, images: list[bytes]) -> int: ...
