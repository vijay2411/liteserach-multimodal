"""Abstract Embedder base + result model."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel


class EmbedResult(BaseModel):
    vectors: list[list[float]]
    input_tokens: int
    output_tokens: int = 0
    raw_response: dict | None = None


class Embedder(ABC):
    """Pluggable embedder. Subclass and register in registry.py.

    Required class attributes (set on the subclass):
      provider_id: str  — preset key, e.g. "openai" / "local" / "ollama"
      model_id: str
      dim: int
      supports_kind: bool  — True if the provider distinguishes doc vs query
      cost_per_million_input_tokens_usd: float
    """

    provider_id: str = ""
    model_id: str = ""
    dim: int = 0
    supports_kind: bool = False
    cost_per_million_input_tokens_usd: float = 0.0

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        kind: Literal["doc", "query"],
    ) -> EmbedResult: ...

    @abstractmethod
    def health_check(self) -> tuple[bool, str]: ...

    @abstractmethod
    def estimate_tokens(self, texts: list[str]) -> int: ...
