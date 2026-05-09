"""OpenAICompatibleEmbedder — generic /v1/embeddings client.

Covers OpenAI, Ollama (with its /v1 compat endpoint), LM Studio, vLLM, llama.cpp
server, OpenRouter, Together, Groq, Fireworks, TEI, and any self-hosted server
that speaks the OpenAI embeddings API.

Constructor: (base_url, api_key, model, dim, dimensions=None).
"""
from __future__ import annotations
from typing import Literal
from openai import OpenAI
from semanticsd.embedders.base import Embedder, EmbedResult


class OpenAICompatibleEmbedder(Embedder):
    provider_id = "openai_compatible"
    supports_kind = False  # generic OpenAI-compat servers ignore input_type
    cost_per_million_input_tokens_usd = 0.0  # subclasses override

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        dimensions: int = 0,
    ):
        self.base_url = base_url
        self.api_key = api_key or "not-needed"  # local servers reject empty strings
        self.model_id = model
        self.dim = dim
        self.dimensions = dimensions
        self._client = OpenAI(base_url=base_url, api_key=self.api_key)

    def embed(
        self,
        texts: list[str],
        kind: Literal["doc", "query"],
    ) -> EmbedResult:
        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self._client.embeddings.create(
            model=self.model_id, input=texts, **kwargs
        )
        vectors = [list(d.embedding) for d in resp.data]
        # Some servers (Ollama) don't populate usage; fall back to heuristic.
        usage_tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
        input_tokens = (
            int(usage_tokens) if usage_tokens is not None else self.estimate_tokens(texts)
        )
        return EmbedResult(vectors=vectors, input_tokens=input_tokens)

    def health_check(self) -> tuple[bool, str]:
        try:
            self.embed(["ping"], kind="query")
            return (True, f"{self.base_url} reachable, model {self.model_id} ok")
        except Exception as e:
            return (False, f"{self.base_url} unreachable: {e}")

    def estimate_tokens(self, texts: list[str]) -> int:
        return sum(len(t) // 4 for t in texts)
