"""OllamaEmbedder — Ollama via its OpenAI-compat /v1 endpoint.
No API key, defaults to localhost:11434. Cost = 0 (local).
"""
from __future__ import annotations
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder


DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIM = 768


class OllamaEmbedder(OpenAICompatibleEmbedder):
    provider_id = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dim: int = DEFAULT_DIM,
    ):
        super().__init__(
            base_url=base_url,
            api_key="ollama",  # placeholder; Ollama ignores it
            model=model,
            dim=dim,
        )
        self.cost_per_million_input_tokens_usd = 0.0
