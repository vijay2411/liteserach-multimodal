"""OpenAIEmbedder — OpenAI's hosted embeddings API.
Thin subclass of OpenAICompatibleEmbedder with cost table and sane defaults.
"""
from __future__ import annotations
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder


# USD per 1M input tokens, as of 2026.
COST_PER_MILLION = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

DIM_BY_MODEL = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedder(OpenAICompatibleEmbedder):
    provider_id = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = 0,
    ):
        dim = dimensions if dimensions else DIM_BY_MODEL.get(model, 0)
        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=model,
            dim=dim,
            dimensions=dimensions,
        )
        self.cost_per_million_input_tokens_usd = COST_PER_MILLION.get(model, 0.0)
