"""GeminiTextEmbedder — Google Gemini Embedding 2 via REST API.

Uses urllib (stdlib). The embedContent endpoint takes one input per
request, so embed(texts) loops sequentially.
"""
from __future__ import annotations
import json
import logging
import urllib.request
from typing import Literal
from semanticsd.embedders.base import Embedder, EmbedResult

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_DIM = 3072
ENDPOINT_FMT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:embedContent?key={key}"
)


class GeminiTextEmbedder(Embedder):
    provider_id = "gemini"
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.15  # placeholder; refresh from billing

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        timeout_s: float = 30.0,
    ):
        if not api_key:
            raise ValueError("GeminiTextEmbedder requires api_key")
        self.api_key = api_key
        self.model_id = model
        self.dim = dim
        self.timeout_s = timeout_s

    def embed(
        self,
        texts: list[str],
        kind: Literal["doc", "query"] = "doc",
    ) -> EmbedResult:
        vectors: list[list[float]] = []
        url = ENDPOINT_FMT.format(model=self.model_id, key=self.api_key)
        for t in texts:
            body = json.dumps({
                "model": f"models/{self.model_id}",
                "content": {"parts": [{"text": t}]},
            }).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read())
            values = data.get("embedding", {}).get("values")
            if not values:
                raise RuntimeError(f"Gemini embed returned no values: {data}")
            vectors.append([float(x) for x in values])
        return EmbedResult(
            vectors=vectors,
            input_tokens=self.estimate_tokens(texts),
        )

    def health_check(self) -> tuple[bool, str]:
        try:
            self.embed(["ping"], kind="query")
            return (True, f"gemini {self.model_id} ok")
        except Exception as e:
            return (False, f"gemini unreachable: {e}")

    def estimate_tokens(self, texts: list[str]) -> int:
        return sum(len(t) // 4 for t in texts)
