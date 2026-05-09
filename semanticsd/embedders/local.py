"""LocalEmbedder — sentence-transformers in-process. Default zero-config provider.

The model is downloaded lazily on first embed() call to ~/Library/Application
Support/semanticsd/models/. No API key, no network at startup, costs $0.
"""
from __future__ import annotations
from typing import Literal
from sentence_transformers import SentenceTransformer
from semanticsd import paths
from semanticsd.embedders.base import Embedder, EmbedResult


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384


class LocalEmbedder(Embedder):
    provider_id = "local"
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0

    def __init__(self, model_id: str = DEFAULT_MODEL, dim: int = DEFAULT_DIM):
        self.model_id = model_id
        self.dim = dim
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            paths.ensure_dirs()
            self._model = SentenceTransformer(
                self.model_id,
                cache_folder=str(paths.models_dir()),
            )
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self.dim:
                self.dim = actual_dim
        return self._model

    def embed(
        self,
        texts: list[str],
        kind: Literal["doc", "query"],
    ) -> EmbedResult:
        model = self._ensure_model()
        arr = model.encode(texts, normalize_embeddings=True)
        # arr is np.ndarray; convert to plain Python list of lists.
        vectors = [v.tolist() for v in arr]
        return EmbedResult(
            vectors=vectors,
            input_tokens=self.estimate_tokens(texts),
        )

    def health_check(self) -> tuple[bool, str]:
        try:
            self._ensure_model()
            return (True, f"local model {self.model_id} loaded")
        except Exception as e:
            return (False, f"local model failed: {e}")

    def estimate_tokens(self, texts: list[str]) -> int:
        return sum(len(t) // 4 for t in texts)
