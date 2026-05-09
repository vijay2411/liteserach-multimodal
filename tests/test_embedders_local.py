"""LocalEmbedder unit tests — sentence-transformers is mocked.
The real-model integration test lives in tests/test_e2e_vec.py.
"""
import numpy as np
import pytest
from semanticsd.embedders.base import EmbedResult
from semanticsd.embedders.local import LocalEmbedder


class FakeST:
    """Fake SentenceTransformer that returns deterministic 384-d vectors."""
    def __init__(self, model_name, device=None, cache_folder=None):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True):
        # Deterministic: each text -> a 384-vec scaled by len(text).
        return np.array(
            [[float(len(t) % 7) / 10.0] * 384 for t in texts],
            dtype=np.float32,
        )

    def get_sentence_embedding_dimension(self):
        return 384


@pytest.fixture
def patched_st(monkeypatch):
    monkeypatch.setattr(
        "semanticsd.embedders.local.SentenceTransformer",
        FakeST,
    )


def test_local_embedder_metadata(patched_st):
    e = LocalEmbedder()
    assert e.provider_id == "local"
    assert e.model_id == "BAAI/bge-small-en-v1.5"
    assert e.dim == 384
    assert e.supports_kind is False
    assert e.cost_per_million_input_tokens_usd == 0.0


def test_local_embed_returns_correct_shape(patched_st):
    e = LocalEmbedder()
    out = e.embed(["alpha", "bravo charlie"], kind="doc")
    assert isinstance(out, EmbedResult)
    assert len(out.vectors) == 2
    assert all(len(v) == 384 for v in out.vectors)
    assert out.input_tokens > 0


def test_local_health_check_ok(patched_st):
    e = LocalEmbedder()
    ok, msg = e.health_check()
    assert ok is True


def test_local_estimate_tokens(patched_st):
    e = LocalEmbedder()
    n = e.estimate_tokens(["abcdefgh", "ijkl"])
    # Heuristic: chars / 4 — "abcdefgh"=2, "ijkl"=1 => 3
    assert n == 3


def test_local_lazy_loads_model(patched_st):
    """Model isn't loaded until first embed() call."""
    e = LocalEmbedder()
    assert e._model is None
    e.embed(["x"], kind="doc")
    assert e._model is not None


def test_custom_model_id(patched_st):
    e = LocalEmbedder(model_id="custom-model")
    assert e.model_id == "custom-model"
