"""Tests for the Embedder ABC and EmbedResult model."""
from semanticsd.embedders.base import Embedder, EmbedResult


def test_embed_result_has_required_fields():
    r = EmbedResult(vectors=[[0.1, 0.2]], input_tokens=4)
    assert r.vectors == [[0.1, 0.2]]
    assert r.input_tokens == 4
    assert r.output_tokens == 0
    assert r.raw_response is None


def test_embedder_is_abstract():
    """Abstract methods cannot be instantiated directly."""
    try:
        Embedder()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("Embedder should be abstract")


def test_concrete_subclass_can_be_instantiated():
    class Stub(Embedder):
        provider_id = "stub"
        model_id = "stub-1"
        dim = 4
        supports_kind = False
        cost_per_million_input_tokens_usd = 0.0

        def embed(self, texts, kind):
            return EmbedResult(vectors=[[0.0] * self.dim for _ in texts], input_tokens=0)

        def health_check(self):
            return (True, "ok")

        def estimate_tokens(self, texts):
            return sum(len(t) // 4 for t in texts)

    s = Stub()
    out = s.embed(["hello", "world"], kind="doc")
    assert len(out.vectors) == 2
    assert s.health_check() == (True, "ok")
    assert s.estimate_tokens(["abcdefgh"]) == 2
