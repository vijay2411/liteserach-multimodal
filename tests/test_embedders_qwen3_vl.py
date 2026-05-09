"""LocalQwen3VisionEmbedder — sentence-transformers + Qwen3-VL-Embedding."""
import io
import pytest
from unittest.mock import MagicMock
from semanticsd.embedders.qwen3_vl import LocalQwen3VisionEmbedder


def test_provider_id_and_dim():
    e = LocalQwen3VisionEmbedder()
    assert e.provider_id == "qwen3_vl_local"
    assert e.model_id == "Qwen/Qwen3-VL-Embedding-2B"
    assert e.dim == 2048
    assert e.cost_per_million_image_tokens_usd == 0.0


def test_lazy_model_load(monkeypatch):
    """Model should not load until embed_images is called."""
    e = LocalQwen3VisionEmbedder()
    assert e._model is None

    fake_model = MagicMock()
    fake_model.encode.return_value = [[0.1] * 2048]
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *a, **k: fake_model,
    )
    from PIL import Image
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, "PNG")

    out = e.embed_images([buf.getvalue()])
    assert e._model is fake_model  # cached after first call
    assert len(out.vectors) == 1
    assert len(out.vectors[0]) == 2048


def test_embed_images_decodes_bytes_to_pil(monkeypatch):
    """Bytes -> PIL conversion happens before model.encode."""
    e = LocalQwen3VisionEmbedder()
    captured = {}

    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda imgs, **kw: (
        captured.update({"images": imgs, "kwargs": kw}) or
        [[0.1] * 2048 for _ in imgs]
    )
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *a, **k: fake_model,
    )

    from PIL import Image
    img = Image.new("RGB", (16, 16), color="blue")
    buf = io.BytesIO()
    img.save(buf, "PNG")

    e.embed_images([buf.getvalue(), buf.getvalue()])
    assert len(captured["images"]) == 2
    # Each one should be a PIL Image in RGB
    assert all(getattr(im, "mode", None) == "RGB" for im in captured["images"])
    # normalize_embeddings is set
    assert captured["kwargs"].get("normalize_embeddings") is True


def test_estimate_image_tokens_scales_with_count():
    e = LocalQwen3VisionEmbedder()
    assert e.estimate_image_tokens([b"a"] * 5) == 256 * 5
