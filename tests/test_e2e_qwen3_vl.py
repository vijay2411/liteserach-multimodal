"""Real-MPS smoke test for LocalQwen3VisionEmbedder.

Loads ~5GB of weights and runs inference on Apple MPS, so it's slow.
Skipped if torch/MPS isn't available or weights aren't cached.
"""
import io
import pytest


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _mps_available(), reason="MPS not available"),
]


def test_local_qwen3_vl_embed_image_real():
    from PIL import Image
    from semanticsd.embedders.qwen3_vl import LocalQwen3VisionEmbedder

    e = LocalQwen3VisionEmbedder()
    img = Image.new("RGB", (128, 128), color=(0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    img_bytes = buf.getvalue()

    out = e.embed_images([img_bytes])
    assert len(out.vectors) == 1
    assert len(out.vectors[0]) == 2048

    # Re-running same input should be ~ deterministic (cosine ≈ 1.0)
    out2 = e.embed_images([img_bytes])
    v1 = out.vectors[0]
    v2 = out2.vectors[0]
    dot = sum(a * b for a, b in zip(v1, v2))
    # Vectors are L2-normalized, so dot ≈ cosine similarity
    assert dot > 0.99, f"deterministic re-encode should give cosine ~1.0, got {dot}"


def test_local_qwen3_vl_health_check():
    from semanticsd.embedders.qwen3_vl import LocalQwen3VisionEmbedder
    e = LocalQwen3VisionEmbedder()
    ok, msg = e.health_check()
    assert ok, msg
    assert "ok" in msg.lower()
