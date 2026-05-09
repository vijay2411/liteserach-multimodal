"""VisionEmbedder ABC contract."""
import pytest
from semanticsd.embedders.vision_base import VisionEmbedder
from semanticsd.embedders.base import EmbedResult


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        VisionEmbedder()


def test_concrete_subclass_works():
    class Fake(VisionEmbedder):
        provider_id = "fake"
        model_id = "fake-vision"
        dim = 64

        def embed_images(self, images, kind="doc"):
            return EmbedResult(
                vectors=[[0.1] * 64 for _ in images],
                input_tokens=len(images),
            )

        def health_check(self):
            return (True, "ok")

        def estimate_image_tokens(self, images):
            return len(images)

    e = Fake()
    out = e.embed_images([b"\x89PNG\r\n", b"\xff\xd8\xff"])
    assert len(out.vectors) == 2
    assert len(out.vectors[0]) == 64
