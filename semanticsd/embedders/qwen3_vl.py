"""LocalQwen3VisionEmbedder — Qwen3-VL-Embedding via sentence-transformers.

Runs locally on Apple Silicon via MPS, fp16. ~5 GB RAM for the 2B model.
Output is 2048-d (L2-normalized).
"""
from __future__ import annotations
import io
import logging
from typing import Literal
from semanticsd.embedders.vision_base import VisionEmbedder
from semanticsd.embedders.base import EmbedResult

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_DIM = 2048


class LocalQwen3VisionEmbedder(VisionEmbedder):
    """Local vision embedder using Qwen3-VL-Embedding via sentence-transformers.

    Lazy-loads the model on first `embed_images()` call (~25s cold load,
    instant once cached). All inference runs on MPS at fp16 by default.
    """

    provider_id = "qwen3_vl_local"
    cost_per_million_image_tokens_usd = 0.0

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        device: str = "mps",
        torch_dtype: str = "float16",
    ):
        self.model_id = model
        self.dim = dim
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("loading %s on %s (%s)...", self.model_id, self.device, self.torch_dtype)
            self._model = SentenceTransformer(
                self.model_id,
                device=self.device,
                model_kwargs={"torch_dtype": self.torch_dtype},
            )
        return self._model

    def embed_images(
        self,
        images: list[bytes],
        kind: Literal["doc", "query"] = "doc",
    ) -> EmbedResult:
        from PIL import Image
        model = self._ensure_model()
        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        vectors = model.encode(pil_images, normalize_embeddings=True)
        # sentence-transformers returns numpy ndarray; convert to list[list[float]]
        return EmbedResult(
            vectors=[[float(x) for x in v] for v in vectors],
            input_tokens=self.estimate_image_tokens(images),
        )

    def health_check(self) -> tuple[bool, str]:
        try:
            from PIL import Image
            img = Image.new("RGB", (32, 32), color=(255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            self.embed_images([buf.getvalue()])
            return (True, f"local Qwen3-VL ({self.model_id}) ok on {self.device}")
        except Exception as e:
            return (False, f"local Qwen3-VL load failed: {e}")

    def estimate_image_tokens(self, images: list[bytes]) -> int:
        return 256 * len(images)
