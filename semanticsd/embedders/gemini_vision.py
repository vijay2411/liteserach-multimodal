"""GeminiVisionEmbedder — Gemini Embedding 2 with inline_data image parts."""
from __future__ import annotations
import base64
import json
import urllib.request
from typing import Literal
from semanticsd.embedders.vision_base import VisionEmbedder
from semanticsd.embedders.base import EmbedResult

DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_DIM = 3072
ENDPOINT_FMT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:embedContent?key={key}"
)


def _detect_mime(data: bytes) -> str:
    if data[:8].startswith(b"\x89PNG\r\n"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


class GeminiVisionEmbedder(VisionEmbedder):
    provider_id = "gemini"
    cost_per_million_image_tokens_usd = 0.15  # placeholder

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        timeout_s: float = 60.0,
    ):
        if not api_key:
            raise ValueError("GeminiVisionEmbedder requires api_key")
        self.api_key = api_key
        self.model_id = model
        self.dim = dim
        self.timeout_s = timeout_s

    def embed_images(
        self,
        images: list[bytes],
        kind: Literal["doc", "query"] = "doc",
    ) -> EmbedResult:
        vectors: list[list[float]] = []
        url = ENDPOINT_FMT.format(model=self.model_id, key=self.api_key)
        for img in images:
            body = json.dumps({
                "model": f"models/{self.model_id}",
                "content": {"parts": [{
                    "inline_data": {
                        "mime_type": _detect_mime(img),
                        "data": base64.b64encode(img).decode("ascii"),
                    }
                }]},
            }).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read())
            values = data.get("embedding", {}).get("values")
            if not values:
                raise RuntimeError(f"Gemini vision embed returned no values: {data}")
            vectors.append([float(x) for x in values])
        return EmbedResult(
            vectors=vectors,
            input_tokens=self.estimate_image_tokens(images),
        )

    def health_check(self) -> tuple[bool, str]:
        try:
            tiny_png = base64.b64decode(
                b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            )
            self.embed_images([tiny_png])
            return (True, f"gemini vision {self.model_id} ok")
        except Exception as e:
            return (False, f"gemini vision unreachable: {e}")

    def estimate_image_tokens(self, images: list[bytes]) -> int:
        return 258 * len(images)
