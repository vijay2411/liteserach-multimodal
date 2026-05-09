"""EmbedderRouter — per-modality routing."""
import pytest
from pathlib import Path
from semanticsd.embedders.router import EmbedderRouter
from semanticsd import config


def test_router_text_only(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("""
[embedding.text]
preset = "local"
model = "BAAI/bge-small-en-v1.5"
""")
    cfg = config.load(p)
    router = EmbedderRouter.from_config(cfg)
    assert router.text is not None
    assert router.vision is None
    assert router.get("text") is router.text
    assert router.get("vision") is None


def test_router_with_vision(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "semanticsd.keychain.get_provider_key",
        lambda preset: "fake-gemini-key" if preset == "gemini" else None,
    )
    p = tmp_path / "c.toml"
    p.write_text("""
[embedding.text]
preset = "local"
model = "BAAI/bge-small-en-v1.5"

[embedding.vision]
preset = "gemini"
model = "gemini-embedding-2"
""")
    cfg = config.load(p)
    router = EmbedderRouter.from_config(cfg)
    assert router.text is not None
    assert router.vision is not None
    assert router.vision.provider_id == "gemini"


def test_router_unknown_modality_raises():
    r = EmbedderRouter()
    with pytest.raises(ValueError):
        r.get("audio")  # type: ignore


def test_router_vision_init_failure_is_silent(tmp_path, monkeypatch):
    """If vision API key is missing, router still loads with vision=None."""
    monkeypatch.setattr(
        "semanticsd.keychain.get_provider_key",
        lambda preset: None,  # no key for gemini
    )
    p = tmp_path / "c.toml"
    p.write_text("""
[embedding.text]
preset = "local"

[embedding.vision]
preset = "gemini"
""")
    cfg = config.load(p)
    router = EmbedderRouter.from_config(cfg)
    assert router.text is not None
    assert router.vision is None
