from pathlib import Path
import textwrap
from semanticsd import config


def test_defaults_when_no_file(tmp_app_support):
    cfg = config.load()
    assert cfg.daemon.http_host == "127.0.0.1"
    assert cfg.daemon.http_port == 47600
    assert cfg.search.default_mode == "semantic"
    assert cfg.power.mode == "active"
    # Plan 2: no surprise auto-indexing on a fresh install.
    assert cfg.watch.directories == []


def test_loads_overrides_from_file(tmp_app_support):
    (tmp_app_support / "config.toml").write_text(textwrap.dedent("""
        [daemon]
        http_port = 9999
        log_level = "debug"

        [search]
        default_mode = "filename"
    """))
    cfg = config.load()
    assert cfg.daemon.http_port == 9999
    assert cfg.daemon.log_level == "debug"
    assert cfg.search.default_mode == "filename"
    # Untouched defaults preserved:
    assert cfg.daemon.http_host == "127.0.0.1"


def test_invalid_mode_rejected(tmp_app_support):
    (tmp_app_support / "config.toml").write_text("[search]\ndefault_mode = 'bogus'\n")
    try:
        config.load()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_write_default(tmp_app_support):
    config.write_default()
    text = (tmp_app_support / "config.toml").read_text()
    assert "[daemon]" in text
    assert "http_port = 47600" in text


def test_default_toml_has_empty_directories(tmp_app_support):
    config.write_default()
    text = (tmp_app_support / "config.toml").read_text()
    assert "directories = []" in text


def test_split_embedding_sections(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(textwrap.dedent("""
        [embedding.text]
        preset = "ollama"
        model = "embeddinggemma"
        base_url = "http://localhost:11434/v1"

        [embedding.vision]
        preset = "gemini"
        model = "gemini-embedding-2"
    """))
    cfg = config.load(p)
    assert cfg.embedding.text.preset == "ollama"
    assert cfg.embedding.text.model == "embeddinggemma"
    assert cfg.embedding.vision is not None
    assert cfg.embedding.vision.preset == "gemini"


def test_legacy_flat_embedding_migrated(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(textwrap.dedent("""
        [embedding]
        preset = "local"
        model = "BAAI/bge-small-en-v1.5"
    """))
    cfg = config.load(p)
    assert cfg.embedding.text.preset == "local"
    assert cfg.embedding.vision is None


def test_vision_optional(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(textwrap.dedent("""
        [embedding.text]
        preset = "ollama"
        model = "embeddinggemma"
    """))
    cfg = config.load(p)
    assert cfg.embedding.vision is None
    assert cfg.embedding.text.preset == "ollama"


def test_default_toml_uses_ollama_text_section(tmp_app_support):
    config.write_default()
    text = (tmp_app_support / "config.toml").read_text()
    assert "[embedding.text]" in text
    assert "embeddinggemma" in text
