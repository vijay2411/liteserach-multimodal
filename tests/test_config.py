from pathlib import Path
import textwrap
from semanticsd import config


def test_defaults_when_no_file(tmp_app_support):
    cfg = config.load()
    assert cfg.daemon.http_host == "127.0.0.1"
    assert cfg.daemon.http_port == 47600
    assert cfg.search.default_mode == "semantic"
    assert cfg.power.mode == "active"


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
