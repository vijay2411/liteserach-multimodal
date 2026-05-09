from pathlib import Path
from semanticsd import paths


def test_app_support_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SEMANTICSD_HOME", raising=False)
    p = paths.app_support()
    assert p == Path.home() / "Library" / "Application Support" / "semanticsd"


def test_app_support_override(tmp_app_support):
    assert paths.app_support() == tmp_app_support


def test_logs_dir(monkeypatch):
    monkeypatch.delenv("SEMANTICSD_HOME", raising=False)
    assert paths.logs_dir() == Path.home() / "Library" / "Logs" / "semanticsd"


def test_db_path_under_app_support(tmp_app_support):
    assert paths.db_path() == tmp_app_support / "index.db"


def test_config_path(tmp_app_support):
    assert paths.config_path() == tmp_app_support / "config.toml"


def test_launch_agent_plist_path(monkeypatch):
    monkeypatch.delenv("SEMANTICSD_HOME", raising=False)
    assert paths.launch_agent_plist() == Path.home() / "Library" / "LaunchAgents" / "com.semanticsd.plist"


def test_logs_dir_override(tmp_app_support):
    assert paths.logs_dir() == tmp_app_support / "logs"


def test_ensure_dirs_creates(tmp_app_support):
    paths.ensure_dirs()
    assert tmp_app_support.exists()
    assert (tmp_app_support / "logs").exists()
    assert (tmp_app_support / "models").exists()
    assert (tmp_app_support / "LaunchAgents").exists()
