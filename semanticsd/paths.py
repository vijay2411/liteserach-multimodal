"""macOS-conventional paths for SemanticsD."""
from __future__ import annotations
import os
from pathlib import Path


LAUNCHD_LABEL = "com.semanticsd"


def app_support() -> Path:
    """Application Support directory. Honors $SEMANTICSD_HOME for tests."""
    override = os.environ.get("SEMANTICSD_HOME")
    if override:
        return Path(override)
    return Path.home() / "Library" / "Application Support" / "semanticsd"


def logs_dir() -> Path:
    override = os.environ.get("SEMANTICSD_HOME")
    if override:
        return Path(override) / "logs"
    return Path.home() / "Library" / "Logs" / "semanticsd"


def db_path() -> Path:
    return app_support() / "index.db"


def config_path() -> Path:
    return app_support() / "config.toml"


def models_dir() -> Path:
    return app_support() / "models"


def launch_agent_plist() -> Path:
    override = os.environ.get("SEMANTICSD_HOME")
    if override:
        return Path(override) / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def ensure_dirs() -> None:
    """Create all directories the daemon needs. Idempotent."""
    for d in (app_support(), logs_dir(), models_dir(), launch_agent_plist().parent):
        d.mkdir(parents=True, exist_ok=True)
