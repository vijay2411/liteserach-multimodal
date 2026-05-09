"""Render and install the launchd plist for SemanticsD."""
from __future__ import annotations
import plistlib
from pathlib import Path
from semanticsd import paths


LABEL = paths.LAUNCHD_LABEL


def render_plist(python_executable: str, package_dir: str) -> str:
    paths.ensure_dirs()
    plist = {
        "Label": LABEL,
        "ProgramArguments": [python_executable, "-m", "semanticsd", "serve"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "WorkingDirectory": package_dir,
        "StandardOutPath": str(paths.logs_dir() / "semanticsd.out.log"),
        "StandardErrorPath": str(paths.logs_dir() / "semanticsd.err.log"),
        "EnvironmentVariables": {
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        },
        "ProcessType": "Background",
    }
    return plistlib.dumps(plist).decode()


def write_plist(target: Path, python_executable: str, package_dir: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_plist(python_executable, package_dir))
