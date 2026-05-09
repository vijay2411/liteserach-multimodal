"""Install / uninstall SemanticsD as a launchd user agent."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from semanticsd import paths, config, keychain
from semanticsd.admin import launchd
from semanticsd.db import connection, migrations


def install() -> dict:
    """Idempotent install. Returns a dict of what was done."""
    actions = []

    paths.ensure_dirs()
    actions.append(f"ensured dirs at {paths.app_support()}")

    cfg_path = config.write_default()
    actions.append(f"config at {cfg_path}")

    token = keychain.generate_or_get_auth_token()
    actions.append("auth token in Keychain (service=semanticsd, account=api_token)")

    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    actions.append(f"db migrated at {paths.db_path()}")

    plist_path = paths.launch_agent_plist()
    package_dir = str(Path(__file__).resolve().parent.parent.parent)
    launchd.write_plist(plist_path, python_executable=sys.executable, package_dir=package_dir)
    actions.append(f"plist at {plist_path}")

    # Bootstrap into the user's launchd domain. Bootout first to make idempotent.
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(plist_path)],
        check=False, capture_output=True,
    )
    result = subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        actions.append(f"launchctl bootstrap failed: {result.stderr.strip()}")
    else:
        actions.append("launchctl bootstrap ok")

    return {"actions": actions, "token_hint": "stored in Keychain", "plist": str(plist_path)}


def uninstall() -> dict:
    """Stop and remove launchd agent. Does not delete index/config."""
    actions = []
    plist_path = paths.launch_agent_plist()
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(plist_path)],
        check=False, capture_output=True,
    )
    actions.append("launchctl bootout")
    if plist_path.exists():
        plist_path.unlink()
        actions.append(f"removed {plist_path}")
    return {"actions": actions}


def print_token() -> str:
    tok = keychain.get_auth_token()
    if not tok:
        raise RuntimeError("no token in Keychain — run `semanticsd install` first")
    return tok
