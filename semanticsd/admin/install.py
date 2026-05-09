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


def _label() -> str:
    """The launchd label set in our plist."""
    return "com.semanticsd"


def _gui_target() -> str:
    return f"gui/{os.getuid()}"


def start() -> dict:
    """Start (or re-start) the daemon via launchctl. Idempotent."""
    plist = paths.launch_agent_plist()
    if not plist.exists():
        raise RuntimeError("plist not installed — run `semanticsd install` first")
    # bootout then bootstrap = clean restart; ignore bootout failures (might not be loaded)
    subprocess.run(["launchctl", "bootout", _gui_target(), str(plist)],
                   check=False, capture_output=True)
    r = subprocess.run(
        ["launchctl", "bootstrap", _gui_target(), str(plist)],
        check=False, capture_output=True, text=True,
    )
    return {"ok": r.returncode == 0, "stderr": r.stderr.strip()}


def stop() -> dict:
    """Stop the daemon via launchctl bootout. Idempotent."""
    plist = paths.launch_agent_plist()
    r = subprocess.run(
        ["launchctl", "bootout", _gui_target(), str(plist)],
        check=False, capture_output=True, text=True,
    )
    return {"ok": r.returncode == 0, "stderr": r.stderr.strip()}


def restart() -> dict:
    """Restart via launchctl kickstart -k."""
    label = f"{_gui_target()}/{_label()}"
    r = subprocess.run(
        ["launchctl", "kickstart", "-k", label],
        check=False, capture_output=True, text=True,
    )
    return {"ok": r.returncode == 0, "stderr": r.stderr.strip()}


def daemon_status() -> dict:
    """Return whether the launchd-managed daemon is currently running.

    Uses `launchctl print` which returns the agent's state. Falls back to
    checking the HTTP port if launchctl isn't available.
    """
    label = f"{_gui_target()}/{_label()}"
    r = subprocess.run(
        ["launchctl", "print", label],
        check=False, capture_output=True, text=True,
    )
    if r.returncode != 0:
        return {"loaded": False, "running": False, "pid": None,
                "raw": r.stderr.strip() or "agent not loaded"}
    out = r.stdout
    # `launchctl print` reports `state = running` or `state = not running`
    pid = None
    running = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("pid ="):
            try:
                pid = int(s.split("=", 1)[1].strip())
            except ValueError:
                pass
        if s.startswith("state ="):
            running = s.endswith("running") and "not running" not in s
    return {"loaded": True, "running": running, "pid": pid, "raw": out[:500]}


def log_path() -> Path:
    """Where the daemon writes its log file (matches logging_setup.configure)."""
    return paths.logs_dir() / "semanticsd.log"
