#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
LAUNCHD_LABEL="com.semanticsd"
PLIST_PATH="$HOME/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"

case "${1:-}" in
  --uninstall)
    echo "Uninstalling SemanticsD…"
    if [[ -f "$PLIST_PATH" ]]; then
      launchctl bootout "gui/$(id -u)" "$PLIST_PATH" || true
      rm -f "$PLIST_PATH"
      echo "  removed $PLIST_PATH"
    fi
    if [[ "${2:-}" == "--purge" ]]; then
      rm -rf "$HOME/Library/Application Support/semanticsd"
      rm -rf "$HOME/Library/Logs/semanticsd"
      echo "  purged Application Support + Logs"
    fi
    echo "Done."
    exit 0
    ;;
esac

# 1. macOS check
if [[ "$(uname)" != "Darwin" ]]; then
  echo "ERROR: SemanticsD is macOS-only." >&2
  exit 1
fi

# 2. Python 3.11+
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found." >&2
  exit 1
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$(printf '%s\n' "3.11" "$PYV" | sort -V | head -n1)" != "3.11" ]]; then
  echo "ERROR: Python 3.11+ required (have $PYV)." >&2
  exit 1
fi

# 3. venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR…"
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
"$VENV_DIR/bin/pip" install -r "$REPO_ROOT/requirements.txt"
"$VENV_DIR/bin/pip" install -e "$REPO_ROOT"

# 4. install launchd agent + token + config
"$VENV_DIR/bin/python" -m semanticsd install

# 5. wait for /v1/health
PORT=$("$VENV_DIR/bin/python" -c "from semanticsd import config; print(config.load().daemon.http_port)")
TOKEN=$("$VENV_DIR/bin/python" -m semanticsd token print)
echo "Waiting for daemon on :$PORT…"
for i in {1..30}; do
  if curl -s -H "X-Auth-Token: $TOKEN" "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
    echo "  ready."
    break
  fi
  sleep 1
done

echo
echo "SemanticsD is installed."
echo "  Try: ssearch --status"
echo "  Logs: ~/Library/Logs/semanticsd/"
echo "  Config: ~/Library/Application Support/semanticsd/config.toml"
