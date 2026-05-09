# 📘 SemanticsD — Usage Guide

Everything you need to install, configure, and run SemanticsD. The
[README](README.md) covers what it is and why; this is the operational
reference.

---

## 🚀 Install

### 1. Clone and install Python deps

```bash
git clone git@github.com:vijay2411/liteserach-multimodal.git
cd liteserach-multimodal
make install      # creates .venv, installs everything
```

### 2. Verify your Python build supports sqlite-vec

```bash
.venv/bin/python -c "
import sqlite3
sqlite3.connect(':memory:').enable_load_extension(True)
print('OK — your Python build supports loadable extensions')
"
```

If this errors with `AttributeError: 'Connection' object has no attribute 'enable_load_extension'`, your Python build is missing the feature. **Pyenv builds usually do**. Install Homebrew Python instead:

```bash
brew install python@3.13
rm -rf .venv
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```

### 3. Configure embedders

Pick the path that matches what you have. You can change later.

#### 🦙 Local text via Ollama (recommended default)

```bash
brew install ollama
ollama serve &
ollama pull embeddinggemma          # 768-d, 600MB
```

#### ☁️ Cloud text via OpenAI / Gemini

```bash
.venv/bin/python -c "
from semanticsd import keychain
keychain.set_provider_key('openai', 'YOUR_OPENAI_KEY')   # or 'gemini'
"
```

#### 🖼️ Vision (optional, for cross-modal text↔image search)

**Cloud — Gemini Embedding 2 (cross-modal text+image in one model):**
```bash
.venv/bin/python -c "
from semanticsd import keychain
keychain.set_provider_key('gemini', 'YOUR_GEMINI_KEY')
"
```

**Local — Qwen3-VL-Embedding-2B via MPS** (no key, ~5GB on first use): just configure the preset. Auto-downloads.

### 4. Initialize config + auth token

```bash
.venv/bin/python -m semanticsd install
```

This:
- writes `~/Library/Application Support/semanticsd/config.toml` (default config)
- generates an auth token in macOS Keychain
- registers a launchd plist (`~/Library/LaunchAgents/com.semanticsd.plist`)
- starts the daemon

Print your token any time:
```bash
.venv/bin/python -m semanticsd token print
```

Uninstall (keeps your index):
```bash
.venv/bin/python -m semanticsd uninstall
```

---

## ⚙️ Configure

Edit `~/Library/Application Support/semanticsd/config.toml`. Restart the daemon after edits (`launchctl kickstart -k gui/$UID/com.semanticsd`).

### Minimum viable config

```toml
[watch]
directories = ["/Users/me/Documents"]
ignore_patterns = [".git", "node_modules", ".DS_Store", "*.min.js", "_astro"]
max_file_size_mb = 25

[embedding.text]
preset = "ollama"
model  = "embeddinggemma"
base_url = "http://localhost:11434/v1"
```

### Full config reference

```toml
# Files to watch. Empty = no auto-indexing.
[watch]
directories = ["/Users/me/Documents", "/Users/me/Code"]
ignore_patterns = [".git", "node_modules", "*.min.js"]
max_file_size_mb = 25

# Text embedder (REQUIRED).
[embedding.text]
preset    = "ollama"            # local | ollama | openai | gemini | lmstudio | vllm | openai_compatible | custom
model     = "embeddinggemma"
base_url  = "http://localhost:11434/v1"   # for ollama / openai_compatible
batch_size = 128
dimensions = 0                  # 0 = use model default (or for openai variable-dim models, override)

# Vision embedder (OPTIONAL — enables cross-modal text↔image search).
[embedding.vision]
preset = "gemini"               # gemini | qwen3_vl_local
model  = "gemini-embedding-2"   # or "Qwen/Qwen3-VL-Embedding-2B"
batch_size = 4

# Search defaults.
[search]
default_mode = "semantic"
max_results = 20

[chunking]
strategy = "sliding"
window_tokens = 512
overlap_tokens = 64

[daemon]
http_host = "127.0.0.1"
http_port = 47600
log_level = "info"

# active = FSEvents watcher running; saver = watcher off, periodic full-walk.
[power]
mode = "active"
saver_reindex_interval = "1h"
auto_saver_on_battery = true

[indexing]
max_attempts = 5
worker_concurrency = 2

# Spend caps for paid embedders. 0 = unlimited.
[budget]
monthly_limit_usd = 5.0
warning_threshold = 0.8
```

### Available embedder presets

```bash
ssearch --presets
```

```
local                model=BAAI/bge-small-en-v1.5
ollama               model=embeddinggemma
lmstudio             model=<user-pick>          (needs base URL)
vllm                 model=<user-pick>          (needs base URL)
openai               model=text-embedding-3-small (needs API key)
gemini               model=gemini-embedding-2     (needs API key)
openai_compatible    model=<user-pick>          (needs base URL)
custom               model=<user-pick>          (needs base URL)
```

For vision: `gemini` or `qwen3_vl_local`.

---

## 🏃 Run

```bash
# Foreground — best for trying things and reading logs
.venv/bin/python -m semanticsd serve

# As a launchd agent — auto-starts on login
.venv/bin/python -m semanticsd install      # done already if you ran install above

# Manual launchctl controls
launchctl kickstart -k gui/$UID/com.semanticsd   # restart
launchctl bootout    gui/$UID/com.semanticsd     # stop
launchctl bootstrap  gui/$UID ~/Library/LaunchAgents/com.semanticsd.plist
```

Health check:
```bash
ssearch --status
```

---

## 🔎 Search (CLI)

```bash
# Default = hybrid mode, CWD-scoped. Both flag orderings work.
ssearch "neural network architecture"
ssearch --all "supply chain ideas"
ssearch "tshirt photo" --all              # cross-modal text → image

# Modes (shortcuts)
ssearch --semantic  "..."
ssearch --filename  "..."
ssearch --grep      "..."
ssearch --mode hybrid "..."               # explicit

# Other flags
ssearch "..." --all              # don't restrict to current shell directory
ssearch "..." --no-vision        # skip cross-modal vision
ssearch "..." --chunks           # show every matching chunk (default collapses)
ssearch "..." --limit 50         # top-N
ssearch "..." --json             # machine-readable output
```

### Output anatomy

```
~/Documents/Acct Statement_6706_28042026_00.50.29.pdf  (filename, text, score=1.000)
~/Documents/OpTransactionHistory28-04-2026.pdf         (vision, vision, score=0.992)  (+17 more chunks)
  <image: OpTransactionHistory28-04-2026.pdf page=1>
~/Documents/Untitled.md                                (grep+semantic, text, score=0.74)
  ...framing flexes per audience. Don't change who you are...

-- 3 results in 612ms --
```

| Column | Meaning |
|---|---|
| **path** | Click to open with default OS app (in supporting terminals) |
| **mode label** | which signals contributed via RRF (e.g. `grep+semantic`) |
| **modality** | `text` (chunk content matched) or `vision` (image embedding matched) |
| **score** | 0–1, max-normalized within the result set |
| **+N more chunks** | This file had N additional matching chunks (use `--chunks` to see them) |

### Watcher / power-mode controls

```bash
ssearch watch              # status: mode, watcher_running, dirty_pending, last_sweep_at
ssearch watch --sweep      # force a full re-walk now (useful in saver mode)
ssearch power              # current mode + power source
ssearch power active       # turn FSEvents watcher on
ssearch power saver        # watcher off, only periodic re-walk
```

### Cost tracking

```bash
ssearch usage              # this-month spend grouped by provider/model/op
ssearch usage --today
ssearch usage --all        # all-time
ssearch usage --provider gemini --csv     # filter + pipe to spreadsheet
```

### Switching embedder providers (re-embed)

```bash
# 1. Edit [embedding.text].preset/model in config.toml
# 2. Restart daemon
# 3. Queue re-embed for chunks lacking the new (provider, model, dim)
ssearch reembed text          # only text
ssearch reembed vision        # only vision
ssearch reembed               # both

# Old vectors stay queryable. To roll back: edit config back, restart, run reembed again.
```

### Manual indexing (out-of-band)

```bash
ssearch --index ~/some/folder           # index this dir, drain worker once
```

---

## 🌐 HTTP API

OpenAPI auto-docs: **http://127.0.0.1:47600/docs** (interactive Swagger UI).

All endpoints require `X-Auth-Token: <your-token>` header.

| Endpoint | Purpose |
|---|---|
| `GET  /v1/health` | status, embedders state, budget block |
| `GET  /v1/search` | search (params: `q`, `mode`, `limit`, `all`, `vision`, `collapse`, `cwd`) |
| `POST /v1/index` | manual indexing trigger |
| `GET  /v1/watch` | watcher status |
| `POST /v1/watch/sweep` | force full re-walk |
| `GET  /v1/power` | current mode |
| `POST /v1/power` | `{"mode": "active" \| "saver"}` |
| `GET  /v1/usage` | cost/volume report (params: `since`, `until`, `provider`) |
| `POST /v1/reembed` | `{"modality": "text" \| "vision" \| "all"}` |
| `GET  /v1/presets` | list available embedder providers |
| `POST /v1/embedder/test` | round-trip a preset to verify config |

### Examples

```bash
TOKEN=$(.venv/bin/python -m semanticsd token print)
H="X-Auth-Token: $TOKEN"
B="http://127.0.0.1:47600"

curl -s "$B/v1/health"     -H "$H" | jq
curl -s "$B/v1/search?q=foo&all=true" -H "$H" | jq
curl -s -X POST "$B/v1/power" -H "$H" -H "Content-Type: application/json" \
     -d '{"mode":"saver"}'
```

---

## 🧪 Tests

```bash
make test                                  # 332 unit tests (fast)
.venv/bin/pytest -m slow                   # 56 e2e tests (need Ollama / Gemini)
.venv/bin/pytest tests/test_e2e_stress_battery.py -m slow   # 33-query ground-truth ranking
```

The stress battery is worth running once after install — it spawns the indexer + embedder against a built-in fixture corpus and asserts the right files surface for 33 queries across modalities, languages, and edge cases.

---

## 🛠️ Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `AttributeError: 'Connection' object has no attribute 'enable_load_extension'` | pyenv-built Python | Switch to Homebrew Python (see Install §2) |
| `ssearch` returns no results despite dropping a file in a watched dir | watcher in saver mode (battery) | `ssearch power active`, or set `auto_saver_on_battery = false` |
| `ssearch` says "ERROR: cannot reach daemon" | daemon not running | `launchctl kickstart -k gui/$UID/com.semanticsd` |
| Cross-modal vision queries return nothing | vision embedder not configured | Add `[embedding.vision]` block, restart daemon, then `ssearch reembed vision` |
| `budget_exceeded` in worker logs | hit your monthly cap | Raise `[budget].monthly_limit_usd` or wait for next month |
| Big bundled JS files dominating results | indexed code-y noise | Add patterns like `*.min.js`, `_astro`, `dist` to `[watch].ignore_patterns`, then re-index |
| Same file repeated in results | running `--chunks` mode (or pre-fix client) | omit `--chunks`; default collapses |

### Reading daemon logs

```bash
# launchd-managed:
log stream --process semanticsd --info

# Or the file the daemon writes:
tail -f ~/Library/Logs/semanticsd/semanticsd.log
```

---

## 🧱 Project layout

```
semanticsd/
├── cli.py                   semanticsd / ssearch entrypoints
├── config.py                TOML schema + parser
├── keychain.py              Keychain wrapper
├── db/                      sqlite-vec + V1..V4 migrations
├── embedders/               one file per provider; ABCs in base.py / vision_base.py
├── extractors/              one file per file class; registry maps extension → class
├── pipeline/                walk → extract → chunk → hash → queue → embed
├── search/                  semantic + grep + filename + RRF + profiles
├── usage/                   recorder + budget gate + reports
├── watcher/                 FSEvents observer + dirty queue + power controller
├── server/                  FastAPI app + routes
└── admin/                   launchd plist + install/uninstall
```

Each subpackage has its own README-worthy story; the `docs/superpowers/specs/` design docs are the cleanest narrative entry points.
