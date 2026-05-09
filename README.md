# SemanticsD — local multimodal semantic search for macOS

A daemon that indexes the files on your Mac and lets you search them
semantically — by meaning, not just by exact text — across **text, code,
PDFs, images, and audio**, all at once. Runs entirely on your machine
(plus optional API providers you configure), watches your folders for
changes via FSEvents, and exposes both an HTTP API and an `ssearch` CLI.

```
$ ssearch "There is no alternative" --all
~/Documents/sku_001_fold.jpg                   (vision, score=1.000)
  <image: sku_001_fold.jpg>

~/Documents/Obsidian Vault/Daily Journal/...   (semantic, score=0.62)
  ...framing flexes per audience. Don't change who you are...
```

The query above works because cross-modal embedders (Gemini Embedding 2,
Qwen3-VL) put text and image vectors in the same space — a slogan tshirt
gets surfaced from a text query.

---

## What's in the box

- **Indexing** — extracts text from `.md/.txt/.pdf/.docx/.xlsx/.pptx/.epub/.rtf/.html/.eml/.ipynb`, OCRs images via Tesseract (optional fallback), transcribes audio via Whisper.
- **Multi-modal embedding** — pluggable per-modality:
  - **Text**: Ollama (embeddinggemma, nomic-embed-text), OpenAI, OpenAI-compatible (LM Studio / vLLM / TEI), local sentence-transformers, Gemini.
  - **Vision**: Gemini Embedding 2 (cloud, 3072-d), local **Qwen3-VL-Embedding-2B** via sentence-transformers + MPS (2048-d, ~5GB RAM, no API key).
- **Storage** — sqlite-vec virtual tables, one per `(modality, dim)` combo so multiple providers coexist non-destructively. FTS5 with porter+unicode61 for grep + filename. Content-hash dedup so identical chunks aren't re-embedded.
- **Search** — semantic + filename + grep + hybrid mode with reciprocal-rank fusion, per-file-type weights (code boosts grep, prose boosts semantic, images route to vision-only), CWD-scoped by default, **file-collapse** so the same PDF doesn't flood the result list, cross-modal text→image queries.
- **FSEvents watcher** — drop a file in a watched dir, it's searchable in ~3-5s. Delete and it's unindexed.
- **Power modes** — `active` (watcher live) vs `saver` (watcher off, periodic full-walk). Optional auto-switch when you unplug from AC.
- **Cost tracking** — every embedder call writes a `usage` row. Monthly budget cap fail-closes when exceeded; `ssearch usage` surfaces spend.
- **Provider-switch reembed** — `ssearch reembed` queues jobs for chunks that don't yet have a vector from your *currently configured* embedder. Old vectors stay queryable for rollback.
- **Auth** — bearer token, stored in macOS Keychain. CORS-locked to localhost.

## Install

Requires macOS, Python ≥3.11 (3.13 recommended), and SQLite with loadable
extension support. Apple-Silicon Homebrew Python works out of the box;
pyenv builds need `--enable-loadable-sqlite-extensions`.

```bash
git clone git@github.com:vijay2411/litesearch-multimodal.git
cd litesearch-multimodal
make install
```

Optional embedders:

```bash
# Local text — Ollama
brew install ollama && ollama serve &
ollama pull embeddinggemma

# Cloud vision — Gemini API key
python -c "
from semanticsd import keychain
keychain.set_provider_key('gemini', 'YOUR_GEMINI_KEY')
"

# Local vision — Qwen3-VL via sentence-transformers (auto on first use, ~5GB)
# No additional setup; just configure the preset in config.toml
```

## Configure

Daemon reads `~/Library/Application Support/semanticsd/config.toml`
(override with `SEMANTICSD_HOME`). A working starter:

```toml
[watch]
directories = ["/Users/me/Documents", "/Users/me/Code"]
ignore_patterns = [".git", "node_modules", "*.min.js", "_astro"]
max_file_size_mb = 25

[embedding.text]
preset   = "ollama"
model    = "embeddinggemma"
base_url = "http://localhost:11434/v1"

[embedding.vision]               # optional
preset = "gemini"                 # or "qwen3_vl_local"
model  = "gemini-embedding-2"

[power]
mode = "active"                  # active | saver
auto_saver_on_battery = true     # flip to saver when unplugged

[budget]
monthly_limit_usd = 5.0          # 0 = unlimited
```

## Run

```bash
# Foreground (great for trying it out)
python -m semanticsd serve

# Or install as a launchd agent (auto-start on login)
python -m semanticsd install
python -m semanticsd uninstall   # tear down
```

## Use

```bash
# Search (CWD-scoped by default; --all for whole corpus)
ssearch "neural network architecture"
ssearch "supply chain ideas" --all
ssearch "spooky action at a distance" --all
ssearch "tshirt photo" --all              # cross-modal text → image

# Search modes
ssearch --semantic "..."     # vector only
ssearch --filename "..."     # path FTS only
ssearch --grep "..."         # chunk-text FTS only
# default = hybrid (all of the above + cross-modal vision, RRF-fused)

# Show repeated chunks of the same file (default collapses to one row per file)
ssearch "transaction history" --all --chunks

# Inspect daemon
ssearch --status                 # health summary
ssearch watch                    # watcher state
ssearch power                    # active | saver
ssearch usage                    # this month's spend by provider

# Switch providers
# 1. Edit [embedding.text] in config
# 2. Restart daemon
# 3. ssearch reembed text         # queue old chunks for re-embedding

# Trigger a full re-walk (saver mode or manual)
ssearch watch --sweep
```

## HTTP API

Auto-generated OpenAPI docs at `http://127.0.0.1:47600/docs`.

```
GET  /v1/health           — status, embedders, budget block
GET  /v1/search?q=...     — search (mode, limit, all, vision, collapse, cwd)
POST /v1/index            — manual indexing trigger
GET  /v1/watch            — watcher status
POST /v1/watch/sweep      — force full re-walk
GET  /v1/power            — current mode
POST /v1/power            — {"mode": "active" | "saver"}
GET  /v1/usage            — cost & volume report (since/until/provider filters)
POST /v1/reembed          — {"modality": "text" | "vision" | "all"}
GET  /v1/presets          — embedder providers available
POST /v1/embedder/test    — round-trip an embedder preset
```

All endpoints require `X-Auth-Token: <token>`. Get yours via `python -m semanticsd token print`.

## Architecture

```
semanticsd/
├── cli.py                    # `semanticsd` (admin) + `ssearch` (client) entrypoints
├── config.py                 # TOML loader + per-modality + budget + power
├── keychain.py               # auth token + per-provider API keys via macOS Keychain
├── paths.py, logging_setup.py
│
├── db/                       # sqlite-vec + migrations (V1..V4)
├── embedders/                # one file per provider
│   ├── base.py               # Embedder ABC (text)
│   ├── vision_base.py        # VisionEmbedder ABC
│   ├── local.py              # sentence-transformers (bge etc.)
│   ├── ollama.py             # /v1 endpoint, no API key
│   ├── openai.py             # OpenAI text-embedding-3-*
│   ├── openai_compatible.py  # LM Studio, vLLM, TEI, OpenRouter, ...
│   ├── gemini.py             # Gemini Embedding 2 (text)
│   ├── gemini_vision.py      # Gemini Embedding 2 (vision/cross-modal)
│   ├── qwen3_vl.py           # Qwen3-VL-Embedding-2B via MPS
│   ├── registry.py           # PROVIDER_REGISTRY + factories
│   └── router.py             # per-modality routing singleton
│
├── extractors/               # one file per file class
│   ├── text.py html.py pdf.py docx.py xlsx.py pptx.py epub.py
│   ├── rtf.py email_msg.py notebook.py image.py audio.py
│
├── pipeline/                 # walk → extract → chunk → hash → queue → embed
│   ├── walker.py ignore.py chunker.py hasher.py
│   ├── indexer.py            # also handles unindex_path for delete events
│   └── worker.py             # routes by modality + writes usage rows + budget gate
│
├── search/                   # query side
│   ├── semantic.py filename.py grep.py
│   ├── fusion.py             # weighted RRF
│   ├── profiles.py           # per-file-type weights (code/prose/data/image/...)
│   ├── snippets.py
│   └── engine.py             # composes everything, owns the LRU query cache
│
├── usage/                    # cost tracking
│   ├── recorder.py budget.py reports.py
│
├── watcher/                  # FSEvents + power
│   ├── events.py             # DirtyPathQueue (debounced, thread-safe)
│   ├── fsevents_watcher.py   # watchdog wrapper
│   ├── sweep.py              # initial + periodic
│   ├── battery.py            # psutil AC/battery probe
│   └── power.py              # PowerController state machine
│
├── server/                   # FastAPI + uvicorn
│   ├── app.py auth.py
│   └── routes/{health,presets,embedder_test,index,search,watch,power,usage,reembed}.py
│
├── admin/                    # launchd install/uninstall + plist
└── reembed.py                # queue-jobs-for-stale-chunks helper
```

## Tests

```bash
make test                                 # 332 unit tests
pytest -m slow                            # 56 e2e tests (need Ollama / Gemini)
pytest tests/test_e2e_stress_battery.py   # 33-query ground-truth ranking
```

## License

MIT — see [LICENSE](LICENSE).

## Status

Personal project that grew over an intense build week. All behavior
described above is shipped and verified end-to-end on a real-world
`~/Documents` corpus. There's a deferred backlog — Raycast extension,
MCP server, macOS Spotlight integration, cross-encoder reranker — in
`docs/superpowers/specs/`. Issues & PRs welcome.
