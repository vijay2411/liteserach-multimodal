# 🔍 SemanticsD

> **Local multimodal semantic search for macOS — find your files by what they mean, not what they're named.**

Type `tshirt photo` and the daemon surfaces a screenshot you saved 3 months ago — even though the filename is `sku_001_fold.jpg`. Type `there is no alternative` and it finds the photo of a slogan tee you took at a friend's place. Across text, code, PDFs, images, and audio. On your Mac. Indexing in the background. No data leaves your machine unless you opt in to a cloud embedder.

---

## ✨ What this is

- 🧠 A **search daemon** that runs in the background and watches your folders.
- 🎨 **Multimodal**: text, code, PDFs, scanned images, screenshots, voice notes — all queryable from one search bar.
- 🔌 **Provider-agnostic**: pick local (Ollama, sentence-transformers, Qwen3-VL on MPS) or cloud (Gemini, OpenAI) embedders, switch at any time, mix and match per modality.
- ⚡ **Live**: drop a file in a watched folder → searchable in ~3 seconds via FSEvents.
- 💾 **One sqlite-vec database** for everything. No vector DB to host.
- 🖥️ **CLI + HTTP API + (coming) web UI** — use it however you want.
- 🪪 **MIT-licensed**, ~5k lines of well-tested Python (332 unit + 56 e2e tests).

## ❌ What this isn't

- 🚫 **Not** a Spotlight replacement — it complements it. Spotlight is exact-match metadata; we do meaning.
- 🚫 **Not** a vector DB. We use one (sqlite-vec) but you don't manage it.
- 🚫 **Not** a RAG framework or LLM agent. It's the *retrieval* half — bring your own LLM if you want generation.
- 🚫 **Not** a cloud service. Runs locally; the only network calls are to your configured embedder providers.
- 🚫 **Not** cross-platform yet — macOS-only (FSEvents, launchd, Keychain, MPS). Linux/Windows would need wiring.
- 🚫 **Not** a polished consumer product. It's a working power-user tool with rough edges.

## 💪 Why this exists

Modern macOS search is broken for the way I actually use my Mac:

| Tool | Limitation |
|---|---|
| **Spotlight / Finder** | Filename + literal text only. Can't find "that tshirt photo" if the file is `IMG_2841.jpg`. |
| **`ripgrep` / grep** | Substring match. "Bank statement" won't find a PDF where it's encoded as the literal word "Statement". |
| **Notion AI / Obsidian Smart Connections** | Locked to one app's data. Can't search across all my files. |
| **Apple Intelligence** | Closed, unpredictable, no API, can't switch the model, can't index what *you* want indexed. |
| **Spotlight QuickLook OCR** | Sometimes works, often doesn't, no semantic layer at all. |

I wanted **one search across everything I own, by meaning, with the model I choose, on hardware I own**. So I built it.

## 👥 Who this is for

✅ **Use this if you:**

- 🛠️ Are a **power user / developer** who's comfortable editing TOML and reading CLI output.
- 💻 Live on **macOS** (Apple Silicon ideally — local models run on MPS).
- 🧑‍💼 Have a sprawl of **Documents / Obsidian / Code / screenshots** you can never find things in.
- 🎯 Want to **swap embedder models** to experiment (Ollama → Gemini → Qwen3-VL → ...).
- 🔒 Care about **keeping your data local** — at least for the text side.
- 🤖 Want to wire local search into Claude / Raycast / your own scripts via HTTP.

❌ **Don't use this if you:**

- 🪟 Are on **Windows or Linux**.
- 🖱️ Want a **one-click installer with a GUI**. (We have a CLI installer, but you'll edit a config.)
- ☁️ Want a **fully-managed cloud** product (try Pinecone / Weaviate / Supabase Vector instead).
- 📧 Want to search **Gmail / Slack / Notion** (only files on disk are indexed; integrations are out of scope for v1).
- 📊 Need it for **enterprise** — no SSO, no audit logs, no SLAs.
- 🤔 Don't know what an "embedding" is and don't want to learn (this needs *some* mental model to configure well).

## 🔧 Tech stack

| Layer | Tech |
|---|---|
| **Language** | Python 3.11+ (3.13 recommended) |
| **Runtime** | macOS launchd, FastAPI + Uvicorn |
| **Storage** | SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) (vector ANN) + FTS5 (full-text) |
| **File watching** | [watchdog](https://python-watchdog.readthedocs.io/) (uses FSEvents on macOS) |
| **Auth** | macOS Keychain via `keyring` |
| **Text embedders** | Ollama, OpenAI, OpenAI-compatible (LM Studio / vLLM / TEI), Gemini, sentence-transformers |
| **Vision embedders** | Gemini Embedding 2 (cloud), Qwen3-VL-Embedding-2B (local via MPS) |
| **Extractors** | pypdf, pypdfium2, python-docx, openpyxl, python-pptx, ebooklib, beautifulsoup4, striprtf, pytesseract (OCR), faster-whisper (audio) |

## ⚠️ Requirements

> 📌 **Read these carefully — most install issues are one of these.**

| Requirement | Why | How to verify |
|---|---|---|
| **macOS 13+** (Apple Silicon recommended) | FSEvents, launchd, Keychain, MPS | `sw_vers` |
| **Python 3.11+ with `enable_load_extension`** | sqlite-vec is a loadable extension | `python -c "import sqlite3; sqlite3.connect(':memory:').enable_load_extension(True)"` should not error. **Pyenv builds usually fail this** — use Homebrew Python (`brew install python@3.13`) instead. |
| **~500MB disk** for the daemon + base models | indexer + small text embedders | — |
| **+5GB disk** if using local Qwen3-VL | the 2B vision model | one-time download on first vision query |
| **Ollama** (optional, for local text embeddings) | hosts embeddinggemma, nomic-embed-text, etc. | `ollama list` |
| **Gemini API key** (optional, for cloud vision) | text→image cross-modal | https://aistudio.google.com/apikey |
| **Tesseract** (optional, for OCR fallback) | when no vision embedder is configured | `brew install tesseract` |

---

## 🧠 How it works under the hood

### Big picture

```
┌─────────────────┐
│  ~/Documents    │  ← you drop / edit / delete files
│  ~/Code         │
└────────┬────────┘
         │
         ▼  FSEvents (kernel-level file change events)
┌─────────────────────────────────────────────────────┐
│  WATCHER → DEBOUNCE → DIRTY QUEUE                   │
└────────┬────────────────────────────────────────────┘
         │
         ▼  per-file pipeline (only when changed)
┌─────────────────────────────────────────────────────┐
│  EXTRACT          (PDF→text, image→bytes,           │
│                   audio→whisper transcription)      │
│      ↓                                              │
│  CHUNK            (sliding window for prose,        │
│                   one image = one chunk)            │
│      ↓                                              │
│  HASH             (sha256 dedup so identical        │
│                   content isn't re-embedded)        │
│      ↓                                              │
│  ROUTE BY MODALITY                                  │
│   • text  → text embedder (Ollama / OpenAI / ...)   │
│   • image → vision embedder (Gemini / Qwen3-VL)     │
│      ↓                                              │
│  STORE            (sqlite-vec table per dim,        │
│                   FTS5 for grep + filename)         │
└────────┬────────────────────────────────────────────┘
         │
         ▼  query side
┌─────────────────────────────────────────────────────┐
│  ssearch "..."  ──► HYBRID SEARCH                   │
│                     • semantic (vector cosine)      │
│                     • grep (FTS5 BM25)              │
│                     • filename (FTS5 BM25)          │
│                     • cross-modal (text→vision)     │
│                            ↓                        │
│                     Reciprocal Rank Fusion          │
│                     + per-file-type weights         │
│                     + collapse-by-file              │
│                            ↓                        │
│                     ranked results                  │
└─────────────────────────────────────────────────────┘
```

### In one paragraph

When you save a file in a watched folder, the kernel tells our watcher within milliseconds. We extract its text (or render it to an image, for vision-eligible content), break it into chunks, hash each chunk, and ask the configured embedder to turn it into a vector. The vector lands in a sqlite-vec table sized for that embedder's output dimension. The chunk text goes into an FTS5 index for word-level matching. When you search, we ask the embedder for the query's vector, look up the nearest chunks, ALSO run FTS5 grep + filename match in parallel, and fuse all four signals via reciprocal-rank fusion — weighted differently for code vs prose vs images. Results are collapsed so each file appears once with its best chunk.

### Why this architecture works

- 🪶 **One DB**, not seven services. Backups are a single file copy.
- 🔁 **Per-modality, per-dim vec tables** mean you can swap embedders without re-embedding everything — old vectors stay queryable for rollback.
- 🧮 **Content hashing** means two notes that contain the same paragraph share one embedding. Free dedup.
- ⚖️ **Rank-based fusion** (RRF) sidesteps the "your scores are on different scales" problem that breaks naive multi-signal search.
- 🏷️ **File-type-aware weights** mean grep dominates for `.py` files and semantic dominates for `.md` — the right tool for the right kind of file.

### Things you can configure

```toml
# ~/Library/Application Support/semanticsd/config.toml

[watch]
directories = ["/Users/me/Documents"]    # watched dirs

[embedding.text]
preset = "ollama"                          # or "openai", "gemini", "local", ...

[embedding.vision]                          # optional
preset = "gemini"                          # or "qwen3_vl_local"

[budget]
monthly_limit_usd = 5.0                    # fail-closed when paid embedders cross cap

[power]
mode = "active"                            # or "saver" for periodic-only re-walks
auto_saver_on_battery = true               # flip to saver when unplugged
```

---

## 📖 More

- 📘 [**USAGE.md**](USAGE.md) — install, configure, run, CLI commands, HTTP API reference
- 🏛️ [**docs/superpowers/specs/**](docs/superpowers/specs/) — design specs for each plan (Foundation → Embedders → Pipeline → Multimodal → Search → Watcher → Cost & Reembed)
- 📜 [**LICENSE**](LICENSE) — MIT

## 🚧 Status

Personal project. All described behavior is shipped and verified end-to-end on a real-world `~/Documents` corpus. Roadmap: web UI → Raycast extension → MCP server → cross-encoder reranker. PRs welcome.
