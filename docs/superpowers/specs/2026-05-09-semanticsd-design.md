# SemanticsD — macOS Semantic Search Daemon

**Status:** Draft v3
**Date:** 2026-05-09
**Author:** Vedant Vijay

**Naming:** Project name is **SemanticsD** (capital D). Binary/service identifier is `semanticsd` (lowercase, daemon convention). launchd label is `com.semanticsd`. Python package is `semanticsd`.

## Goal

A plug-and-play semantic search service for macOS. A long-running Python daemon indexes files in the background, watches for changes via FSEvents, and serves search queries over a localhost HTTP API. The HTTP contract is the universal pluggability surface: any client (CLI, Raycast extension, MCP server, Shortcuts, Alfred, Hammerspoon, browser tools) reduces to thin code calling localhost.

Three search modes (semantic default, filename, grep). Pluggable embedding backends — anything OpenAI-compatible plus a zero-config local fallback.

## Phase Split (changed from v1)

**Phase 1: backend only.** Daemon, FastAPI HTTP API, CLI client, full embedding/indexing/search machinery. Ship a working `curl`-able service.

**Phase 2: frontend clients.** Raycast extension, MCP server, macOS Services menu, URL scheme handler, CoreSpotlight `.mdimporter`. Each is a thin wrapper over the Phase 1 HTTP API.

This split is deliberate: the backend contract is the load-bearing part. Get it right and every client is trivial.

## Non-Goals

- No custom UI / standalone app.
- No cross-machine sync.
- No hybrid search or rerankers in Phase 1 (deferred).
- No multi-user — single-user daemon, launchd user agent.
- No client integrations in Phase 1 except the CLI.

## Tech Stack

- **Language:** Python 3.11+
- **HTTP server:** FastAPI + Uvicorn (auto-generated OpenAPI/Swagger)
- **Vector store:** `sqlite-vec` (single-file SQLite extension)
- **File watching:** `watchdog` (FSEvents-backed on macOS)
- **Local embeddings:** `sentence-transformers` (default fallback)
- **OpenAI-compatible HTTP:** official `openai` Python SDK with custom `base_url`
- **Power detection:** `pyobjc-framework-IOKit` for `IOPSCopyPowerSourcesInfo`
- **Keychain:** `keyring` library (uses macOS Keychain Services backend)
- **Config:** TOML via `tomli`/`tomllib`

## Architecture Overview

```
                        ┌────────────────────────────────────────────┐
                        │              semanticsd                    │
                        │      (launchd user agent, Python)          │
                        │                                            │
   FSEvents (watchdog)─▶│  Watcher → Extractor → Chunker → Hasher    │
                        │                                ↓           │
                        │                       Job Queue (SQLite)   │
                        │                                ↓           │
                        │                          Embedder          │
                        │                  ┌─────────────┼─────────┐ │
                        │                  ▼             ▼         ▼ │
                        │            Local        OpenAI-     LiteLLM│
                        │      (sentence-trans.) compatible  (Phase 2)│
                        │                                ↓           │
                        │                  sqlite-vec + FTS5 + meta  │
                        │                                ↓           │
                        │                  Search Engine (3 modes)   │
                        │                                ↓           │
                        │            FastAPI on 127.0.0.1:47600      │
                        └────────────────────┬───────────────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼ (Phase 1)        (Phase 2)  ▼
                        ┌──────────┐    ┌────────────────────────────────┐
                        │ ssearch  │    │ Raycast / MCP / Services menu /│
                        │  (CLI)   │    │ URL scheme / CoreSpotlight     │
                        └──────────┘    └────────────────────────────────┘
```

## Filesystem Layout

| Path | Purpose |
|---|---|
| `~/Library/Application Support/semanticsd/config.toml` | User config |
| `~/Library/Application Support/semanticsd/index.db` | SQLite database (vectors via `sqlite-vec`, metadata, FTS, jobs) |
| `~/Library/Application Support/semanticsd/.semanticsdignore` | Optional gitignore-style exclusion file |
| `~/Library/Logs/semanticsd/semanticsd.log` | Structured logs (rotated) |
| `~/Library/LaunchAgents/com.semanticsd.plist` | launchd agent definition |
| macOS Keychain (`semanticsd` service) | API keys per provider preset |

## Components

### 1. Daemon Shell

- Bound to **`127.0.0.1` only** — never `0.0.0.0`. Loopback-only local service.
- Default port `47600` (configurable).
- Bearer token auth: `X-Auth-Token` header. Token generated on first run, stored in Keychain (`semanticsd / api_token`). CLI reads it from Keychain transparently.
- CORS configured to allow `localhost:*` and `127.0.0.1:*` origins (Raycast extensions, Tauri apps, browser tools all need this).
- API versioned at `/v1/...` from day one.
- launchd plist with `RunAtLoad=true`, `KeepAlive=true`, stdout/stderr redirected to log file.
- Graceful shutdown: drain in-flight indexing jobs (or mark them resumable) before exit.

### 2. Embedder Layer

The pluggable AI part. **One file per provider**, all deriving from a single base class. Adding a new provider = drop a new file in `semanticsd/embedders/`, register it, done.

```
semanticsd/embedders/
    __init__.py            # registry / factory
    base.py                # abstract Embedder
    local.py               # sentence-transformers (default)
    openai_compatible.py   # generic OpenAI /v1/embeddings shape
    openai.py              # OpenAI specific (subclass of openai_compatible)
    ollama.py              # Ollama (subclass of openai_compatible, no API key, model auto-pull)
    voyage.py              # Voyage (input_type semantics)
    cohere.py              # Cohere (input_type semantics, distinct API)
    gemini.py              # Google Gemini (distinct API shape, single-content batching quirk)
    vertex.py              # Vertex AI (auth via gcloud creds)
    bedrock.py             # AWS Bedrock (sigv4 signing)
    registry.py            # PROVIDER_REGISTRY mapping for /v1/presets
```

**Base interface (`base.py`):**

```python
class Embedder(ABC):
    provider_id: str            # "local" | "openai" | "gemini" | "cohere" | ...
    model_id: str               # e.g. "text-embedding-3-small"
    dim: int                    # vector dim (after Matryoshka truncation if applied)
    supports_kind: bool         # True if provider distinguishes doc vs query (Cohere, Voyage)

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        kind: Literal["doc", "query"],
    ) -> EmbedResult: ...

    @abstractmethod
    def health_check(self) -> tuple[bool, str]: ...

    @abstractmethod
    def estimate_tokens(self, texts: list[str]) -> int: ...

    # Cost — providers override with their per-1M-token pricing.
    cost_per_million_input_tokens_usd: float = 0.0


class EmbedResult(BaseModel):
    vectors: list[list[float]]
    input_tokens: int            # for cost accounting
    output_tokens: int = 0       # embeddings have no output tokens, kept for symmetry
    raw_response: dict | None    # for debugging
```

**Phase 1 ships these implementations:**

| File | Class | Notes |
|---|---|---|
| `local.py` | `LocalEmbedder` | sentence-transformers, default `BAAI/bge-small-en-v1.5` (384-d, ~30MB, fast on CPU). Cost = 0. Zero-config. |
| `openai_compatible.py` | `OpenAICompatibleEmbedder` | Generic base for any `/v1/embeddings` endpoint. Constructor: `(base_url, api_key, model, dimensions=None)`. Used directly for LM Studio, vLLM, llama.cpp, OpenRouter, Together, Groq, Fireworks, TEI, custom self-hosted. |
| `openai.py` | `OpenAIEmbedder(OpenAICompatibleEmbedder)` | Subclass with hardcoded `base_url="https://api.openai.com/v1"` + cost tables for `text-embedding-3-small/large` + `ada-002`. |
| `ollama.py` | `OllamaEmbedder(OpenAICompatibleEmbedder)` | Subclass with `base_url="http://localhost:11434/v1"`, no API key, auto-pulls model via `ollama pull` if missing. |
| `voyage.py` | `VoyageEmbedder` | Native Voyage SDK call with `input_type` mapping (doc → `"document"`, query → `"query"`). Cost table for voyage-3 family. |
| `cohere.py` | `CohereEmbedder` | Native Cohere SDK with strict `input_type`. Cost table for embed-v3 family. |
| `gemini.py` | `GeminiEmbedder` | Google `google-generativeai` SDK. Handles single-content batching quirk (loop one-at-a-time when needed). Cost table. |
| `vertex.py` | `VertexEmbedder` | Vertex AI via `google-cloud-aiplatform`, gcloud-creds auth. Cost table. |
| `bedrock.py` | `BedrockEmbedder` | AWS Bedrock via `boto3`, sigv4. Cost table. |

**Phase 2 backends:**
- `litellm.py` — escape hatch via LiteLLM library for any provider not yet covered.
- Reranker module mirroring this pattern (`semanticsd/rerankers/`).

**Adding a new provider:**

1. Create `semanticsd/embedders/<provider>.py` subclassing `Embedder` (or `OpenAICompatibleEmbedder` if it's `/v1/embeddings`-shaped).
2. Implement `embed`, `health_check`, `estimate_tokens`, set `cost_per_million_input_tokens_usd`.
3. Register in `registry.py` `PROVIDER_REGISTRY` dict.
4. That's it — preset shows up in `/v1/presets`, frontends pick it up automatically.

**Preset registry** (built-in dict in `registry.py`, exposed via `GET /v1/presets`):

```python
PROVIDER_REGISTRY = {
    "local":      {"class": "LocalEmbedder",              "default_model": "BAAI/bge-small-en-v1.5",  "needs_api_key": False, "needs_base_url": False},
    "ollama":     {"class": "OllamaEmbedder",             "default_model": "nomic-embed-text",        "needs_api_key": False, "needs_base_url": False},
    "lmstudio":   {"class": "OpenAICompatibleEmbedder",   "default_model": None,                       "needs_api_key": False, "needs_base_url": True,  "default_base_url": "http://localhost:1234/v1"},
    "vllm":       {"class": "OpenAICompatibleEmbedder",   "default_model": None,                       "needs_api_key": False, "needs_base_url": True,  "default_base_url": "http://localhost:8000/v1"},
    "openai":     {"class": "OpenAIEmbedder",             "default_model": "text-embedding-3-small",  "needs_api_key": True,  "needs_base_url": False},
    "voyage":     {"class": "VoyageEmbedder",             "default_model": "voyage-3",                "needs_api_key": True,  "needs_base_url": False},
    "cohere":     {"class": "CohereEmbedder",             "default_model": "embed-english-v3.0",      "needs_api_key": True,  "needs_base_url": False},
    "gemini":     {"class": "GeminiEmbedder",             "default_model": "text-embedding-004",      "needs_api_key": True,  "needs_base_url": False},
    "vertex":     {"class": "VertexEmbedder",             "default_model": "text-embedding-004",      "needs_api_key": False, "needs_base_url": False},
    "bedrock":    {"class": "BedrockEmbedder",            "default_model": "amazon.titan-embed-text-v2:0", "needs_api_key": False, "needs_base_url": False},
    "openrouter": {"class": "OpenAICompatibleEmbedder",   "default_model": None,                       "needs_api_key": True,  "needs_base_url": True,  "default_base_url": "https://openrouter.ai/api/v1"},
    "together":   {"class": "OpenAICompatibleEmbedder",   "default_model": "togethercomputer/m2-bert-80M-8k-retrieval", "needs_api_key": True, "needs_base_url": True, "default_base_url": "https://api.together.xyz/v1"},
    "custom":     {"class": "OpenAICompatibleEmbedder",   "default_model": None,                       "needs_api_key": None,  "needs_base_url": True},
}
```

API keys live in macOS Keychain under service `semanticsd`, account `<provider_id>` (e.g. `openai`, `voyage`, `gemini`, `cohere`).

### 3. Vector Store + Schema

`sqlite-vec` extension loaded into the SQLite database.

```sql
-- Files known to the indexer.
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    modified_at INTEGER NOT NULL,
    size INTEGER NOT NULL,
    file_type TEXT NOT NULL,
    indexed_at INTEGER,
    last_error TEXT
);

-- Chunks extracted from files. Content-addressed via hash for dedup.
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,        -- sha256 of normalized text
    byte_start INTEGER NOT NULL,
    byte_end INTEGER NOT NULL,
    UNIQUE(file_id, chunk_index)
);
CREATE INDEX idx_chunks_hash ON chunks(content_hash);

-- Embeddings table tagged with provenance triplet.
CREATE VIRTUAL TABLE vec_embeddings USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[384]               -- dim set at table creation; recreated on dim change
);

CREATE TABLE embedding_meta (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    content_hash TEXT NOT NULL          -- denormalized for dedup lookup
);
CREATE INDEX idx_emb_meta_hash ON embedding_meta(content_hash, provider_id, model_id, dim);

-- Job queue for resumable indexing.
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    status TEXT NOT NULL,               -- 'pending' | 'in_flight' | 'done' | 'failed'
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX idx_jobs_status ON jobs(status);

-- FTS5 for filename and grep search.
CREATE VIRTUAL TABLE fts_chunks USING fts5(text, content='chunks', content_rowid='id');
CREATE VIRTUAL TABLE fts_paths USING fts5(path, content='files', content_rowid='id');

-- Service-level metadata.
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- meta keys: schema_version, active_provider_id, active_model_id, active_dim,
--           last_full_reindex, power_mode
```

**Vector-space integrity:** every embedding is tagged `(provider_id, model_id, dim)`. Search refuses to mix triplets — if the active embedder triplet differs from the indexed triplet, return a clear `409 Conflict {"error": "reindex_required", "current": ..., "indexed": ...}` and surface this in `/health` and `/stats`. No silent re-embed, no silent fail.

**Dim change handling:** `vec0` virtual table fixes its dimension at creation. Switching to a different-dim embedder triggers `DROP TABLE vec_embeddings; CREATE VIRTUAL TABLE vec_embeddings USING vec0(chunk_id, embedding FLOAT[<new_dim>])` as part of the explicit `POST /v1/reindex`. We also clear `embedding_meta`.

### 4. Document Pipeline

Five stages, each pluggable.

1. **Source / Watcher** — `watchdog` (FSEvents on macOS) over user-configured directories. Default: `$HOME` minus the `.semanticsdignore`. On startup: reconciliation pass diffing filesystem against `files` table.

2. **Extractor** (per file type) — comprehensive Phase 1 coverage. **One extractor per file class, all deriving from `Extractor` base** so adding new types is the same drop-in pattern as embedders. Lives in `semanticsd/extractors/`.

   | File class | Extensions | Library | Strategy |
   |---|---|---|---|
   | Plain text / markdown | `.txt .md .rst .org` | builtin | direct read |
   | Source code | `.py .js .ts .rs .go .java .c .cpp .h .swift .rb .php .sh ...` | builtin | direct read (tree-sitter chunking in Phase 2) |
   | Structured data | `.json .yaml .yml .toml .csv .tsv .xml` | builtin | direct read; CSV row-aware |
   | HTML / web pages | `.html .htm` | `beautifulsoup4` | strip tags, keep text + headings |
   | PDF | `.pdf` | `pypdf` (text) + `pytesseract` (OCR fallback for image-only PDFs) | per-page; OCR triggers when extracted text is empty |
   | DOCX / DOC | `.docx .doc` | `python-docx` (DOCX), `textract` (DOC) | paragraph-level |
   | XLSX | `.xlsx .xls` | `openpyxl` / `xlrd` | sheet-level rows |
   | PPTX | `.pptx .ppt` | `python-pptx` | slide-level |
   | EPUB | `.epub` | `ebooklib` | chapter-level |
   | RTF | `.rtf` | `striprtf` | direct read |
   | Email | `.eml .mbox` | builtin `email` / `mailbox` | headers + body |
   | Notebooks | `.ipynb` | builtin (JSON parse) | code + markdown cells |
   | Archives | `.zip .tar .gz` | builtin | walk in, extract, recurse extractors (size-bounded) |
   | **Images** | `.png .jpg .jpeg .heic .gif .bmp .webp .tiff` | `pytesseract` (OCR text) + optional CLIP vision embedding | OCR result becomes text chunk; image bytes optionally produce a vision-modality vector via CLIP if a vision-capable embedder is configured |
   | **Audio** | `.mp3 .wav .m4a .flac .ogg .aac` | `faster-whisper` (local) or Whisper API | transcribe → chunk transcript by segment with timestamps |
   | **Video** | `.mp4 .mov .mkv .webm` | `ffmpeg` (audio extract) → `faster-whisper` | strip audio, transcribe |

   **Default extractors enabled at install:** all of the above. Heavy deps (`pytesseract`, `faster-whisper`, `ebooklib`, `python-pptx`) listed in `requirements.txt`. The `install.sh` script also installs system-level Tesseract (`brew install tesseract`) and ffmpeg (`brew install ffmpeg`) if missing.

   **Vision embedding for images (Phase 1):**
   - If the active embedder advertises `supports_vision = True` (e.g., a CLIP-based local embedder, or an OpenAI vision-compat endpoint), images additionally produce a vision-modality chunk with image bytes embedded directly.
   - Default `LocalEmbedder` ships an optional **CLIP path** (`open-clip-torch` with `ViT-B-32`, ~150MB, downloaded lazily on first image). Off by default to keep first-run lean; enabled via `[extractors.images] enable_clip = true`.
   - Without CLIP, images still searchable via OCR text.

   **Audio embedding (Phase 1):**
   - Audio is transcribed to text and embedded as text. No separate audio-modality embedding.
   - Default Whisper model: `faster-whisper`'s `base` (~140MB) for balance. Configurable: `tiny` (~75MB) / `small` (~500MB) / `medium` / `large`.

3. **Chunker** — sliding window 512 tokens, 64 overlap (configurable). Pluggable interface so semantic chunking can be swapped later.

4. **Hasher** — SHA-256 of normalized chunk text. **Cost lever:** before embedding, look up `(content_hash, provider_id, model_id, dim)` in `embedding_meta`. If a vector already exists for this exact content+model, copy it instead of re-embedding. Same content across many files = one API call.

5. **Embedder** — batch-call the active embedder; persist vectors + meta; mark jobs done.

### 5. Resumable Indexing (explicit user requirement)

The job queue is the single source of truth for "what still needs embedding."

- File change detected → upsert chunks → for each new/changed chunk, insert `jobs(status='pending')`.
- Worker (background asyncio task) loops: pick N pending jobs → mark `in_flight` → batch-embed → on success, write to `vec_embeddings` + `embedding_meta`, mark `done`. On failure, increment `attempts`, store error, mark `pending` again (until `attempts >= max_attempts`, then `failed`).
- **Crash recovery:** on daemon startup, run `UPDATE jobs SET status='pending' WHERE status='in_flight'`. Any work that was mid-batch when the process died gets retried.
- **Cost protection:** the dedup lookup runs *before* the embedding call. A re-run after a crash will find already-completed embeddings (by content_hash) and skip them, even if the corresponding jobs row was lost.
- **Idempotency:** content-hash + triplet means re-running indexing is a no-op for unchanged content. Safe to interrupt and resume any time.

### 6. Search Engine

Three modes; all return the same result shape:

```python
class SearchResult(BaseModel):
    path: str
    chunk_text: str
    score: float
    match_type: Literal["semantic", "filename", "grep"]
    chunk_index: int
    byte_range: tuple[int, int]
```

- **Semantic (default):** `embed([query], kind="query")` → `sqlite-vec` MATCH query for top-K nearest.
- **Filename:** FTS5 over `fts_paths` plus `LIKE` fallback for substring. One result per matched file.
- **Grep:** FTS5 over `fts_chunks` for literal/phrase. Regex queries (via `--regex` flag) fall back to streaming Python `re` over chunk text.

User picks mode per query via `mode` parameter. Phase 2 adds hybrid (run all 3, dedupe, rerank).

### 7. Cost & Accuracy Defaults (explicit user priority: cost > 100% accuracy)

- **Default model:** `bge-small-en-v1.5` (free, local, 384-d).
- **API model default (when API selected):** `text-embedding-3-small` over `-large` — meaningfully cheaper, near-equivalent quality for most use cases.
- **Matryoshka truncation knob:** `dimensions` config field. text-embedding-3-* models support truncating from 1536/3072 down to e.g. 512 with minimal quality loss and ~3x storage/query savings.
- **Content-hash dedup:** never re-embed identical chunks across files.
- **Aggressive batch size:** default 128 (configurable). Single-call API costs are dominated by per-request overhead; bigger batches are strictly cheaper.
- **Skip large files:** default `max_file_size_mb=50`. Configurable.
- **Active mode debounce:** 500ms after a file change before re-embedding (collapses rapid sequential saves).

### 8. Cost Tracking

Every embedding call is metered. Per-call usage and cost are persisted; aggregates exposed via the API.

**Schema addition:**

```sql
CREATE TABLE usage (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    operation TEXT NOT NULL,        -- 'embed_doc' | 'embed_query'
    input_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,         -- 0.0 for local
    chunk_count INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL
);
CREATE INDEX idx_usage_time ON usage(timestamp);
CREATE INDEX idx_usage_model ON usage(provider_id, model_id, timestamp);
```

**Behavior:**
- Each `embed()` call returns `EmbedResult` with `input_tokens`. The worker records a `usage` row with `cost_usd = (input_tokens / 1_000_000) * cost_per_million_input_tokens_usd`.
- `LocalEmbedder` records rows with `cost_usd = 0` (still useful for measuring throughput).
- Token counts: each provider class implements `estimate_tokens()` — for OpenAI-shaped APIs we use `tiktoken`; for others we use the provider's response field if returned, else a heuristic (chars / 4).

**API endpoints:**

```
GET /v1/usage?since=<ts>&until=<ts>&group_by=day|model|provider
  returns: {
    total_cost_usd,
    total_input_tokens,
    total_calls,
    breakdown: [
      { key, cost_usd, input_tokens, calls, ... }
    ]
  }

GET /v1/usage/budget
  returns: { monthly_budget_usd, month_to_date_usd, projected_usd, status: "ok"|"warning"|"exceeded" }

PUT /v1/usage/budget
  body: { monthly_budget_usd: float, on_exceed: "warn"|"block" }
```

**Budget enforcement:** if `on_exceed = "block"` and the monthly budget is hit, the worker pauses and `/v1/health` reports `status: "degraded", reason: "budget_exceeded"`. User must raise budget or switch backend.

**CLI:**
```
ssearch --usage                                 # last 30 days, totals + per-model
ssearch --usage --since 2026-04-01
ssearch --usage --budget                        # show budget status
ssearch --usage --set-budget 10                 # set monthly cap to $10
```

### 9. Power Modes

Two modes, switchable at runtime via `POST /v1/power` or `ssearch --power active|saver`. Auto-detection via `IOKit` `IOPSCopyPowerSourcesInfo`.

| Aspect | Active (default on AC) | Saver (default on battery) |
|---|---|---|
| FSEvents watcher | running | paused |
| File-change → embed | real-time (500ms debounce) | not triggered |
| Periodic diff-reindex | not run | every `saver_reindex_interval` (default 1h): walk dirs, compare mtime+size vs `files` table, queue diffs |
| Manual reindex | always available | always available |
| Search queries | full | full (against existing index) |

Hot-switch — no restart, no reindex.

### 10. HTTP API (`/v1/...`)

All endpoints require `X-Auth-Token` header. Bound to `127.0.0.1`. CORS allows localhost origins.

```
POST /v1/search
  body:    { query, mode: "semantic"|"filename"|"grep", limit, paths?, collection? }
  returns: { results: SearchResult[], elapsed_ms, embedder: {provider_id, model_id, dim} }

POST /v1/index
  body:    { path? } | { source, content, metadata? }
  returns: { job_id, queued: int }
  -- triggers indexing for a path (file or dir) or for inline content.

DELETE /v1/documents/{path}
  returns: { deleted: int }

POST /v1/reindex
  body:    { force?: bool, paths?: [str] }
  returns: { queued: int, status: "started" }
  -- if embedder triplet changed, force=true is required (acknowledges destructive op).

GET /v1/health
  returns: { status: "ok"|"degraded", embedder: {ok, message}, vector_store: {ok}, doc_count, queue_depth }

GET /v1/stats
  returns: { files_indexed, chunks_indexed, embeddings, jobs: {pending, in_flight, done, failed},
             active_triplet: {provider_id, model_id, dim}, last_index_at, power_mode }

GET /v1/config
  returns: <current non-secret config>

PUT /v1/config
  body:    <partial config>
  returns: { applied, requires_reindex, dim_changed }

GET /v1/presets
  returns: { presets: PRESETS }
  -- frontends use this to render their provider dropdown.

POST /v1/embedder/test
  body:    { backend: "local"|"openai", base_url?, api_key?, model, dimensions? }
  returns: { ok, dim, latency_ms, error? }
  -- frontends use this for "Test Connection" button.

POST /v1/power
  body:    { mode: "active"|"saver" }
  returns: { applied }

GET /v1/file/spotlight_metadata?path=<path>
  returns: { kMDItemTextContent, semantic_keywords }
  -- reserved for Phase 2 CoreSpotlight importer; stable contract from day 1.

GET /docs              -- FastAPI Swagger UI
GET /openapi.json      -- OpenAPI spec for client codegen
```

### 11. CLI (`ssearch`)

Thin client over the HTTP API. The CLI's only job is to be a usable shell front-end and to prove the API contract is good enough that any future client is easy.

```
ssearch "how does auth work"                # semantic
ssearch --mode filename "config"
ssearch --mode grep "TODO:.*fixme" --regex
ssearch --status                            # GET /v1/health + /v1/stats
ssearch --reindex [path] [--force]
ssearch --index <path>
ssearch --config get embedding.backend
ssearch --config set embedding.backend ollama
ssearch --power active|saver
ssearch --presets                           # list providers
ssearch --test-embedder <preset>            # round-trip test, show dim + latency
```

Plus admin subcommands:
```
semanticsd serve                            # run daemon (normally invoked by launchd)
semanticsd install                          # install plist, generate token, start agent
semanticsd uninstall                        # stop and remove agent (does not delete index)
semanticsd token print                      # print current API token (CLI uses it transparently)
```

## Installation

Two deliverables ship at the repo root for setup:

### `requirements.txt`

Single file with all Python dependencies pinned to compatible ranges. Includes:

```
# Core
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.6
tomli>=2.0; python_version < "3.11"
typer>=0.12             # CLI

# Vector store
sqlite-vec>=0.1

# File watching
watchdog>=4.0

# Default local embedder
sentence-transformers>=2.7
torch>=2.2

# Optional vision (CLIP)
open-clip-torch>=2.24
pillow>=10.0

# OpenAI-compatible / OpenAI / Voyage / Cohere / Gemini / Vertex / Bedrock
openai>=1.30
voyageai>=0.2
cohere>=5.0
google-generativeai>=0.5
google-cloud-aiplatform>=1.50
boto3>=1.34

# Token counting
tiktoken>=0.7

# Extractors
pypdf>=4.0
python-docx>=1.1
openpyxl>=3.1
xlrd>=2.0
python-pptx>=0.6
beautifulsoup4>=4.12
ebooklib>=0.18
striprtf>=0.0.27
pytesseract>=0.3.10
faster-whisper>=1.0

# Keychain
keyring>=24.0

# macOS power detection
pyobjc-framework-IOKit>=10.0

# Dev (optional, gated)
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27
```

LiteLLM is intentionally not listed — pulled in lazily in Phase 2.

### `install.sh`

Bash script at repo root that gets a fresh Mac to a running daemon. Idempotent.

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Verify macOS + Python 3.11+
# 2. Install Homebrew system deps if missing: tesseract, ffmpeg
#      command -v brew >/dev/null || { echo "Install Homebrew first"; exit 1; }
#      brew list tesseract >/dev/null 2>&1 || brew install tesseract
#      brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
# 3. Create venv at ~/Library/Application\ Support/semanticsd/venv
# 4. pip install -r requirements.txt into the venv
# 5. Create config dir + default config.toml
# 6. Generate auth token, store in Keychain (service=semanticsd, account=api_token)
# 7. Render and install launchd plist at ~/Library/LaunchAgents/com.semanticsd.plist
#      Plist invokes the venv's python -m semanticsd serve
# 8. launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.semanticsd.plist
# 9. Wait for /v1/health to return 200 (timeout 30s)
# 10. Print "SemanticsD installed" + token location + next-step hints
```

Pairs with `install.sh --uninstall` (or a separate `uninstall.sh`) to reverse: `launchctl bootout`, remove plist, optionally remove venv (with `--purge` to also remove index/config).

A `Makefile` at repo root provides `make install`, `make uninstall`, `make dev` (run daemon in foreground without launchd), `make test`.

## Configuration

`~/Library/Application Support/semanticsd/config.toml`:

```toml
[watch]
directories = ["~/"]
ignore_patterns = [".git", "node_modules", ".DS_Store", "target", "build", "*.o"]
max_file_size_mb = 50

[embedding]
backend = "local"               # "local" | "openai"  (Phase 2: "litellm")
preset = "local"                # one of PRESETS keys; "custom" allowed
model = "BAAI/bge-small-en-v1.5"

# For backend="openai":
base_url = ""                   # set when preset="custom"
dimensions = 0                  # 0 = native dim; >0 = Matryoshka truncation
batch_size = 128

[search]
default_mode = "semantic"
max_results = 20

[chunking]
strategy = "sliding"            # Phase 2 adds: "semantic", "tree-sitter"
window_tokens = 512
overlap_tokens = 64

[daemon]
http_host = "127.0.0.1"
http_port = 47600
log_level = "info"

[power]
mode = "active"                 # "active" | "saver"
saver_reindex_interval = "1h"
saver_pause_watcher = true
auto_saver_on_battery = true

[indexing]
max_attempts = 5                # job retry budget before status="failed"
worker_concurrency = 2
```

API keys live in Keychain, never in this file.

## Error Handling

- **Single-file parse failure:** log, mark `files.last_error`, set `indexed_at = -1` (poisoned), skip until next change.
- **Embedding job failure:** increment `attempts`, store error, retry with exponential backoff. After `max_attempts`, mark `status='failed'` and surface count in `/v1/stats`.
- **Embedder unreachable** (e.g., `backend="ollama"` but Ollama not running): `/v1/health` returns `status: "degraded"`, the worker pauses (jobs stay `pending`), search and read endpoints continue serving stale-but-valid results. No silent fallback.
- **Triplet mismatch on query:** `409 Conflict {"error": "reindex_required", "current": ..., "indexed": ...}`.
- **Crash mid-embed:** in-flight jobs reset to pending on next start; content-hash dedup ensures any vectors that did get written aren't re-billed.
- **SQLite errors:** fatal, daemon exits, launchd restarts.
- **FSEvents drop / coalesce:** rely on startup reconciliation pass and (in saver mode) periodic diff-reindex.

## Security & Privacy

- Daemon runs as user, never root.
- HTTP bound to `127.0.0.1` only.
- Bearer token required on every API call. Token in Keychain.
- API keys in Keychain, never config file.
- No outbound network unless an API embedder is configured.
- Indexed text and paths stored in SQLite under user-only filesystem permissions.

## Testing Strategy

- **Unit:** parser per file type with fixture files; chunker; hasher; embedder protocol with a deterministic mock backend.
- **Integration:** spin up daemon with temp dir + mock embedder, write files, assert index state and search results via HTTP.
- **Resumability test:** start indexing, kill the worker mid-batch, restart, assert no duplicate API calls (verified via mock counter) and final index correct.
- **FSEvents test:** write/modify/delete files in temp watched dir; assert index reflects within deadline.
- **Triplet-mismatch test:** index with embedder A, switch to B with different dim, assert search returns 409 and reindex resolves.
- **End-to-end (gated):** real Ollama and real OpenAI in CI behind env-var gates.

## Phase Plan

**Phase 1 — Backend (this spec):**

1. Project scaffold (Python 3.11+, FastAPI, sqlite-vec, watchdog, sentence-transformers, openai SDK, keyring, tomli) + `requirements.txt` + `install.sh` + `Makefile`.
2. Filesystem layout + macOS-conventional paths + launchd plist generator.
3. SQLite schema + migrations + sqlite-vec extension load.
4. **Embedder layer**: base + per-provider files — `local`, `openai_compatible`, `openai`, `ollama`, `voyage`, `cohere`, `gemini`, `vertex`, `bedrock`. Registry + factory.
5. Document pipeline: full extractor set (text/code/markdown/PDF/DOCX/XLSX/PPTX/HTML/EPUB/RTF/email/notebook/archive/**images via OCR + optional CLIP**/**audio via Whisper**/**video via ffmpeg+Whisper**) + sliding-window chunker + content hasher.
6. Job queue + resumable indexing worker (crash recovery + content-hash dedup).
7. File watcher + reconciliation + power modes (active / saver).
8. **Cost tracking**: usage table, per-call accounting, `/v1/usage` + budget enforcement.
9. Search engine (semantic + filename + grep).
10. FastAPI HTTP API (`/v1/*`) + bearer token + CORS + OpenAPI.
11. CLI (`ssearch`) + admin subcommands (`install`, `uninstall`, `token print`, `serve`).
12. Keychain integration (`keyring` for API keys + auth token).
13. Architecture hooks reserved: `/v1/file/spotlight_metadata` endpoint stub for Phase 2 CoreSpotlight.

**Phase 2 — Clients & extras (deferred):**

- Raycast extension (TypeScript, calls HTTP API).
- MCP server (`semanticsd-mcp` stdio subprocess that proxies to the HTTP API).
- macOS Services menu entry + URL scheme handler (`semanticsd://search?q=...`).
- CoreSpotlight `.mdimporter` bundle.
- LiteLLM escape hatch for any provider not yet covered by the per-provider files.
- Hybrid search mode (semantic + filename + grep merged).
- Reranker support (`semanticsd/rerankers/` mirroring embedders pattern: Cohere reranker, local cross-encoder).
- Tree-sitter / semantic chunking.
- Audio-modality embeddings (separate from transcript text).
- macOS distribution: signed `.pkg` and Homebrew tap.

## Resolved Decisions

- **Project name:** SemanticsD (capital D). Daemon binary / package: `semanticsd`. launchd label: `com.semanticsd`.
- **Language:** Python 3.11+ + FastAPI. Rust deferred — clean migration path later if distribution becomes the bottleneck.
- **Vector store:** `sqlite-vec`. Single file, no server, modern, embedded.
- **Default embedder:** `LocalEmbedder` with `bge-small-en-v1.5`. Zero-config, offline, free.
- **Embedder pluggability:** one file per provider in `semanticsd/embedders/`, all deriving from `Embedder` base. Phase 1 ships local, openai_compatible, openai, ollama, voyage, cohere, gemini, vertex, bedrock. Adding a new provider = drop a new file + register.
- **Extractor pluggability:** mirrors embedders. One file per file class in `semanticsd/extractors/`. Phase 1 ships full coverage including images (OCR + optional CLIP), audio (Whisper), video (ffmpeg + Whisper).
- **File-type coverage Phase 1:** text/code/markdown/PDF/DOCX/XLSX/PPTX/HTML/EPUB/RTF/email/notebook/archive/images/audio/video. No file-type deferred to Phase 2.
- **Cost > perfection:** content-hash dedup, Matryoshka truncation knob, small default models, aggressive batching, explicit usage metering with budget enforcement.
- **Cost tracking:** `usage` table + per-provider cost-per-million-tokens tables + `/v1/usage` + monthly budget with optional hard block.
- **Resumable indexing:** explicit job queue + content-hash idempotency + crash recovery on startup.
- **Installation:** `requirements.txt` (single file, all deps) + `install.sh` (Bash bootstrap: brew deps, venv, pip install, config init, Keychain token, launchd plist install) + `Makefile`.
- **CoreSpotlight & all client integrations:** Phase 2. HTTP API contract in Phase 1 reserves their endpoints.
- **MCP server:** Phase 2 (thin stdio proxy over HTTP API).
- **API keys & auth token:** macOS Keychain via `keyring` library.
- **Auth:** bearer token (`X-Auth-Token`) generated on first run.
- **Bind:** `127.0.0.1` only, never `0.0.0.0`.
- **Versioning:** `/v1/...` from day one.
- **Triplet vector-space integrity:** every vector tagged `(provider_id, model_id, dim)`; cross-triplet queries return `409 reindex_required`.
