# Multi-Modal Embedding for SemanticsD

**Status:** approved, pending implementation
**Date:** 2026-05-09
**Supersedes section of:** `2026-05-09-semanticsd-design.md` (the "Embedder Layer" + "Document Pipeline" sections)

## Goal

Replace the current text-only embedding pipeline with a hybrid multi-modal one. Different file types route to different embedding models:

- **Text/code/markdown/docs** → text embedding model (Ollama embeddinggemma, nomic-embed-text, etc.)
- **Images, scanned PDFs, slide renders** → vision embedding model (Gemini Embedding 2 via API; local Ollama vision later when supported)
- **Audio** → Whisper → text → text embedding (unchanged)

The embedding model for each modality must be plug-and-play — switchable per data type via config. Adding a new provider should be a single new file in `embedders/`.

## Non-goals

- Tri-encoder (text+image+audio in one model) is out of scope.
- Cross-modal search ("find images that match this text query") is enabled by Gemini Embedding 2's shared text/image space, but the search engine itself is Plan 5; this plan only stores the vectors.
- Reembedding existing corpus when switching providers is deferred. Old vectors stay queryable; new content uses the new provider.

## Architecture

### Two ABCs

```
semanticsd/embedders/base.py
    Embedder (text)
        embed(texts: list[str], kind) -> EmbedResult       # unchanged

semanticsd/embedders/vision_base.py    (new)
    VisionEmbedder
        embed_images(images: list[bytes], kind) -> EmbedResult
```

`Embedder` is unchanged — every existing provider (Local, OpenAI, Ollama, OpenAI-compatible) keeps its current contract.

`VisionEmbedder` accepts raw image bytes. Each provider handles its own format (base64-encoding for HTTP APIs, PIL conversion for local models). Supported input formats: PNG, JPEG, WebP. Provider declares `dim`, `provider_id`, `model_id`, `cost_per_million_image_tokens_usd`.

A provider that supports both modalities (Gemini Embedding 2) ships as **two separate classes** in two files — one per ABC. Keeps each class focused; no diamond hierarchy.

### Embedder Router

```
semanticsd/embedders/router.py     (new)
    EmbedderRouter
        text:   Embedder | None
        vision: VisionEmbedder | None
        get(modality) -> Embedder | VisionEmbedder | None
```

The router is the singleton replacing `_active`. Lazy-loads each embedder from config the first time it's requested. `get_embedder_for_modality(m)` is the new public API. `get_active_embedder()` becomes an alias returning `router.get("text")` for back-compat with the health endpoint.

### Modality on segments and chunks

`ExtractedSegment` gains:
```python
modality: Literal["text", "vision"] = "text"
image_data: bytes | None = None    # raw bytes when modality="vision"
```

The `chunks` table gains:
```sql
modality TEXT NOT NULL DEFAULT 'text'
image_blob BLOB                     -- nullable; only for vision chunks
```

The `embedding_meta` table gains:
```sql
modality TEXT NOT NULL DEFAULT 'text'
```

For vision chunks: `text` column stores a synthetic descriptor like `"<image: page=3 file=foo.pdf>"` so FTS and search results have something to display. The actual embedding input is `image_blob`, not `text`.

### Vector storage

Two separate `vec0` tables — different embedders produce different dimensions and sqlite-vec requires fixed dim per table:

```sql
CREATE VIRTUAL TABLE vec_text_embeddings   USING vec0(embedding FLOAT[768]);
CREATE VIRTUAL TABLE vec_vision_embeddings USING vec0(embedding FLOAT[3072]);
```

Dimensions are read from the configured embedders at first daemon init and stored in the `meta` table. If a user changes provider to one with a different dim, the daemon refuses to start with a clear error pointing to the `ssearch reembed` command (deferred to Plan 5). For this plan, we hard-code 768 (text) and 3072 (vision) matching the chosen defaults.

### Content-hash dedup (multi-modal)

Already-built dedup logic extends naturally:

- **Text chunks**: `content_hash = sha256(normalize_for_hash(text))` — unchanged.
- **Vision chunks**: `content_hash = sha256(image_bytes)` — hashes the raw rendered PNG bytes.
- The dedup key `(content_hash, provider_id, model_id, dim)` already namespaces per-provider, so text and vision live in independent dedup spaces and can't collide.

File-level mtime+size skip in `Indexer._index_one_file` is unchanged: untouched files are never re-extracted, never re-embedded, regardless of modality.

Verified guarantee: re-indexing the same corpus twice → zero new embedder calls in either pipeline.

## Per-modality config

The `[embedding]` section is split into `[embedding.text]` and `[embedding.vision]`:

```toml
[embedding.text]
preset    = "ollama"
model     = "embeddinggemma"
base_url  = "http://localhost:11434/v1"
batch_size = 128

[embedding.vision]
preset    = "gemini"
model     = "gemini-embedding-2"
# api_key looked up in keychain under provider_id "gemini"
batch_size = 16
```

Either section may be omitted. If `[embedding.vision]` is absent, vision-capable extractors fall back to OCR/text (current behavior). If `[embedding.text]` is absent, the daemon refuses to start (text is mandatory).

A legacy flat `[embedding]` section is migrated transparently: `config.load()` rewrites it to `[embedding.text]` in memory.

## Extractor changes

| Extractor | Today | After |
|-----------|-------|-------|
| Text/HTML/Docx/Xlsx/Epub/Rtf/Email/Notebook | text segments | unchanged |
| Audio (Whisper) | text segments | unchanged |
| **Image** | OCR → text segment | vision segment with raw bytes; OCR fallback only if no vision embedder configured |
| **PDF** | text segments per page | text segments per page **+** vision segments per page (rendered via `pypdfium2`) when vision embedder configured |
| **PPTX** | text segments per slide | text segments per slide; vision rendering deferred (libreoffice dependency too heavy) |

Vision rendering is gated on `router.vision is not None` — extractors check this before producing vision segments to avoid wasted work.

PDF page rendering: 150 DPI PNG, capped at 5 MB per image. Pages over the cap are skipped with a warning.

## New providers (this plan)

- **`embedders/gemini.py`** — `GeminiTextEmbedder(Embedder)` using `models/gemini-embedding-2:embedContent` REST endpoint, returns 3072-d. API key in keychain.
- **`embedders/gemini_vision.py`** — `GeminiVisionEmbedder(VisionEmbedder)` using same REST endpoint with `inline_data` parts (base64-encoded PNG/JPEG), 3072-d.

Existing providers (Local, Ollama, OpenAI, OpenAICompatible) unchanged.

## Worker routing

```python
def drain_once(self):
    rows = fetch_pending_jobs_with_modality()
    by_modality = group_by(rows, key="modality")
    for modality, group in by_modality.items():
        embedder = self.router.get(modality)
        if embedder is None:
            mark_skipped(group, reason="no_embedder_for_modality")
            continue
        # existing dedup + embed flow, parameterized by modality
        process_group(embedder, modality, group)
```

The worker constructor takes `router: EmbedderRouter` instead of `embedder: Embedder`. `health_check` exercises both modalities if both are configured.

## Testing strategy

**Unit tests (fast):** mocked HTTP for Gemini, mocked Ollama for text. Verify routing, dedup keys, modality flow.

**E2E tests (`@pytest.mark.slow @pytest.mark.network`):** real Ollama text + real Gemini vision against a small fixture set:
- 1 markdown file → 1 text chunk, 768-d
- 1 plain PNG → 1 vision chunk, 3072-d
- 1 PDF with text+diagram → N text chunks + N vision chunks (one per page), both tables populated
- Re-index same corpus → zero new embedder calls (dedup verified)

**Real-world smoke test:** `ssearch --index ~/Documents` with text-only first (Ollama embeddinggemma), then again with vision enabled (Gemini). Verify chunk counts, vec table sizes, embedder call counts.

## Migration path

V3 migration:
1. `ALTER TABLE chunks ADD COLUMN modality TEXT NOT NULL DEFAULT 'text'`
2. `ALTER TABLE chunks ADD COLUMN image_blob BLOB`
3. `ALTER TABLE embedding_meta ADD COLUMN modality TEXT NOT NULL DEFAULT 'text'`
4. `CREATE VIRTUAL TABLE vec_vision_embeddings USING vec0(embedding FLOAT[3072])`
5. Existing `vec_embeddings` renamed to `vec_text_embeddings`. (sqlite-vec virtual tables don't support rename — drop+recreate, copy data.)

Existing data stays valid: every chunk gets `modality='text'`, lives in the renamed text vec table.

## Out of scope (for this plan)

- Re-embedding command when switching providers (Plan 5).
- Cross-modal search query interface (Plan 5).
- Local Ollama vision embedder (waits on Ollama VL embedding API to stabilize).
- Audio embedding directly (e.g., CLAP) — Whisper→text remains the path.
- Video extraction.

## Files added

```
semanticsd/embedders/vision_base.py
semanticsd/embedders/router.py
semanticsd/embedders/gemini.py
semanticsd/embedders/gemini_vision.py
tests/test_embedders_vision_base.py
tests/test_embedders_router.py
tests/test_embedders_gemini.py
tests/test_embedders_gemini_vision.py
tests/test_extractors_image_vision.py
tests/test_extractors_pdf_vision.py
tests/test_pipeline_modality.py
tests/test_e2e_multimodal.py
```

## Files modified

```
semanticsd/embedders/base.py             — no behavior change, just docs
semanticsd/embedders/__init__.py         — re-exports + alias
semanticsd/embedders/registry.py         — add gemini text + vision entries
semanticsd/extractors/base.py            — modality + image_data on ExtractedSegment
semanticsd/extractors/image.py           — emit vision segment, OCR fallback
semanticsd/extractors/pdf.py             — emit vision segments per page
semanticsd/pipeline/hasher.py            — bytes hashing helper
semanticsd/pipeline/chunker.py           — bypass for vision segments
semanticsd/pipeline/indexer.py           — handle modality + image_blob
semanticsd/pipeline/worker.py            — router-based routing
semanticsd/db/schema.py                  — V3 migration
semanticsd/db/migrations.py              — V3 entry
semanticsd/config.py                     — split [embedding.text] / [embedding.vision]
semanticsd/server/routes/health.py       — exercise both modalities
requirements.txt                          — pypdfium2
```
