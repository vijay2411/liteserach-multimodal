# Search Engine for SemanticsD

**Status:** approved, pending implementation
**Date:** 2026-05-09
**Builds on:** `2026-05-09-semanticsd-design.md` (Search Engine section), `2026-05-09-multimodal-embedding-design.md` (per-modality vec tables)

## Goal

Make the daemon useful end-to-end: turn `ssearch <query>` into a working search command that combines semantic similarity, filename matching, and grep-style content search across the indexed corpus, with cross-modal results when a vision embedder is configured.

## User-facing surface

```
ssearch "neural network architecture"            # default: hybrid, cwd-scoped
ssearch "design.md"          --filename          # filename FTS only
ssearch "TODO.*urgent"       --grep              # chunk text FTS only
ssearch "red car screenshot" --semantic          # semantic only
ssearch "neural networks"    --all               # whole corpus, not just cwd
ssearch "..."                --limit 5           # top-N
ssearch "..."                --no-vision         # disable cross-modal
ssearch "..."                --json              # machine-readable output
```

The default mode is **hybrid** — combines semantic + filename + grep with Reciprocal Rank Fusion. Default scope is **cwd-relative** (only files under the current working directory). Default cross-modal is **always-on when a vision embedder is configured**.

### Output format (default: rich)

```
docs/architecture.md  (semantic, score=0.91)
  ## Neural Network Architecture
  We use a transformer encoder with 12 layers and 768-d hidden state...

screenshots/diagram.png  (vision, score=0.87)
  <image: diagram.png>  page=1

src/model.py:42  (grep, score=0.74)
  class TransformerBlock(nn.Module):
      """A single transformer block with self-attention."""
```

`--json` outputs structured records: `{path, modality, mode, score, snippet, byte_start, byte_end, metadata}`.

## Architecture

### Package layout

```
semanticsd/search/
    __init__.py              — public API: search(query, opts) -> list[SearchResult]
    engine.py                — Engine class, orchestrates modes + fusion
    semantic.py              — vector similarity over vec_text/vec_vision tables
    filename.py              — FTS over fts_paths
    grep.py                  — FTS over fts_chunks
    fusion.py                — reciprocal rank fusion
    snippets.py              — text snippet extraction with query terms highlighted
    types.py                 — SearchResult, SearchOptions models
```

### `SearchResult` model

```python
class SearchResult(BaseModel):
    path: str               # absolute file path
    modality: Literal["text", "vision"]
    mode: Literal["semantic", "filename", "grep", "hybrid"]
    score: float            # 0.0..1.0 (or fused RRF score)
    snippet: str | None     # text segment or "<image: ...>" descriptor
    byte_start: int | None
    byte_end: int | None
    chunk_id: int
    file_id: int
    metadata: dict          # file_type, page, slide, etc.
```

### Per-mode flow

**Semantic (text):**
1. Embed `query` with `router.text` (single embed call, kind="query")
2. `SELECT chunk_id, distance FROM vec_text_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT k`
3. JOIN to `chunks` + `files` to fetch path/snippet/metadata
4. Apply CWD filter (`WHERE files.path LIKE :cwd_prefix`) if not `--all`

**Semantic (vision, cross-modal):**
1. If `router.vision` is None or `--no-vision`, skip
2. Embed `query` with `router.vision` — most vision embedders (Gemini 2, Qwen3-VL) accept text input and produce vectors in a shared text+image space
3. For each vision-vec table (`vec_vision_embeddings`, `vec_vision_embeddings_<dim>`):
   - Skip tables whose dim doesn't match `router.vision.dim`
   - Run `MATCH ? LIMIT k`
4. JOIN, filter, return as `SearchResult(modality="vision")`

**Filename FTS:**
- `SELECT rowid FROM fts_paths WHERE fts_paths MATCH ? ORDER BY rank LIMIT k`
- JOIN to `files`

**Grep FTS:**
- `SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT k`
- JOIN to `chunks` + `files`
- Vision chunks (modality='vision') are excluded — their `text` field is just a descriptor, grep on it is noise.

**FTS index population:** `fts_chunks` and `fts_paths` are FTS5 contentless tables. Currently the indexer doesn't populate them. Plan 5 adds inserts to these tables in the indexer alongside chunk inserts.

### Reciprocal Rank Fusion

For hybrid mode, each of (semantic-text, semantic-vision, filename, grep) returns ranked results. RRF combines them:

```python
def rrf(rank_lists: list[list[ChunkID]], k: int = 60) -> list[tuple[ChunkID, float]]:
    scores: dict[ChunkID, float] = defaultdict(float)
    for ranks in rank_lists:
        for i, cid in enumerate(ranks):
            scores[cid] += 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

The fused score replaces the per-mode score in `SearchResult`. The `mode` field is set to "hybrid" with a `metadata.contributing_modes` list explaining which modes ranked it.

### CWD scoping

`SearchOptions(cwd: Path | None = None, all: bool = False)`:
- `all=True` → no path filter
- `cwd=None, all=False` → resolve `Path.cwd()` and filter `files.path LIKE f"{cwd}/%"`
- Filter applied after the per-mode top-K — we over-fetch (`k * 3`) at the SQL level so cwd-filtering doesn't starve.

## HTTP API

```
GET /v1/search?q=<query>&mode=hybrid&limit=20&cwd=/abs/path&all=false&vision=true
Response: {"results": [SearchResult, ...], "took_ms": 42}
```

The CLI (`ssearch`) uses this endpoint, passing `cwd=Path.cwd()` automatically unless `--all`.

## Caching

Per-query embedding cost matters: every search calls the embedder(s) once. We add a small LRU cache keyed by `(query, embedder.provider_id, embedder.model_id)` with capacity 256 in-memory inside the search engine. No persistent cache for now.

## Tests

**Unit (mocked):**
- `test_search_semantic`: fake embedder + fake vec0 result rows → verifies SQL + JOIN
- `test_search_filename`: seeded FTS → match
- `test_search_grep`: seeded FTS + ensure vision chunks excluded
- `test_rrf`: known input rankings → expected fused order
- `test_cwd_filter`: results outside cwd are filtered
- `test_cross_modal_skipped_when_dim_mismatch`: vision tables at wrong dim are not queried

**E2E (slow, real Ollama + Gemini):**
- Index small fixture corpus, query for known content, assert top result is the expected file
- Cross-modal: query "red square" → vision result for the red PNG fixture beats text results

**Real-world smoke:**
- Reuse `sandbox-docs/` corpus, run `ssearch "bank statement"` → expect AccountStatement_*.pdf in top 3 (after deferring its decryption issue)

## Out of scope

- Reranker (cross-encoder) — Phase 2.
- Tree-sitter chunking for code-aware search — Phase 2.
- Search history / saved queries — later.
- Result deduplication across overlapping chunks (sliding window overlap means adjacent chunks may both match) — initial implementation does naive top-K; can add chunk-merge later.
- Re-embedding when provider changes — Plan 8.

## Files added

```
semanticsd/search/__init__.py
semanticsd/search/engine.py
semanticsd/search/semantic.py
semanticsd/search/filename.py
semanticsd/search/grep.py
semanticsd/search/fusion.py
semanticsd/search/snippets.py
semanticsd/search/types.py
semanticsd/server/routes/search.py
tests/test_search_types.py
tests/test_search_semantic.py
tests/test_search_filename.py
tests/test_search_grep.py
tests/test_search_fusion.py
tests/test_search_engine.py
tests/test_server_search.py
tests/test_e2e_search.py
```

## Files modified

```
semanticsd/cli.py                   — replace `ssearch <q>` CLI body with real search
semanticsd/server/app.py            — register search router
semanticsd/pipeline/indexer.py      — populate fts_chunks + fts_paths on insert
```
