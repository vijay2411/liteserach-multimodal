# Cost Tracking + Reembed Command

**Status:** approved, pending implementation
**Date:** 2026-05-09
**Builds on:** all prior plans

Two small features that share infrastructure (both touch the worker's
embedding pathway):

1. **Cost tracking + budget enforcement.** Every embedder call writes a
   `usage` row. CLI/HTTP report aggregate spend. A configurable budget
   cap fail-closes the worker when exceeded.

2. **Reembed command.** Switch text/vision providers in config; one
   command queues re-embedding jobs for any chunks not yet covered by
   the new (provider, model, dim).

## Plan 7: Cost tracking + budget enforcement

### Existing infrastructure
The Plan 1 schema already created the `usage` table:
```sql
CREATE TABLE usage (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    operation TEXT NOT NULL,        -- "text_embed" | "vision_embed" | "query_embed"
    input_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    chunk_count INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL
);
```
Plus indexes on `timestamp` and `(provider_id, model_id, timestamp)`. Nothing
writes to it yet — that's the gap.

### What we add

**Worker writes usage rows.** After every successful embed batch, insert one
row capturing provider_id, model_id, operation, input_tokens, cost_usd
(`tokens / 1e6 * provider.cost_per_million_input_tokens_usd`), chunk_count,
and duration_ms (wall-clock for the embed call only, not surrounding I/O).

The same applies on the search-engine side for query-time embeddings — we
wrap `Embedder.embed([query], kind="query")` and `VisionEmbedder.embed_images`
calls so the cost of a search is also tracked.

**Budget config:**
```toml
[budget]
monthly_limit_usd = 0.0          # 0 = unlimited
warning_threshold = 0.8          # log at 80% of cap
```

**Budget enforcement.** Before each embed batch, the worker queries:
```sql
SELECT COALESCE(SUM(cost_usd), 0) FROM usage
WHERE timestamp >= :start_of_window
```
For monthly windows, `start_of_window` = first second of the current calendar
month. If `total >= monthly_limit_usd` (and `monthly_limit_usd > 0`), the
worker refuses, marks the batch's jobs as `failed` with reason
`budget_exceeded`, and logs a warning. Local providers
(`cost_per_million_input_tokens_usd == 0`) bypass the check entirely.

A separate threshold-warn log fires at the configured `warning_threshold`
once per process lifetime to avoid log spam.

**Surfaces:**
- `GET /v1/usage` → totals + breakdown (defaults: this month, group by provider+model)
  Query params: `since=YYYY-MM-DD`, `until=YYYY-MM-DD`, `provider=...`
- `ssearch usage` → CLI table; `--this-month` / `--today` / `--all` flags
- `ssearch usage --csv` for piping
- Health endpoint adds a `budget` block: `{spent_this_month_usd, limit_usd, percent_used}`

### Worker integration sketch

```python
# semanticsd/usage/recorder.py  (new)
@dataclass
class UsageEvent:
    provider_id: str
    model_id: str
    operation: str
    input_tokens: int
    chunk_count: int
    duration_ms: int
    cost_usd: float

def record_usage(conn, event: UsageEvent) -> None:
    conn.execute(
        "INSERT INTO usage(timestamp, provider_id, model_id, operation, "
        "input_tokens, cost_usd, chunk_count, duration_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (int(time.time()), event.provider_id, event.model_id, event.operation,
         event.input_tokens, event.cost_usd, event.chunk_count, event.duration_ms),
    )
```

```python
# in worker._process_group:
t0 = time.monotonic()
result = embedder.embed_or_embed_images(...)
dt_ms = int((time.monotonic() - t0) * 1000)
cost = (result.input_tokens / 1e6) * embedder.cost_per_million_input_tokens_usd
record_usage(self.conn, UsageEvent(
    provider_id=embedder.provider_id, model_id=embedder.model_id,
    operation=f"{modality}_embed",
    input_tokens=result.input_tokens, chunk_count=len(to_embed),
    duration_ms=dt_ms, cost_usd=cost,
))
```

## Plan 8: Reembed command

### Use case

User has a corpus indexed with text embedder A (e.g. local bge 384-d). They
edit config to use embedder B (e.g. ollama embeddinggemma 768-d). On daemon
restart, only NEW files get B-embeddings; existing chunks still have A's
vectors. `ssearch reembed text` queues all chunks lacking a B embedding for
re-processing.

Vision case: user switches from `gemini` (3072-d cloud) to `qwen3_vl_local`
(2048-d MPS). The per-(modality, dim) vec tables already coexist, but no
existing vision chunks have a Qwen3-VL embedding. `ssearch reembed vision`
queues them.

### Identifying chunks needing re-embed

```sql
-- Chunks that DO NOT yet have an embedding from (current_provider, current_model, current_dim)
SELECT c.id
FROM chunks c
WHERE c.modality = :modality
  AND NOT EXISTS (
    SELECT 1 FROM embedding_meta em
    WHERE em.chunk_id = c.id
      AND em.provider_id = :current_provider_id
      AND em.model_id    = :current_model_id
      AND em.dim         = :current_dim
  )
```

For each such chunk, insert a fresh `jobs` row with `status='pending'`. The
existing worker pipeline picks them up using the current router. The
content-hash dedup logic in `worker._process_group` will copy from any
matching cached embedding — meaning if you run reembed twice, the second
call is a no-op, and if a chunk's content matches another already-reembedded
chunk, the vector is reused.

### Surfaces

- `POST /v1/reembed` body: `{"modality": "text" | "vision" | "all"}` →
  returns `{queued: N}`
- `ssearch reembed [text|vision|all]` → triggers + reports

### What we DON'T do (deferred)

- **Garbage-collect old vectors.** Old (provider, model, dim) embeddings stay
  in `embedding_meta` and their vec tables. They cost a few MB per thousand
  chunks. Cheaper than re-embedding if the user rolls back. We can add
  `ssearch gc` later.
- **Per-file reembed.** `ssearch reembed --path X` could be added but isn't
  needed yet — full corpus is fine.

## Files added

```
semanticsd/usage/__init__.py
semanticsd/usage/recorder.py            — UsageEvent + record_usage
semanticsd/usage/budget.py              — month-window aggregator + cap check
semanticsd/usage/reports.py             — aggregation queries used by HTTP/CLI
semanticsd/server/routes/usage.py
semanticsd/server/routes/reembed.py
semanticsd/reembed.py                   — queue_reembed(conn, router, modality)
tests/test_usage_recorder.py
tests/test_usage_budget.py
tests/test_usage_reports.py
tests/test_reembed.py
tests/test_server_usage.py
tests/test_server_reembed.py
tests/test_e2e_usage_and_reembed.py
```

## Files modified

```
semanticsd/config.py            — BudgetConfig section
semanticsd/pipeline/worker.py   — record usage + budget gate
semanticsd/search/engine.py     — record query-time usage
semanticsd/server/routes/health.py  — add budget block
semanticsd/cli.py               — `ssearch usage`, `ssearch reembed`
```
