"""Schema DDL.

Plan 1 created the relational tables. Plan 2 adds vec_embeddings (sqlite-vec
virtual table) at dim=384 to match the default LocalEmbedder
(BAAI/bge-small-en-v1.5).
"""

SCHEMA_VERSION = 2

DDL_V1 = [
    """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE NOT NULL,
        modified_at INTEGER NOT NULL,
        size INTEGER NOT NULL,
        file_type TEXT NOT NULL,
        indexed_at INTEGER,
        last_error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        byte_start INTEGER NOT NULL,
        byte_end INTEGER NOT NULL,
        UNIQUE(file_id, chunk_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash)",
    """
    CREATE TABLE IF NOT EXISTS embedding_meta (
        chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
        provider_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        dim INTEGER NOT NULL,
        content_hash TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_emb_meta_hash ON embedding_meta(content_hash, provider_id, model_id, dim)",
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY,
        chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
    """
    CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY,
        timestamp INTEGER NOT NULL,
        provider_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        operation TEXT NOT NULL,
        input_tokens INTEGER NOT NULL,
        cost_usd REAL NOT NULL,
        chunk_count INTEGER NOT NULL,
        duration_ms INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_usage_time ON usage(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(provider_id, model_id, timestamp)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
        text, content='chunks', content_rowid='id'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_paths USING fts5(
        path, content='files', content_rowid='id'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
]

DDL_V2 = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
        embedding FLOAT[384]
    )
    """,
]
