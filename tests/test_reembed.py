"""queue_reembed function tests."""
from semanticsd.db import connection, migrations
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.vision_base import VisionEmbedder
from semanticsd.embedders.router import EmbedderRouter
from semanticsd.reembed import queue_reembed


class _Text(Embedder):
    provider_id = "ollama"; model_id = "embeddinggemma"; dim = 768
    supports_kind = False; cost_per_million_input_tokens_usd = 0.0
    def embed(self, texts, kind):
        return EmbedResult(vectors=[[0.0]*768 for _ in texts], input_tokens=1)
    def health_check(self): return (True, "ok")
    def estimate_tokens(self, texts): return 1


class _Vision(VisionEmbedder):
    provider_id = "qwen3_vl_local"; model_id = "Qwen/Qwen3-VL-Embedding-2B"; dim = 2048
    cost_per_million_image_tokens_usd = 0.0
    def embed_images(self, images, kind="doc"):
        return EmbedResult(vectors=[[0.0]*2048 for _ in images], input_tokens=1)
    def health_check(self): return (True, "ok")
    def estimate_image_tokens(self, images): return 1


def _seed(conn, n_text=3, n_vision=2, *, with_old_text_embedding_for_first=True,
          old_provider="local", old_model="bge-small", old_dim=384):
    """Seed N text + M vision chunks. Optionally give the first text chunk
    an embedding row from a different (old) provider — to simulate the
    'switched providers' scenario."""
    conn.execute(
        "INSERT INTO files(path, modified_at, size, file_type, indexed_at) "
        "VALUES('/x', 1, 1, 'text', 1)"
    )
    fid = int(conn.execute("SELECT id FROM files").fetchone()[0])
    text_chunk_ids = []
    for i in range(n_text):
        conn.execute(
            "INSERT INTO chunks(file_id, chunk_index, text, content_hash, "
            "byte_start, byte_end, modality) "
            "VALUES (?, ?, ?, ?, 0, ?, 'text')",
            (fid, i, f"text content {i}", f"h_text_{i}", 32),
        )
        text_chunk_ids.append(int(
            conn.execute("SELECT id FROM chunks WHERE chunk_index = ?", (i,)).fetchone()[0]
        ))
    vision_chunk_ids = []
    for i in range(n_vision):
        conn.execute(
            "INSERT INTO chunks(file_id, chunk_index, text, content_hash, "
            "byte_start, byte_end, modality, image_blob) "
            "VALUES (?, ?, ?, ?, 0, ?, 'vision', ?)",
            (fid, n_text + i, f"<image: {i}>", f"h_vis_{i}", 16, b"\x89PNG"),
        )
        vision_chunk_ids.append(int(
            conn.execute("SELECT id FROM chunks WHERE chunk_index = ?",
                         (n_text + i,)).fetchone()[0]
        ))
    if with_old_text_embedding_for_first and text_chunk_ids:
        conn.execute(
            "INSERT INTO embedding_meta(chunk_id, provider_id, model_id, dim, "
            "content_hash, modality) VALUES (?, ?, ?, ?, ?, 'text')",
            (text_chunk_ids[0], old_provider, old_model, old_dim, "old"),
        )
    return text_chunk_ids, vision_chunk_ids


def test_reembed_queues_chunks_lacking_current_embedding(tmp_path):
    db = tmp_path / "r.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn, n_text=3, n_vision=2)

    router = EmbedderRouter(text=_Text(), vision=_Vision())
    counts = queue_reembed(conn, router, modality="all")
    # Every chunk lacks the new (provider, model, dim) combo, so all queued.
    assert counts["text"] == 3
    assert counts["vision"] == 2

    n_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert n_jobs == 5


def test_reembed_skips_chunks_already_embedded_by_current(tmp_path):
    db = tmp_path / "r.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    # Seed without an old-provider preexisting row so chunk 0 is free for
    # us to claim with the current embedder.
    text_ids, _ = _seed(conn, n_text=3, n_vision=0,
                         with_old_text_embedding_for_first=False)
    # Mark chunk 0 as already embedded by the CURRENT text embedder.
    em = _Text()
    conn.execute(
        "INSERT INTO embedding_meta(chunk_id, provider_id, model_id, dim, "
        "content_hash, modality) VALUES (?, ?, ?, ?, ?, 'text')",
        (text_ids[0], em.provider_id, em.model_id, em.dim, "h"),
    )
    router = EmbedderRouter(text=em)
    counts = queue_reembed(conn, router, modality="text")
    # Two chunks lack current embedding (chunk 1 and 2)
    assert counts["text"] == 2


def test_reembed_modality_filter(tmp_path):
    db = tmp_path / "r.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn, n_text=3, n_vision=2)
    router = EmbedderRouter(text=_Text(), vision=_Vision())

    # Only vision
    counts = queue_reembed(conn, router, modality="vision")
    assert counts == {"text": 0, "vision": 2}

    # Re-running same modality is idempotent for the per-chunk delta — but
    # creates new jobs (worker will dedup via content-hash). The function
    # spec only guarantees "queue jobs for chunks lacking current embedding";
    # we don't claim it dedups against pending jobs. So second call:
    counts2 = queue_reembed(conn, router, modality="vision")
    assert counts2["vision"] == 2  # same 2 chunks queued again


def test_reembed_no_text_embedder_returns_zero(tmp_path):
    db = tmp_path / "r.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    _seed(conn, n_text=3, n_vision=0)
    router = EmbedderRouter(text=None, vision=None)
    counts = queue_reembed(conn, router, modality="all")
    assert counts == {"text": 0, "vision": 0}
