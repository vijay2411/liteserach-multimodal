"""End-to-end: real Ollama (text) + real Gemini (vision) pipeline against
a fixture corpus of mixed file types."""
import os
import pathlib
import socket
import pytest


def _ollama_up() -> bool:
    s = socket.socket()
    try:
        s.settimeout(0.5)
        s.connect(("localhost", 11434))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _gem_key():
    p = pathlib.Path.home() / "secrets" / "gemini_api_key"
    return p.read_text().strip() if p.exists() else os.environ.get("GEMINI_API_KEY")


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_ollama_up() and _gem_key()),
        reason="needs ollama running + gemini key",
    ),
]


def test_multimodal_e2e(tmp_path):
    from semanticsd.db import connection, migrations
    from semanticsd.embedders.router import EmbedderRouter
    from semanticsd.embedders.ollama import OllamaEmbedder
    from semanticsd.embedders.gemini_vision import GeminiVisionEmbedder
    from semanticsd.pipeline.indexer import Indexer
    from semanticsd.pipeline.worker import Worker
    from tests._fixtures import make_image_with_text, make_pdf, make_text

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    make_text(corpus, name="notes.md", body="# Notes\nSemantic search is great.")
    make_pdf(corpus)
    make_image_with_text(corpus, text="Hello vision world")

    db = tmp_path / "mm.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    text_em = OllamaEmbedder(model="embeddinggemma")
    vis_em = GeminiVisionEmbedder(api_key=_gem_key())
    router = EmbedderRouter(text=text_em, vision=vis_em)

    indexer = Indexer(conn=conn, max_file_size_mb=50)
    stats = indexer.index_path(corpus)
    assert stats["files_indexed"] >= 3

    worker = Worker(conn=conn, router=router, batch_size=8)
    while worker.drain_once() > 0:
        pass

    n_text = conn.execute("SELECT COUNT(*) FROM vec_text_embeddings").fetchone()[0]
    n_vision = conn.execute("SELECT COUNT(*) FROM vec_vision_embeddings").fetchone()[0]
    assert n_text > 0, "expected text embeddings"
    assert n_vision > 0, "expected vision embeddings (PDF pages + image)"

    pending = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0]
    failed = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='failed'").fetchone()[0]
    assert pending == 0, f"unexpected pending: {pending}"
    assert failed == 0, f"unexpected failed: {failed}"

    chunks_before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    indexer.index_path(corpus)
    chunks_after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert chunks_before == chunks_after, "re-index should not duplicate chunks"
