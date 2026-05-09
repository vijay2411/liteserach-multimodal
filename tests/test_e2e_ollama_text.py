"""Real-Ollama smoke test — requires `ollama serve` with embeddinggemma pulled."""
import socket
import pytest
from semanticsd.embedders.ollama import OllamaEmbedder


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


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _ollama_up(), reason="ollama not running"),
]


def test_ollama_embeddinggemma_real():
    e = OllamaEmbedder(model="embeddinggemma")
    out = e.embed(["semantic search test"], kind="doc")
    assert len(out.vectors) == 1
    assert len(out.vectors[0]) == 768
