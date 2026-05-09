import pytest
from semanticsd.extractors.audio import AudioExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_wav_silence


def test_graceful_degrade_when_whisper_fails(tmp_path, monkeypatch):
    p = make_wav_silence(tmp_path)

    class BoomModel:
        def transcribe(self, *a, **kw):
            raise RuntimeError("ffmpeg not found")

    monkeypatch.setattr(
        "semanticsd.extractors.audio.WhisperModel",
        lambda *a, **kw: BoomModel(),
    )
    out = AudioExtractor().extract(p)
    assert out.file_type == "audio"
    assert out.segments == []
    assert "transcribe_error" in out.metadata


def test_registry_picks_wav(tmp_path):
    p = make_wav_silence(tmp_path)
    assert isinstance(registry.get_extractor(p), AudioExtractor)


@pytest.mark.slow
def test_real_transcription(tmp_path):
    """Tiny silent wav — Whisper should return empty or near-empty segments."""
    p = make_wav_silence(tmp_path, duration_s=0.5)
    out = AudioExtractor(model_size="tiny").extract(p)
    assert out.file_type == "audio"
    assert all(not s.text.strip() for s in out.segments) or out.segments == []
