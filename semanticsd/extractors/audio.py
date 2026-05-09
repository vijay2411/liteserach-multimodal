"""Audio extractor — faster-whisper transcription. Graceful degrade if model
fails to load or ffmpeg is missing."""
from __future__ import annotations
import logging
from pathlib import Path
from faster_whisper import WhisperModel
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register

log = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"


@register
class AudioExtractor(Extractor):
    file_type = "audio"
    extensions = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".mp4a")

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE):
        self.model_size = model_size
        self._model: WhisperModel | None = None

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def extract(self, path: Path) -> ExtractedDoc:
        try:
            model = self._ensure_model()
            segments_iter, _info = model.transcribe(str(path), beam_size=1)
            wsegs = list(segments_iter)
        except (RuntimeError, OSError, FileNotFoundError) as e:
            log.warning("Whisper transcription failed for %s: %s", path, e)
            return ExtractedDoc(
                path=str(path),
                file_type=self.file_type,
                segments=[],
                metadata={"transcribe_error": str(e)},
            )

        out_segments: list[ExtractedSegment] = []
        cursor = 0
        for s in wsegs:
            text = (s.text or "").strip()
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            out_segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"start_s": s.start, "end_s": s.end},
            ))
            cursor += seg_bytes + 1
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=out_segments,
        )
