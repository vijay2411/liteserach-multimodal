"""Sliding-window chunker. Token count approximated by chars/4 to stay
provider-agnostic — exact tiktoken-based counting can be plugged in later."""
from __future__ import annotations
from dataclasses import dataclass


CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    text: str
    byte_start: int   # absolute, includes base_byte_offset
    byte_end: int


class SlidingWindowChunker:
    def __init__(self, window_tokens: int = 512, overlap_tokens: int = 64):
        self.window_chars = window_tokens * CHARS_PER_TOKEN
        self.overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    def chunk(self, text: str, base_byte_offset: int) -> list[Chunk]:
        text = text or ""
        if not text.strip():
            return []
        n = len(text)
        step = max(1, self.window_chars - self.overlap_chars)
        chunks: list[Chunk] = []
        start = 0
        while start < n:
            end = min(n, start + self.window_chars)
            piece = text[start:end]
            byte_start = base_byte_offset + len(text[:start].encode("utf-8"))
            byte_end = base_byte_offset + len(text[:end].encode("utf-8"))
            chunks.append(Chunk(text=piece, byte_start=byte_start, byte_end=byte_end))
            if end == n:
                break
            start += step
        return chunks
