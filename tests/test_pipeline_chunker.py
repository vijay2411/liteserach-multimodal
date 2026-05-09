from semanticsd.pipeline.chunker import SlidingWindowChunker, Chunk


def test_short_text_returns_one_chunk():
    c = SlidingWindowChunker(window_tokens=512, overlap_tokens=64)
    chunks = c.chunk("Hello world.", base_byte_offset=0)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].byte_start == 0
    assert chunks[0].byte_end == len("Hello world.".encode("utf-8"))


def test_long_text_splits_into_overlapping_windows():
    body = ("word " * 10000).strip()
    c = SlidingWindowChunker(window_tokens=512, overlap_tokens=64)
    chunks = c.chunk(body, base_byte_offset=0)
    assert len(chunks) > 1
    for ch in chunks:
        assert len(ch.text) // 4 <= 512 + 64
    assert chunks[0].byte_end > chunks[1].byte_start


def test_byte_offsets_continuous():
    body = "First part. Second part. Third part."
    c = SlidingWindowChunker(window_tokens=2, overlap_tokens=0)
    chunks = c.chunk(body, base_byte_offset=100)
    assert chunks[0].byte_start == 100
    for ch in chunks:
        assert ch.byte_start >= 100
        assert ch.byte_end > ch.byte_start


def test_empty_text_returns_no_chunks():
    c = SlidingWindowChunker()
    assert c.chunk("", base_byte_offset=0) == []
    assert c.chunk("   \n\t", base_byte_offset=0) == []
