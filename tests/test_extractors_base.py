"""ExtractedSegment + ExtractedDoc model tests."""
from semanticsd.extractors.base import ExtractedSegment, ExtractedDoc


def test_segment_default_modality_is_text():
    s = ExtractedSegment(text="hello", byte_start=0, byte_end=5)
    assert s.modality == "text"
    assert s.image_data is None


def test_segment_vision_with_bytes():
    s = ExtractedSegment(
        text="<image: page=1>",
        byte_start=0,
        byte_end=15,
        modality="vision",
        image_data=b"\x89PNG\r\n",
    )
    assert s.modality == "vision"
    assert s.image_data == b"\x89PNG\r\n"


def test_doc_holds_segments():
    d = ExtractedDoc(
        path="/x", file_type="text",
        segments=[ExtractedSegment(text="a", byte_start=0, byte_end=1)],
    )
    assert len(d.segments) == 1
