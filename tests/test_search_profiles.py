"""Per-file-type RRF weight profiles."""
from semanticsd.search.profiles import file_class_for, weight_for


def test_code_extension_classified_as_code():
    assert file_class_for("text", "/x/foo.py") == "code"
    assert file_class_for("text", "/x/foo.rs") == "code"
    assert file_class_for("text", "/x/foo.ts") == "code"


def test_prose_extension_overrides_text_default():
    # TextExtractor produces file_type='text' for both code and prose;
    # the path extension routes to the prose profile.
    assert file_class_for("text", "/x/notes.md") == "prose"
    assert file_class_for("text", "/x/notes.txt") == "prose"


def test_data_extension_overrides_text_default():
    assert file_class_for("text", "/x/conf.json") == "data"
    assert file_class_for("text", "/x/conf.yaml") == "data"


def test_pdf_is_prose():
    assert file_class_for("pdf", "/x/foo.pdf") == "prose"


def test_image_is_image():
    assert file_class_for("image", "/x/foo.png") == "image"


def test_unknown_falls_back_to_default():
    assert file_class_for(None) == "default"
    assert file_class_for("totally-unknown-type") == "default"


def test_code_grep_weight_higher_than_semantic():
    assert weight_for("code", "grep") > weight_for("code", "semantic")


def test_prose_semantic_weight_higher_than_grep():
    assert weight_for("prose", "semantic") > weight_for("prose", "grep")


def test_image_class_excludes_grep_and_semantic():
    assert weight_for("image", "grep") == 0.0
    assert weight_for("image", "semantic") == 0.0
    # Vision still applies
    assert weight_for("image", "vision") > 0.0


def test_default_profile_is_neutral():
    for mode in ("semantic", "grep", "filename", "vision"):
        assert weight_for("default", mode) == 1.0
