"""Per-file-type RRF weight profiles.

Different file kinds want different mode emphasis:
- Code: identifier-exact match (grep) matters more than embedding similarity.
- Prose: semantic similarity dominates; grep is helpful but secondary.
- Data/config: filename + grep are most useful; semantic of structured
  data tends to be noisy.
- Image: only vision-semantic and filename are meaningful — grep on the
  synthetic "<image: ...>" descriptor is dead-air.

Weights are applied to the per-mode RRF contribution:
    score(d) = Σ_i  weight(file_type, mode_i) / (k + rank_i + 1)

A weight of 0 means "exclude this mode for this file type" — used to
remove filename-FTS-driven noise on prose where the path tokens are
unrelated to query intent.
"""
from __future__ import annotations
from typing import Literal

Mode = Literal["semantic", "grep", "filename", "vision"]


# (file_type, file_class) — see _FILE_CLASS_BY_EXT below for the extension map.
_PROFILES: dict[str, dict[str, float]] = {
    "code":     {"semantic": 0.6, "grep": 2.0, "filename": 1.5, "vision": 0.0},
    "prose":    {"semantic": 1.5, "grep": 0.7, "filename": 1.0, "vision": 1.0},
    "data":     {"semantic": 0.5, "grep": 0.7, "filename": 1.5, "vision": 0.0},
    "notebook": {"semantic": 1.2, "grep": 1.0, "filename": 1.2, "vision": 0.5},
    "image":    {"semantic": 0.0, "grep": 0.0, "filename": 1.5, "vision": 2.0},
    "email":    {"semantic": 1.2, "grep": 1.5, "filename": 1.0, "vision": 0.0},
    "html":     {"semantic": 1.2, "grep": 1.0, "filename": 1.0, "vision": 0.0},
    "audio":    {"semantic": 1.5, "grep": 0.7, "filename": 1.0, "vision": 0.0},
    "default":  {"semantic": 1.0, "grep": 1.0, "filename": 1.0, "vision": 1.0},
}

# Map file_type strings (set by extractors) to a class name. Anything not
# listed maps to "default".
_FILE_CLASS_BY_TYPE: dict[str, str] = {
    "text":     "code",      # TextExtractor handles both prose-y .txt/.md AND code
    "html":     "html",
    "pdf":      "prose",
    "docx":     "prose",
    "xlsx":     "data",
    "pptx":     "prose",
    "epub":     "prose",
    "rtf":      "prose",
    "email":    "email",
    "notebook": "notebook",
    "image":    "image",
    "audio":    "audio",
}

# Sub-classify by extension for the catch-all "text" file_type so that
# code (.py/.rs/.ts/...) gets code weights, while .md/.txt/.rst get prose.
_PROSE_EXTS = {".md", ".txt", ".rst", ".org", ".tex"}
_DATA_EXTS = {".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".xml", ".ini", ".conf"}


def file_class_for(file_type: str | None, path: str | None = None) -> str:
    """Pick a profile class for a result. file_type wins; path extension
    refines the catch-all 'text' kind."""
    if not file_type:
        return "default"
    base = _FILE_CLASS_BY_TYPE.get(file_type, "default")
    if base == "code" and path:
        # TextExtractor covers .md/.json/.py/etc. — sub-classify.
        i = path.rfind(".")
        if i >= 0:
            ext = path[i:].lower()
            if ext in _PROSE_EXTS:
                return "prose"
            if ext in _DATA_EXTS:
                return "data"
    return base


def weight_for(file_class: str, mode: str) -> float:
    """Return the RRF multiplier for a (class, mode) pair. Falls back to default."""
    profile = _PROFILES.get(file_class, _PROFILES["default"])
    return float(profile.get(mode, 1.0))
