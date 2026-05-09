"""Extension -> Extractor mapping. Filled in as each concrete extractor lands."""
from __future__ import annotations
from pathlib import Path
from typing import Type
from semanticsd.extractors.base import Extractor


EXTENSION_TO_CLASS: dict[str, Type[Extractor]] = {}


def register(extractor_cls: Type[Extractor]) -> Type[Extractor]:
    """Register a class for each of its extensions. Used as a class decorator."""
    for ext in extractor_cls.extensions:
        EXTENSION_TO_CLASS[ext.lower()] = extractor_cls
    return extractor_cls


def get_extractor(path: Path) -> Extractor | None:
    """Return an Extractor instance for the file at `path`, or None if unsupported."""
    ext = path.suffix.lower()
    cls = EXTENSION_TO_CLASS.get(ext)
    return cls() if cls else None
