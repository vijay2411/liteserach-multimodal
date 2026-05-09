"""Structured logging for the daemon."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from semanticsd import paths


def configure(level: str = "info", to_file: bool = True) -> None:
    """Configure root logger. Idempotent."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Avoid duplicate handlers if called twice.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(fmt)
    root.addHandler(stderr)

    if to_file:
        paths.ensure_dirs()
        fh = logging.FileHandler(paths.logs_dir() / "semanticsd.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)
