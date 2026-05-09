import os
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def tmp_app_support(monkeypatch) -> Path:
    """Redirect SemanticsD's Application Support dir to a tmp path for the test."""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("SEMANTICSD_HOME", d)
        yield Path(d)
