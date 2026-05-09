import plistlib
from pathlib import Path
from semanticsd.admin import launchd


def test_render_returns_valid_plist(tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\necho test\n")
    py.chmod(0o755)
    text = launchd.render_plist(python_executable=str(py), package_dir=str(tmp_path))
    parsed = plistlib.loads(text.encode())
    assert parsed["Label"] == "com.semanticsd"
    assert parsed["RunAtLoad"] is True
    assert parsed["KeepAlive"] is True
    # Full list assertion (corrected from the plan's buggy slice assertion).
    # The plan said [-2:] == ["-m", "semanticsd"] but the spec produces
    # [python, "-m", "semanticsd", "serve"], so [-2:] would be
    # ["semanticsd", "serve"], not ["-m", "semanticsd"]. We assert the full
    # list instead, which is both stronger and unambiguous.
    assert parsed["ProgramArguments"] == [str(py), "-m", "semanticsd", "serve"]


def test_write_plist_to_path(tmp_path):
    plist_path = tmp_path / "com.semanticsd.plist"
    launchd.write_plist(plist_path, python_executable="/usr/bin/python3", package_dir=str(tmp_path))
    assert plist_path.exists()
    parsed = plistlib.loads(plist_path.read_bytes())
    assert parsed["Label"] == "com.semanticsd"
