"""sweep_directories + battery probe tests."""
from pathlib import Path
from unittest.mock import patch
from semanticsd.watcher.sweep import sweep_directories
from semanticsd.watcher.battery import power_source, is_on_battery


# --- sweep ---

class _FakeIndexer:
    def __init__(self):
        self.calls = []
        self._counter = 0

    def index_path(self, p):
        self.calls.append(p)
        self._counter += 1
        return {
            "files_indexed": self._counter,
            "files_skipped_unsupported": 0,
            "files_skipped_unchanged": 0,
            "chunks_created": self._counter * 2,
            "jobs_queued": self._counter * 2,
        }


def test_sweep_iterates_existing_directories(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    idx = _FakeIndexer()
    out = sweep_directories(idx, [a, b])
    assert out["directories"] == 2
    assert out["files_indexed"] == 1 + 2  # 1 then 2
    assert out["chunks_created"] == 2 + 4
    assert "elapsed_s" in out
    assert idx.calls == [a, b]


def test_sweep_skips_missing_directory(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    idx = _FakeIndexer()
    out = sweep_directories(idx, [a, tmp_path / "no-such-dir"])
    assert out["directories"] == 1
    assert idx.calls == [a]


def test_sweep_continues_on_indexer_failure(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()

    class _BoomIndexer:
        def __init__(self):
            self.n = 0
        def index_path(self, p):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("boom")
            return {"files_indexed": 1, "chunks_created": 1, "jobs_queued": 1,
                    "files_skipped_unsupported": 0, "files_skipped_unchanged": 0}

    idx = _BoomIndexer()
    out = sweep_directories(idx, [a, b])
    assert out["directories"] == 1  # only one succeeded
    assert idx.n == 2  # both attempted


# --- battery ---

def test_power_source_with_ac():
    fake = type("Bat", (), {"power_plugged": True})()
    with patch("psutil.sensors_battery", return_value=fake):
        assert power_source() == "ac"
        assert is_on_battery() is False


def test_power_source_on_battery():
    fake = type("Bat", (), {"power_plugged": False})()
    with patch("psutil.sensors_battery", return_value=fake):
        assert power_source() == "battery"
        assert is_on_battery() is True


def test_power_source_no_battery_sensor():
    """Desktop without a battery — psutil returns None."""
    with patch("psutil.sensors_battery", return_value=None):
        assert power_source() == "unknown"
        assert is_on_battery() is False  # don't flip to saver on unknown


def test_power_source_handles_exception():
    with patch("psutil.sensors_battery", side_effect=RuntimeError("boom")):
        assert power_source() == "unknown"
