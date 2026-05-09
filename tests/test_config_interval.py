"""parse_interval — human-friendly time string → seconds."""
import pytest
from semanticsd.config import parse_interval


def test_seconds_suffix():
    assert parse_interval("30s") == 30


def test_minutes_suffix():
    assert parse_interval("30m") == 1800


def test_hours_suffix():
    assert parse_interval("1h") == 3600
    assert parse_interval("6h") == 21600


def test_days_suffix():
    assert parse_interval("2d") == 172800


def test_decimal_value():
    assert parse_interval("1.5h") == 5400


def test_int_passthrough():
    assert parse_interval(42) == 42
    assert parse_interval(3600) == 3600


def test_float_passthrough():
    assert parse_interval(60.0) == 60


def test_digit_string_is_seconds():
    assert parse_interval("60") == 60


def test_whitespace_tolerant():
    assert parse_interval("  1h  ") == 3600


def test_empty_raises():
    with pytest.raises(ValueError):
        parse_interval("")


def test_unknown_suffix_raises():
    with pytest.raises(ValueError, match="suffix"):
        parse_interval("1y")


def test_garbage_raises():
    with pytest.raises(ValueError):
        parse_interval("not-a-time")
