"""Tests for configuration helpers."""

import pytest

from my_conversation_app import config


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unset/blank falls back to the default
        ("soon", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unparseable falls back to the default
        ("0", None),  # non-positive disables the watchdog
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch, raw_value, expected) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("0.3", 0.3),
        ("1", 1.0),
        ("1.99", 1.99),
        ("", None),  # unset keeps the backend default
        ("warm", None),  # unparseable keeps the backend default
        ("2", None),  # DashScope's range is [0, 2), 2 excluded
        ("2.5", None),  # out of range keeps the backend default
        ("-0.1", None),
    ],
)
def test_resolve_dashscope_temperature(monkeypatch, raw_value, expected) -> None:
    """The DashScope temperature parses within 0-2, else falls back to None."""
    monkeypatch.setenv(config.DASHSCOPE_TEMPERATURE_ENV, raw_value)

    assert config.resolve_dashscope_temperature() == expected
