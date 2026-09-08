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


def test_dashscope_voice_catalog_follows_model_family(monkeypatch) -> None:
    """Voice catalogs resolve per DashScope realtime model family."""
    monkeypatch.setattr(config.config, "REALTIME_BACKEND", "dashscope")

    monkeypatch.setattr(config.config, "DASHSCOPE_REALTIME_MODEL", "qwen3.5-omni-flash-realtime")
    assert config.get_available_voices() == config.DASHSCOPE_AVAILABLE_VOICES

    monkeypatch.setattr(config.config, "DASHSCOPE_REALTIME_MODEL", "qwen-audio-3.0-realtime-plus")
    assert config.get_available_voices() == config.DASHSCOPE_AUDIO_REALTIME_VOICES


def test_dashscope_default_voice_falls_back_across_families(monkeypatch) -> None:
    """A voice configured for another model family falls back to the family default."""
    monkeypatch.setattr(config.config, "REALTIME_BACKEND", "dashscope")
    monkeypatch.setattr(config.config, "DASHSCOPE_REALTIME_MODEL", "qwen-audio-3.0-realtime-plus")
    monkeypatch.setattr(config.config, "DASHSCOPE_REALTIME_VOICE", "Mione")

    assert config.get_default_voice() == "longanqian"


def test_dashscope_default_voice_matches_configured_case_insensitively(monkeypatch) -> None:
    """A supported configured voice resolves regardless of case."""
    monkeypatch.setattr(config.config, "REALTIME_BACKEND", "dashscope")
    monkeypatch.setattr(config.config, "DASHSCOPE_REALTIME_MODEL", "qwen-audio-3.0-realtime-plus")
    monkeypatch.setattr(config.config, "DASHSCOPE_REALTIME_VOICE", "Sherry")

    assert config.get_default_voice() == "sherry"
