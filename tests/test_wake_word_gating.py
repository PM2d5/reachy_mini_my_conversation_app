"""Tests for the wake word standby/active gating in LocalStream."""

import time
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import my_conversation_app.console as console_mod
from my_conversation_app import config
from my_conversation_app.console import LocalStream


def _make_stream(monkeypatch: pytest.MonkeyPatch) -> tuple[LocalStream, MagicMock]:
    """Build a LocalStream with a mocked handler and wake word mode enabled."""
    monkeypatch.setattr(config.config, "WAKE_WORD_ENABLED", True)
    monkeypatch.setattr(config.config, "WAKE_WORD_ACTIVE_TIMEOUT_S", 300.0)
    # record_loop must not load real openWakeWord models in tests.
    monkeypatch.setattr(console_mod, "WakeWordDetector", lambda **kwargs: SimpleNamespace(available=False))
    handler = MagicMock()
    handler.receive = AsyncMock()
    handler.pause_session = AsyncMock()
    handler.resume_session = AsyncMock()
    handler.last_activity_time = time.monotonic()
    handler.deps.movement_manager.set_listening = MagicMock()
    frame = np.zeros(1280, dtype=np.int16)
    robot = SimpleNamespace(
        media=SimpleNamespace(
            audio=None,
            get_input_audio_samplerate=lambda: 16000,
            get_audio_sample=lambda: frame,
        ),
    )
    return LocalStream(handler, robot), handler


@pytest.mark.asyncio
async def test_goodbye_transcript_pauses_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A final user transcript containing a goodbye keyword enters standby."""
    stream, handler = _make_stream(monkeypatch)

    stream._dispatch_transcript("user", "好的，再见啦", True)
    await asyncio.sleep(0.05)

    handler.pause_session.assert_awaited_once()
    assert stream._standby is True


@pytest.mark.asyncio
async def test_startup_loop_stays_paused_in_standby(monkeypatch: pytest.MonkeyPatch) -> None:
    """The startup loop must not reopen a session on its own while in standby."""
    stream, handler = _make_stream(monkeypatch)
    stream._standby = True
    handler.start_up = AsyncMock()

    startup_task = asyncio.create_task(stream._run_handler_startup_loop())
    await asyncio.sleep(0.2)
    stream._stop_event.set()
    await startup_task

    handler.start_up.assert_not_awaited()


@pytest.mark.asyncio
async def test_wake_leaves_standby_before_the_startup_loop_reconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Waking clears standby so the startup loop reconnects and re-greets the user."""
    stream, handler = _make_stream(monkeypatch)
    stream._standby = True

    await stream._wake_from_standby()

    handler.resume_session.assert_awaited_once()
    assert stream._standby is False


@pytest.mark.asyncio
async def test_plain_transcript_keeps_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transcripts without a goodbye keyword, or assistant ones, never pause."""
    stream, handler = _make_stream(monkeypatch)

    stream._dispatch_transcript("user", "今天天气怎么样", True)
    stream._dispatch_transcript("assistant", "再见，下次见", True)
    await asyncio.sleep(0.05)

    handler.pause_session.assert_not_awaited()
    assert stream._standby is False


@pytest.mark.asyncio
async def test_wake_word_hit_resumes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A detector hit leaves standby and reopens the session."""
    stream, handler = _make_stream(monkeypatch)
    stream._standby = True
    detector = SimpleNamespace(predict=lambda rate, frame: "hey_mycroft")

    stream._detect_wake_word(detector, 16000, np.zeros(1280, dtype=np.int16))
    await asyncio.sleep(0.05)

    handler.resume_session.assert_awaited_once()
    assert stream._standby is False


@pytest.mark.asyncio
async def test_standby_frames_never_reach_the_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """In standby, mic frames go nowhere even when the detector is unavailable."""
    stream, handler = _make_stream(monkeypatch)
    stream._standby = True

    record_task = asyncio.create_task(stream.record_loop())
    await asyncio.sleep(0.05)
    stream._stop_event.set()
    await record_task

    handler.receive.assert_not_awaited()


@pytest.mark.asyncio
async def test_active_frames_are_forwarded_to_the_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """While active and not idle, mic frames flow to the handler as before."""
    stream, handler = _make_stream(monkeypatch)

    record_task = asyncio.create_task(stream.record_loop())
    await asyncio.sleep(0.05)
    stream._stop_event.set()
    await record_task

    handler.receive.assert_awaited()


@pytest.mark.asyncio
async def test_idle_expiry_pauses_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence beyond the active timeout returns the stream to standby."""
    stream, handler = _make_stream(monkeypatch)
    handler.last_activity_time = time.monotonic() - 301.0

    record_task = asyncio.create_task(stream.record_loop())
    await asyncio.sleep(0.05)
    stream._stop_event.set()
    await record_task

    handler.pause_session.assert_awaited_once()
    assert stream._standby is True
