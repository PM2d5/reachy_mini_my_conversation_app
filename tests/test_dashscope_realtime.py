"""Tests for the DashScope realtime adapter."""

import json
import base64
import asyncio

import numpy as np
import pytest

from my_conversation_app.dashscope_realtime import (
    DASHSCOPE_OUTPUT_SAMPLE_RATE,
    DashScopeConnection,
    DashScopeRealtimeHandler,
    resample_pcm16,
    normalize_session,
    _TranscriptDeltaState,
)


class TestNormalizeSession:
    """Session payload normalization from OpenAI to DashScope shapes."""

    def test_flattens_nested_openai_shape(self):
        """A nested OpenAI session flattens to the DashScope field layout."""
        session = {
            "type": "realtime",
            "instructions": "You are Reachy.",
            "modalities": ["text", "audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": None},
                    "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                    "turn_detection": {"type": "server_vad", "interrupt_response": True},
                },
                "output": {"format": {"type": "audio/pcm", "rate": None}, "voice": "Cherry"},
            },
            "tools": [
                {
                    "type": "function",
                    "name": "move_head",
                    "description": "Move the head.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }

        flat = normalize_session(session)

        assert flat["instructions"] == "You are Reachy."
        assert flat["input_audio_format"] == "pcm"
        assert flat["output_audio_format"] == "pcm"
        assert flat["voice"] == "Cherry"
        assert flat["turn_detection"] == {"type": "server_vad", "interrupt_response": True}
        # DashScope runs its own ASR; the transcription config must not be forwarded.
        assert "input_audio_transcription" not in flat
        assert flat["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "move_head",
                    "description": "Move the head.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        assert "audio" not in flat and "type" not in flat

    def test_keeps_flat_payload(self):
        """An already-flat session passes through unchanged."""
        session = {"modalities": ["text", "audio"], "voice": "Chelsie", "input_audio_format": "pcm"}
        assert normalize_session(session) == session


class TestResamplePcm16:
    """PCM resampling between sample rates."""

    def test_resamples_24k_to_16k(self):
        """24 kHz PCM resamples to the correct 16 kHz length and dtype."""
        pcm = np.ones(24000, dtype=np.int16)
        resampled = resample_pcm16(pcm, DASHSCOPE_OUTPUT_SAMPLE_RATE, 16000)
        assert resampled.size == 16000
        assert resampled.dtype == np.int16

    def test_same_rate_is_identity(self):
        """Matching rates return the input unchanged."""
        pcm = np.arange(100, dtype=np.int16)
        assert resample_pcm16(pcm, 16000, 16000) is pcm


class TestTranscriptDeltaState:
    """Confirmed-prefix snapshots converted to incremental deltas."""

    def test_snapshots_become_incremental_deltas(self):
        """Extending snapshots yield only the newly confirmed text."""
        state = _TranscriptDeltaState()
        assert state.to_incremental_delta({"item_id": "a", "text": "你好"}) == "你好"
        assert state.to_incremental_delta({"item_id": "a", "text": "你好，世界"}) == "，世界"

    def test_regression_resets_delta(self):
        """A rewritten transcript yields the whole new text."""
        state = _TranscriptDeltaState()
        state.to_incremental_delta({"item_id": "a", "text": "你好"})
        assert state.to_incremental_delta({"item_id": "a", "text": "完全不同"}) == "完全不同"

    def test_items_tracked_separately(self):
        """Item transcripts are tracked independently."""
        state = _TranscriptDeltaState()
        state.to_incremental_delta({"item_id": "a", "text": "你好"})
        assert state.to_incremental_delta({"item_id": "b", "text": "Hello"}) == "Hello"


def _connection() -> DashScopeConnection:
    """Return a DashScope connection without a live websocket."""
    return DashScopeConnection(websocket=None)  # type: ignore[arg-type]


class TestEventTranslation:
    """Server event translation in the connection."""

    def test_audio_delta_is_resampled_to_16k(self):
        """Audio deltas are renamed and resampled from 24 kHz to 16 kHz."""
        pcm = np.full(2400, 1000, dtype=np.int16)
        event = _connection().parse_event(
            json.dumps(
                {
                    "type": "response.audio.delta",
                    "delta": base64.b64encode(pcm.tobytes()).decode("utf-8"),
                }
            )
        )
        assert event.type == "response.output_audio.delta"
        decoded = np.frombuffer(base64.b64decode(event.delta), dtype=np.int16)
        assert decoded.size == 1600
        assert decoded.dtype == np.int16

    def test_transcript_done_is_renamed(self):
        """Audio transcript done events map to the modern OpenAI name."""
        event = _connection().parse_event(json.dumps({"type": "response.audio_transcript.done", "transcript": "你好"}))
        assert event.type == "response.output_audio_transcript.done"
        assert event.transcript == "你好"

    def test_transcription_delta_becomes_incremental(self):
        """Transcription snapshots become incremental deltas."""
        conn = _connection()
        first = conn.parse_event(
            json.dumps({"type": "conversation.item.input_audio_transcription.delta", "item_id": "i1", "text": "你好"})
        )
        second = conn.parse_event(
            json.dumps(
                {"type": "conversation.item.input_audio_transcription.delta", "item_id": "i1", "text": "你好世界"}
            )
        )
        assert first.delta == "你好"
        assert second.delta == "世界"

    def test_known_events_pass_through_unchanged(self):
        """Events already in modern OpenAI form pass through."""
        conn = _connection()
        for name in (
            "input_audio_buffer.speech_started",
            "response.created",
            "response.done",
        ):
            event = conn.parse_event(json.dumps({"type": name}))
            assert event.type == name

    def test_completed_transcript_is_forwarded(self):
        """Completed user transcripts keep their transcript field."""
        event = _connection().parse_event(
            json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": "你好"})
        )
        assert event.type == "conversation.item.input_audio_transcription.completed"
        assert event.transcript == "你好"


def test_build_client_requires_api_key(monkeypatch):
    """Building the client without an API key raises."""
    from my_conversation_app.config import config

    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", None)
    handler = DashScopeRealtimeHandler.__new__(DashScopeRealtimeHandler)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        asyncio.run(handler._build_realtime_client())


def test_build_client_targets_configured_model(monkeypatch):
    """The client websocket URL carries the configured model and endpoint."""
    from my_conversation_app.config import config

    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setattr(config, "DASHSCOPE_REALTIME_MODEL", "qwen-omni-turbo-realtime-latest")
    monkeypatch.setattr(config, "DASHSCOPE_REALTIME_WS_BASE", "wss://example.test/api-ws/v1")

    handler = DashScopeRealtimeHandler.__new__(DashScopeRealtimeHandler)
    client = asyncio.run(handler._build_realtime_client())

    manager = client.realtime.connect()
    assert manager._url == "wss://example.test/api-ws/v1/realtime?model=qwen-omni-turbo-realtime-latest"
