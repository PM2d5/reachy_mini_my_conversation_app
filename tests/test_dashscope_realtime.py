"""Tests for the DashScope realtime adapter."""

import json
import base64
import asyncio
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from my_conversation_app.dashscope_realtime import (
    DASHSCOPE_FRAME_LIMIT_BYTES,
    DASHSCOPE_OUTPUT_SAMPLE_RATE,
    DASHSCOPE_IMAGE_B64_SAFE_BYTES,
    DashScopeConnection,
    DashScopeRealtimeHandler,
    resample_pcm16,
    normalize_session,
    _shrink_image_payload,
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


class TestToolNameAliasing:
    """Namespaced MCP tool names are aliased to the model and restored on calls."""

    def _sent_payload(self, conn, session):
        """Capture the websocket frame emitted for a session.update."""
        import asyncio

        sent: list[str] = []

        class FakeWebsocket:
            async def send(self, message: str) -> None:
                sent.append(message)

        conn._connection = FakeWebsocket()  # type: ignore[assignment]
        asyncio.run(conn.send({"type": "session.update", "session": session}))
        return json.loads(sent[0])

    def test_temperature_rides_along_when_configured(self, monkeypatch):
        """A configured DashScope temperature is injected into every session update."""
        from my_conversation_app.config import config

        monkeypatch.setattr(config, "DASHSCOPE_TEMPERATURE", 0.3)
        conn = _connection()
        payload = self._sent_payload(conn, {"voice": "Mione"})
        assert payload["session"]["temperature"] == 0.3

    def test_temperature_absent_without_config(self, monkeypatch):
        """No configured temperature leaves the session payload untouched."""
        from my_conversation_app.config import config

        monkeypatch.setattr(config, "DASHSCOPE_TEMPERATURE", None)
        conn = _connection()
        payload = self._sent_payload(conn, {"voice": "Mione"})
        assert "temperature" not in payload["session"]

    def test_long_tool_names_are_aliased_and_restored(self):
        """Namespaced tools ship as short aliases and call events map back."""
        conn = _connection()
        original = "pollen_robotics_reachy_mini_search_tool__search_web"
        payload = self._sent_payload(
            conn,
            {
                "tools": [
                    {"type": "function", "name": original, "description": "Search the web.", "parameters": {}},
                    {"type": "function", "name": "dance", "description": "Dance.", "parameters": {}},
                ]
            },
        )

        shipped_names = [t["function"]["name"] for t in payload["session"]["tools"]]
        assert "dance" in shipped_names
        alias = "ext0_search_web"
        assert alias in shipped_names
        assert original not in shipped_names
        assert conn._tool_aliases == {alias: original}

        event = conn.parse_event(
            json.dumps({"type": "response.function_call_arguments.done", "name": alias, "arguments": "{}"})
        )
        assert event.name == original

    def test_alias_map_resets_between_sessions(self):
        """A new session.update clears stale aliases before re-registering tools."""
        conn = _connection()
        self._sent_payload(
            conn, {"tools": [{"type": "function", "name": "a__get_time", "description": "", "parameters": {}}]}
        )
        self._sent_payload(
            conn, {"tools": [{"type": "function", "name": "b__get_time", "description": "", "parameters": {}}]}
        )
        assert set(conn._tool_aliases) == {"ext0_get_time"}
        assert conn._tool_aliases["ext0_get_time"] == "b__get_time"


def test_partial_session_update_keeps_tool_aliases():
    """A voice-only session.update must not wipe the alias map."""
    conn = _connection()
    original = "pollen_robotics_reachy_mini_weather_tool__get_weather"

    class FakeWebsocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, message: str) -> None:
            self.sent.append(message)

    fake_ws = FakeWebsocket()
    conn._connection = fake_ws  # type: ignore[assignment]

    asyncio.run(
        conn.send(
            {
                "type": "session.update",
                "session": {"tools": [{"type": "function", "name": original, "description": "", "parameters": {}}]},
            }
        )
    )
    asyncio.run(
        conn.send(
            {
                "type": "session.update",
                "session": {"audio": {"output": {"voice": "Mione"}}},
            }
        )
    )

    event = conn.parse_event(
        json.dumps({"type": "response.function_call_arguments.done", "name": "ext0_get_weather", "arguments": "{}"})
    )
    assert event.name == original


class TestImageBufferTranslation:
    """Camera input_image messages become DashScope image buffer events."""

    def _attach_fake_websocket(self, conn: DashScopeConnection) -> list[str]:
        """Capture every websocket frame the connection emits from now on."""
        sent: list[str] = []

        class FakeWebsocket:
            async def send(self, message: str) -> None:
                sent.append(message)

        conn._connection = FakeWebsocket()  # type: ignore[assignment]
        return sent

    def _sent_frames(self, conn: DashScopeConnection, event: dict) -> list[dict]:
        """Capture the websocket frames emitted for a client event."""
        sent = self._attach_fake_websocket(conn)
        asyncio.run(conn.send(event))
        return [json.loads(frame) for frame in sent]

    def _camera_item_create(self, image_b64: str) -> dict:
        """Return the item.create event the camera tool result triggers."""
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"}],
            },
        }

    def test_camera_image_becomes_image_buffer_events(self):
        """A camera image ships via the image buffer inside a manual turn-detection window."""
        conn = _connection()
        frames = self._sent_frames(conn, self._camera_item_create("QUJD"))
        assert [frame["type"] for frame in frames] == [
            "session.update",
            "input_audio_buffer.append",
            "input_image_buffer.append",
            "input_audio_buffer.commit",
            "session.update",
        ]
        # Manual mode for the image turn, then the server VAD restored.
        assert frames[0]["session"]["turn_detection"] is None
        assert frames[-1]["session"]["turn_detection"] == {"type": "server_vad", "interrupt_response": True}
        # The image travels as raw base64, without the data URL prefix.
        assert frames[2]["image"] == "QUJD"
        # The placeholder audio keeps the commit viable while the user is silent.
        assert base64.b64decode(frames[1]["audio"])

    def test_camera_image_restores_configured_turn_detection(self):
        """The turn detection from the app's session.update is restored after the image turn."""
        conn = _connection()
        configured = {"type": "server_vad", "interrupt_response": True, "threshold": 0.3}
        sent = self._attach_fake_websocket(conn)
        asyncio.run(
            conn.send(
                {
                    "type": "session.update",
                    "session": {"audio": {"input": {"turn_detection": configured}}},
                }
            )
        )
        asyncio.run(conn.send(self._camera_item_create("QUJD")))
        frames = [json.loads(frame) for frame in sent]
        # First frame is the app session.update; the image turn closes by restoring it.
        assert frames[-1]["session"]["turn_detection"] == configured

    def test_function_call_output_item_passes_through(self):
        """function_call_output items keep the OpenAI item.create transport."""
        conn = _connection()
        frames = self._sent_frames(
            conn,
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_1", "output": '{"image_attached": true}'},
            },
        )
        assert [frame["type"] for frame in frames] == ["conversation.item.create"]
        assert frames[0]["item"]["type"] == "function_call_output"

    def test_text_only_message_passes_through(self):
        """Messages without images (e.g. the relay prompt) keep today's transport."""
        conn = _connection()
        frames = self._sent_frames(
            conn,
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Relay the assistant result."}],
                },
            },
        )
        assert [frame["type"] for frame in frames] == ["conversation.item.create"]

    def test_oversize_image_is_reencoded_under_the_frame_limit(self):
        """A camera JPEG above DashScope's 256 KiB frame cap ships re-encoded smaller.

        Regression test: the raw 1080p frame used to be sent as-is and DashScope
        closed the whole websocket with 1009 (message too big).
        """
        rng = np.random.default_rng(7)
        noise = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
        raw_buffer = BytesIO()
        Image.fromarray(noise, "RGB").save(raw_buffer, format="JPEG", quality=95)
        oversize_b64 = base64.b64encode(raw_buffer.getvalue()).decode("utf-8")
        assert len(oversize_b64) > DASHSCOPE_IMAGE_B64_SAFE_BYTES

        conn = _connection()
        frames = self._sent_frames(conn, self._camera_item_create(oversize_b64))
        image_frames = [frame for frame in frames if frame["type"] == "input_image_buffer.append"]
        assert len(image_frames) == 1
        assert len(json.dumps(image_frames[0])) < DASHSCOPE_FRAME_LIMIT_BYTES
        decoded = Image.open(BytesIO(base64.b64decode(image_frames[0]["image"])))
        assert max(decoded.size) <= 1280
        # The commit still closes the image turn.
        assert frames[-2]["type"] == "input_audio_buffer.commit"

    def test_shrink_payload_returns_original_on_undecodable_image(self, caplog):
        """A corrupt image is returned unchanged and the failure is logged."""
        broken_b64 = base64.b64encode(b"not a jpeg").decode("utf-8")
        with caplog.at_level("ERROR", logger="my_conversation_app.dashscope_realtime"):
            assert _shrink_image_payload(broken_b64) == broken_b64
        assert "Failed to re-encode" in caplog.text
