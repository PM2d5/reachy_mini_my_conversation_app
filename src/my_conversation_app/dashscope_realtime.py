"""DashScope Qwen-Omni-Realtime backend.

Wraps the Alibaba DashScope realtime WebSocket API
(https://www.alibabacloud.com/help/en/model-studio/realtime) in the OpenAI
SDK realtime classes so ``HuggingFaceRealtimeHandler`` drives it unchanged.
Translation happens at the connection boundary:

- outgoing ``session.update`` payloads are flattened from the OpenAI nested
  ``audio.input/output`` shape to the DashScope flat shape;
- incoming event names are mapped to the modern OpenAI realtime names the
  conversation loop expects, confirmed transcript snapshots become
  incremental deltas, and the 24 kHz output audio is resampled to the app's
  16 kHz playback rate.
"""

import json
import uuid
import base64
import logging
from typing import Any, Mapping

import numpy as np
import websockets
from openai import AsyncOpenAI
from numpy.typing import NDArray
from openai.types.realtime import RealtimeClientEvent, RealtimeServerEvent, RealtimeClientEventParam
from websockets.asyncio.client import ClientConnection
from openai.resources.realtime.realtime import (
    AsyncRealtime,
    AsyncRealtimeConnection,
    AsyncRealtimeConnectionManager,
)

from my_conversation_app.config import config
from my_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


logger = logging.getLogger(__name__)

# DashScope delivers output audio as 24 kHz PCM regardless of the requested
# output format; the app plays back at 16 kHz.
DASHSCOPE_OUTPUT_SAMPLE_RATE = 24000

# DashScope event name -> modern OpenAI realtime event name.
_EVENT_NAME_MAP = {
    "response.audio.delta": "response.output_audio.delta",
    "response.audio.done": "response.output_audio.done",
    "response.audio_transcript.delta": "response.output_audio_transcript.delta",
    "response.audio_transcript.done": "response.output_audio_transcript.done",
    "response.text.delta": "response.output_text.delta",
    "response.text.done": "response.output_text.done",
}


def resample_pcm16(
    pcm: NDArray[np.int16],
    source_rate: int,
    target_rate: int,
) -> NDArray[np.int16]:
    """Linearly resample mono 16-bit PCM between sample rates."""
    if source_rate == target_rate or pcm.size == 0:
        return pcm
    target_length = max(1, round(pcm.size * target_rate / source_rate))
    source_positions = np.linspace(0.0, pcm.size - 1, target_length)
    resampled = np.interp(source_positions, np.arange(pcm.size), pcm.astype(np.float64))
    return resampled.astype(np.int16)


def _pcm_format_name(audio_format: Any) -> Any:
    """Normalize an audio format descriptor to DashScope's bare ``pcm`` name."""
    if isinstance(audio_format, Mapping):
        audio_format = audio_format.get("type")
    if isinstance(audio_format, str) and audio_format.startswith("audio/"):
        return audio_format.removeprefix("audio/")
    return audio_format


def _nest_function_tool(tool: Any) -> Any:
    """Convert an OpenAI flat function tool to the nested DashScope shape."""
    if not isinstance(tool, Mapping) or tool.get("function") is not None:
        return tool
    nested = {k: v for k, v in tool.items() if k not in ("name", "description", "parameters", "strict")}
    nested["function"] = {
        k: tool[k] for k in ("name", "description", "parameters", "strict") if tool.get(k) is not None
    }
    return nested


def normalize_session(session: Any) -> dict[str, Any]:
    """Flatten an OpenAI-style nested session payload to the DashScope shape."""
    source: Mapping[str, Any] = session if isinstance(session, Mapping) else session.model_dump(exclude_none=True)
    # DashScope runs its own multilingual ASR, so the transcription config is dropped.
    flat: dict[str, Any] = {k: v for k, v in source.items() if k not in ("audio", "type", "input_audio_transcription")}

    audio = source.get("audio") or {}
    audio_in = audio.get("input") or {}
    audio_out = audio.get("output") or {}
    if audio_in.get("format") is not None:
        flat["input_audio_format"] = _pcm_format_name(audio_in["format"])
    if audio_in.get("turn_detection") is not None:
        flat["turn_detection"] = dict(audio_in["turn_detection"])
    if audio_out.get("format") is not None:
        flat["output_audio_format"] = _pcm_format_name(audio_out["format"])
    if audio_out.get("voice") is not None:
        flat["voice"] = audio_out["voice"]

    if "tools" in flat:
        flat["tools"] = [_nest_function_tool(tool) for tool in flat["tools"]]
    return flat


class _TranscriptDeltaState:
    """Track confirmed transcript prefixes so DashScope snapshots become deltas."""

    def __init__(self) -> None:
        """Start with no confirmed transcript for any item."""
        self._confirmed_by_item: dict[str, str] = {}

    def to_incremental_delta(self, event: Mapping[str, Any]) -> str:
        """Return the newly confirmed text for this item."""
        item_id = str(event.get("item_id") or "")
        confirmed = str(event.get("text") or "")
        previous = self._confirmed_by_item.get(item_id, "")
        if confirmed.startswith(previous):
            delta = confirmed[len(previous) :]
        else:
            delta = confirmed
        self._confirmed_by_item[item_id] = confirmed
        return delta


class DashScopeConnection(AsyncRealtimeConnection):
    """OpenAI realtime connection bound to a DashScope websocket."""

    def __init__(self, websocket: ClientConnection) -> None:
        """Bind the DashScope translations to an open websocket."""
        super().__init__(websocket)
        self._transcript_state = _TranscriptDeltaState()

    async def send(self, event: RealtimeClientEvent | RealtimeClientEventParam) -> None:
        """Send a client event, flattening session updates for DashScope."""
        if isinstance(event, dict) and event.get("type") == "session.update":
            payload = {
                "type": "session.update",
                "event_id": f"event_{uuid.uuid4().hex}",
                "session": normalize_session(event.get("session") or {}),
            }
            await self._connection.send(json.dumps(payload))
            return
        await super().send(event)

    def _translate_payload(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Return the payload rewritten with modern event names and app-rate audio."""
        event_type = str(raw.get("type") or "")
        fields = dict(raw)

        if event_type == "conversation.item.input_audio_transcription.delta":
            fields["delta"] = self._transcript_state.to_incremental_delta(raw)
        elif event_type == "response.audio.delta":
            pcm_bytes = base64.b64decode(raw.get("delta") or "")
            pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
            resampled = resample_pcm16(pcm, DASHSCOPE_OUTPUT_SAMPLE_RATE, HuggingFaceRealtimeHandler.SAMPLE_RATE)
            fields["delta"] = base64.b64encode(resampled.tobytes()).decode("utf-8")

        fields["type"] = _EVENT_NAME_MAP.get(event_type, event_type)
        return fields

    def parse_event(self, data: str | bytes) -> RealtimeServerEvent:
        """Parse a DashScope message into an OpenAI realtime server event."""
        try:
            raw = json.loads(data)
        except ValueError:
            return super().parse_event(data)
        if isinstance(raw, dict) and raw.get("type"):
            return super().parse_event(json.dumps(self._translate_payload(raw)))
        return super().parse_event(data)


class _DashScopeConnectionManager(AsyncRealtimeConnectionManager):
    def __init__(self, url: str, headers: dict[str, str]) -> None:
        """Store the DashScope websocket target."""
        self._url = url
        self._headers = headers
        self._connection: DashScopeConnection | None = None

    async def enter(self) -> DashScopeConnection:
        """Open the DashScope websocket."""
        websocket = await websockets.connect(self._url, additional_headers=self._headers, max_size=16 * 1024 * 1024)
        self._connection = DashScopeConnection(websocket)
        return self._connection

    async def __aenter__(self) -> DashScopeConnection:
        """Open the DashScope websocket on context entry."""
        return await self.enter()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the websocket on context exit."""
        if self._connection is not None:
            await self._connection.close()


class _DashScopeRealtime(AsyncRealtime):
    def __init__(self, client: AsyncOpenAI, url: str, headers: dict[str, str]) -> None:
        """Attach the DashScope websocket target to an OpenAI client shell."""
        super().__init__(client)
        self._url = url
        self._headers = headers

    def connect(self, **_kwargs: Any) -> _DashScopeConnectionManager:
        """Return a connection manager for the DashScope endpoint."""
        return _DashScopeConnectionManager(self._url, self._headers)


class DashScopeRealtimeClient(AsyncOpenAI):
    """OpenAI client shell whose realtime transport talks to DashScope."""

    def __init__(self, api_key: str, url: str) -> None:
        """Initialize the shell and its DashScope realtime transport."""
        super().__init__(api_key=api_key)
        self._dashscope_realtime = _DashScopeRealtime(self, url, {"Authorization": f"Bearer {api_key}"})

    @property
    def realtime(self) -> _DashScopeRealtime:
        """Return the DashScope realtime resource instead of the OpenAI one."""
        return self._dashscope_realtime


class DashScopeRealtimeHandler(HuggingFaceRealtimeHandler):
    """Realtime handler driving the DashScope Qwen-Omni-Realtime backend."""

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the DashScope realtime client from runtime config."""
        api_key = (config.DASHSCOPE_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY must be set to use the DashScope realtime backend.")
        model = config.DASHSCOPE_REALTIME_MODEL
        ws_base = config.DASHSCOPE_REALTIME_WS_BASE
        logger.info("Using DashScope realtime backend: model=%s endpoint=%s", model, ws_base)
        return DashScopeRealtimeClient(api_key=api_key, url=f"{ws_base}/realtime?model={model}")
