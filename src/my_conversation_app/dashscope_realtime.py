"""DashScope Qwen-Omni-Realtime backend.

Wraps the Alibaba DashScope realtime WebSocket API
(https://www.alibabacloud.com/help/en/model-studio/realtime) in the OpenAI
SDK realtime classes so ``HuggingFaceRealtimeHandler`` drives it unchanged.
Translation happens at the connection boundary:

- outgoing ``session.update`` payloads are flattened from the OpenAI nested
  ``audio.input/output`` shape to the DashScope flat shape;
- outgoing ``conversation.item.create`` user messages carrying ``input_image``
  parts become ``input_image_buffer.append`` + ``input_audio_buffer.commit``,
  the only image path DashScope accepts (it only supports ``function_call_output``
  items); oversized camera JPEGs are re-encoded down first, since DashScope
  closes the connection on frames above 256 KiB, and because the server VAD
  only commits images alongside speech, the connection briefly switches to
  manual turn detection around the image turn;
- incoming event names are mapped to the modern OpenAI realtime names the
  conversation loop expects, confirmed transcript snapshots become
  incremental deltas, and the 24 kHz output audio is resampled to the app's
  16 kHz playback rate.
"""

import json
import uuid
import base64
import logging
from io import BytesIO
from typing import Any, Mapping

import numpy as np
import websockets
from PIL import Image
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

# DashScope closes the whole websocket (1009) when a single frame exceeds
# 256 KiB, and its image input documents the same 256 KiB base64 cap per
# image, recommending ~190 KiB raw JPEGs at 480p/720p
# (help.aliyun.com/en/model-studio/client-events). The JSON envelope around
# the base64 payload takes a few hundred bytes.
DASHSCOPE_FRAME_LIMIT_BYTES = 256 * 1024
DASHSCOPE_IMAGE_B64_SAFE_BYTES = DASHSCOPE_FRAME_LIMIT_BYTES - 512
DASHSCOPE_JPEG_TARGET_BYTES = 190 * 1024

# Ladder walked by _shrink_image_payload: smaller edges, then lower quality.
_SHRINK_MAX_EDGES = (1280, 960, 640)
_SHRINK_QUALITY_STEPS = (85, 70, 55)

# Fallback restored after an image turn when no app session.update was seen.
_DEFAULT_TURN_DETECTION = {"type": "server_vad", "interrupt_response": True}

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


def _data_url_base64_payload(image_url: Any) -> str | None:
    """Return the base64 payload of a data URL, or None for any other URL form."""
    if not isinstance(image_url, str) or not image_url.startswith("data:"):
        return None
    _, separator, payload = image_url.partition(",")
    return payload if separator else None


def _shrink_image_payload(image_b64: str) -> str:
    """Re-encode an oversized base64 JPEG down to DashScope's image budget.

    Camera frames commonly exceed the limit (a 1080p JPEG easily reaches
    500 KiB+); DashScope would drop the connection on the oversized frame.
    Returns the original payload when no encoding fits, so callers re-check.
    """
    raw_bytes = base64.b64decode(image_b64)
    try:
        with Image.open(BytesIO(raw_bytes)) as image:
            rgb = image.convert("RGB")
            for max_edge in _SHRINK_MAX_EDGES:
                rgb.thumbnail((max_edge, max_edge))
                for quality in _SHRINK_QUALITY_STEPS:
                    buffer = BytesIO()
                    rgb.save(buffer, format="JPEG", quality=quality)
                    encoded = buffer.getvalue()
                    if len(encoded) <= DASHSCOPE_JPEG_TARGET_BYTES:
                        logger.info(
                            "Re-encoded camera image for DashScope: %d -> %d bytes (edge<=%d, q=%d)",
                            len(raw_bytes),
                            len(encoded),
                            max_edge,
                            quality,
                        )
                        return base64.b64encode(encoded).decode("utf-8")
    except Exception as e:
        logger.error("Failed to re-encode camera image for DashScope: %s", e)
    return image_b64


def _extract_image_buffer_payloads(event: Mapping[str, Any]) -> list[str]:
    """Return base64 payloads of the data-URL images a user message item carries."""
    item = event.get("item")
    if not isinstance(item, Mapping) or item.get("type") != "message" or item.get("role") != "user":
        return []
    content = item.get("content")
    if not isinstance(content, list):
        return []
    payloads: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "input_image":
            payload = _data_url_base64_payload(part.get("image_url"))
            if payload is None:
                logger.warning(
                    "DashScope image input only supports base64 data URLs; dropping %s",
                    str(part.get("image_url"))[:80],
                )
            else:
                payloads.append(payload)
        elif part.get("type") is not None:
            logger.warning("DashScope image messages cannot carry %s parts; dropping it", part.get("type"))
    return payloads


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
        # Alias -> original tool name for namespaced MCP tools; DashScope
        # garbles long function names (observed duplicated prefixes), so only
        # short aliases travel to the model and call events map back.
        self._tool_aliases: dict[str, str] = {}
        # Last turn detection the app configured, restored after an image turn.
        self._app_turn_detection: dict[str, Any] | None = None

    def _alias_session_tools(self, session: dict[str, Any]) -> dict[str, Any]:
        """Replace namespaced tool names with short unique aliases."""
        tools = session.get("tools")
        if not isinstance(tools, list):
            return session
        aliased = dict(session)
        rewritten: list[Any] = []
        for index, tool in enumerate(tools):
            if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping):
                name = tool["function"].get("name")
                if isinstance(name, str) and "__" in name:
                    alias = f"ext{index}_{name.rsplit('__', 1)[-1]}"
                    self._tool_aliases[alias] = name
                    tool = dict(tool)
                    tool["function"] = {**tool["function"], "name": alias}
            rewritten.append(tool)
        aliased["tools"] = rewritten
        return aliased

    def _restore_tool_names(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Map aliased function-call names in an event back to their originals."""
        name = fields.get("name")
        if isinstance(name, str) and name in self._tool_aliases:
            fields = dict(fields)
            fields["name"] = self._tool_aliases[name]
        return fields

    async def send(self, event: RealtimeClientEvent | RealtimeClientEventParam) -> None:
        """Send a client event, flattening session updates for DashScope."""
        if isinstance(event, dict) and event.get("type") == "session.update":
            normalized = normalize_session(event.get("session") or {})
            if config.DASHSCOPE_TEMPERATURE is not None:
                # The flash realtime model wanders between calling the camera
                # tool and improvising an answer; a configured temperature pins
                # it down. DashScope-only: the OpenAI/HF session shape is untouched.
                normalized["temperature"] = config.DASHSCOPE_TEMPERATURE
            if "turn_detection" in normalized:
                self._app_turn_detection = normalized["turn_detection"]
            if isinstance(normalized.get("tools"), list):
                # A full update re-registers the tools; rebuild the alias map.
                self._tool_aliases.clear()
                session = self._alias_session_tools(normalized)
            else:
                # Partial update (voice/personality live change): the server
                # keeps the aliased tools, so the existing map must survive.
                session = normalized
            payload = {
                "type": "session.update",
                "event_id": f"event_{uuid.uuid4().hex}",
                "session": session,
            }
            await self._connection.send(json.dumps(payload))
            return
        if isinstance(event, dict) and event.get("type") == "conversation.item.create":
            image_payloads = _extract_image_buffer_payloads(event)
            if image_payloads:
                # conversation.item.create only carries function_call_output on
                # DashScope; images ride the input image buffer instead.
                await self._deliver_images_via_audio_buffer(image_payloads)
                return
        await super().send(event)

    async def _deliver_images_via_audio_buffer(self, image_payloads: list[str]) -> None:
        """Append images to the input image buffer and commit them into the conversation.

        DashScope only commits the image buffer together with a non-empty audio
        buffer, and its server VAD discards non-speech audio — a silently
        waiting user would make the commit fail. So the connection briefly
        switches to manual turn detection, appends one second of synthetic
        room noise (only sent to the server, never played on the robot), and
        restores the app's turn detection once the images are committed.
        """
        await self._connection.send(self._dashscope_session_payload({"turn_detection": None}))
        room_noise = np.random.randint(-600, 600, HuggingFaceRealtimeHandler.SAMPLE_RATE, dtype=np.int16)
        await self._connection.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"event_{uuid.uuid4().hex}",
                    "audio": base64.b64encode(room_noise.tobytes()).decode("utf-8"),
                }
            )
        )
        sent_any_image = False
        for image_payload in image_payloads:
            if len(image_payload) > DASHSCOPE_IMAGE_B64_SAFE_BYTES:
                image_payload = _shrink_image_payload(image_payload)
            # An oversized frame would get the whole connection closed (1009).
            if len(image_payload) > DASHSCOPE_IMAGE_B64_SAFE_BYTES:
                logger.error(
                    "Dropping camera image of %d bytes: still above DashScope's %d byte frame limit after re-encoding",
                    len(image_payload),
                    DASHSCOPE_FRAME_LIMIT_BYTES,
                )
                continue
            await self._connection.send(
                json.dumps(
                    {
                        "type": "input_image_buffer.append",
                        "event_id": f"event_{uuid.uuid4().hex}",
                        "image": image_payload,
                    }
                )
            )
            sent_any_image = True
        if sent_any_image:
            await self._connection.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.commit",
                        "event_id": f"event_{uuid.uuid4().hex}",
                    }
                )
            )
        await self._connection.send(
            self._dashscope_session_payload(
                {"turn_detection": self._app_turn_detection or dict(_DEFAULT_TURN_DETECTION)}
            )
        )

    @staticmethod
    def _dashscope_session_payload(session: dict[str, Any]) -> str:
        """Wrap an already DashScope-shaped partial session in an update event."""
        return json.dumps(
            {
                "type": "session.update",
                "event_id": f"event_{uuid.uuid4().hex}",
                "session": session,
            }
        )

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
        elif event_type == "response.function_call_arguments.done":
            fields = self._restore_tool_names(fields)

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
