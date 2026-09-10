"""Tests for the camera tool."""

import base64
from unittest.mock import MagicMock

import numpy as np
import pytest

from my_conversation_app.config import config
from my_conversation_app.tools.camera import Camera
from my_conversation_app.face_recognition import SessionIdentity, RecognitionOutcome
from my_conversation_app.tools.core_tools import ToolDependencies


class _FakeRecognizer:
    """Stands in for FaceRecognitionService with a canned outcome."""

    def __init__(self, outcome: RecognitionOutcome) -> None:
        self.available = True
        self._outcome = outcome

    def recognize(self, frame: np.ndarray) -> RecognitionOutcome:
        return self._outcome


def _deps(reachy_mini: MagicMock) -> ToolDependencies:
    movement_manager = MagicMock()
    movement_manager.is_moving.return_value = False
    return ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=movement_manager,
        camera_enabled=True,
    )


@pytest.mark.asyncio
async def test_camera_tool_returns_base64_of_sdk_jpeg() -> None:
    """The tool base64-encodes the JPEG bytes returned by the SDK."""
    jpeg_bytes = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = jpeg_bytes

    result = await Camera()(_deps(reachy_mini), question="What color is this?")

    assert result["b64_im"] == base64.b64encode(jpeg_bytes).decode("utf-8")


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_no_frame() -> None:
    """With no frame available the tool returns an error."""
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = None

    result = await Camera()(_deps(reachy_mini), question="What color is this?")

    assert "error" in result


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_camera_disabled() -> None:
    """With the camera disabled the tool returns an error and never reads a frame."""
    reachy_mini = MagicMock()
    deps = _deps(reachy_mini)
    deps.camera_enabled = False

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result
    reachy_mini.media.get_frame_jpeg.assert_not_called()


@pytest.mark.asyncio
async def test_camera_tool_waits_for_head_motion_to_settle() -> None:
    """A parallel move_head must finish before the frame is captured."""
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = b"\xff\xd8jpeg\xff\xd9"
    deps = _deps(reachy_mini)
    # Busy for two polls, then settled.
    call_order: list[str] = []
    deps.movement_manager.is_moving.side_effect = [call_order.append("poll") or busy for busy in (True, True, False)]
    reachy_mini.media.get_frame_jpeg.side_effect = lambda: call_order.append("capture") or b"\xff\xd8jpeg\xff\xd9"

    result = await Camera()(deps, question="What is on my left?")

    assert "b64_im" in result
    # The frame is only read once the motion settled, never during it.
    assert call_order == ["poll", "poll", "poll", "capture"]


@pytest.mark.asyncio
async def test_camera_tool_appends_face_note_for_session_user() -> None:
    """The piggyback recognition labels the recognized session user."""
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini.media.get_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    deps = _deps(reachy_mini)
    deps.face_recognizer = _FakeRecognizer(RecognitionOutcome("凯蕾", "f_1", 0.9, True))
    deps.current_identity = SessionIdentity(name="凯蕾", face_id="f_1")

    result = await Camera()(deps, question="How do I look?")

    assert result["face"] == {"name": "凯蕾", "relation": "user"}


@pytest.mark.asyncio
async def test_camera_tool_marks_other_enrolled_and_unknown_faces() -> None:
    """A different enrolled person is "other"; an unmatched face is "unknown"."""
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini.media.get_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    deps = _deps(reachy_mini)
    deps.current_identity = SessionIdentity(name="凯蕾", face_id="f_1")

    deps.face_recognizer = _FakeRecognizer(RecognitionOutcome("李雷", "f_2", 0.9, True))
    other = await Camera()(deps, question="Who is here?")
    assert other["face"] == {"name": "李雷", "relation": "other"}

    deps.face_recognizer = _FakeRecognizer(RecognitionOutcome(None, None, 0.2, True))
    unknown = await Camera()(deps, question="Who is here?")
    assert unknown["face"] == {"name": None, "relation": "unknown"}


@pytest.mark.asyncio
async def test_camera_tool_omits_face_note_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No recognizer, no detected face, or a disabled flag all omit the note."""
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini.media.get_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

    without_recognizer = await Camera()(_deps(reachy_mini), question="What do you see?")
    assert "face" not in without_recognizer

    no_face_deps = _deps(reachy_mini)
    no_face_deps.face_recognizer = _FakeRecognizer(RecognitionOutcome(None, None, 0.0, False))
    without_face = await Camera()(no_face_deps, question="What do you see?")
    assert "face" not in without_face

    monkeypatch.setattr(config, "FACE_RECOGNITION_ENABLED", False)
    disabled_deps = _deps(reachy_mini)
    disabled_deps.face_recognizer = _FakeRecognizer(RecognitionOutcome("凯蕾", "f_1", 0.9, True))
    disabled = await Camera()(disabled_deps, question="What do you see?")
    assert "face" not in disabled
