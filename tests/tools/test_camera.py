"""Tests for the camera tool."""

import base64
from unittest.mock import MagicMock

import pytest

from my_conversation_app.tools.camera import Camera
from my_conversation_app.tools.core_tools import ToolDependencies


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
