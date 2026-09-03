from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from reachy_mini.reachy_mini import SLEEP_HEAD_POSE, SLEEP_ANTENNAS_JOINT_POSITIONS
from my_conversation_app import app_lifecycle
from my_conversation_app.tools.core_tools import ToolDependencies


def test_request_stop_current_app_posts_to_daemon(monkeypatch) -> None:
    """The app stop request should call the connected Reachy daemon endpoint."""

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://192.168.1.42:8000/api/apps/stop-current-app"
        assert request.get_method() == "POST"
        assert timeout == 2.0
        return FakeResponse()

    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))

    assert app_lifecycle.request_stop_current_app(robot, MagicMock())


def test_wake_up_if_sleeping_enables_motors_before_wake_up() -> None:
    """Startup should enable sleeping motors before playing the wake-up movement."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()

    assert app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    assert robot.method_calls == [
        call.get_current_head_pose(),
        call.enable_motors(),
        call.wake_up(),
    ]


def test_wake_up_if_sleeping_skips_non_sleep_head_pose() -> None:
    """Startup should leave an already-awake robot alone."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()


def test_goto_sleep_moves_straight_to_sleep_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown must reach the sleep pose directly instead of detouring through neutral."""
    monkeypatch.setattr(app_lifecycle.time, "sleep", MagicMock())
    robot = MagicMock()

    app_lifecycle.goto_sleep_from_current_pose(robot, MagicMock())

    robot.media.play_sound.assert_called_once_with("go_sleep.wav")
    robot.goto_sleep.assert_not_called()
    robot.goto_target.assert_called_once_with(
        head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS_JOINT_POSITIONS, duration=2.0
    )
    app_lifecycle.time.sleep.assert_called_once_with(app_lifecycle._SLEEP_SOUND_TAIL_S)


def test_goto_sleep_continues_without_the_sound(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed sleep sound must not block the move to the sleep pose."""
    monkeypatch.setattr(app_lifecycle.time, "sleep", MagicMock())
    robot = MagicMock()
    robot.media.play_sound.side_effect = RuntimeError("no sound backend")

    app_lifecycle.goto_sleep_from_current_pose(robot, MagicMock())

    robot.goto_target.assert_called_once()


def test_run_go_to_sleep_tool_uses_runtime_callback() -> None:
    """Synchronous lifecycle paths should enter through the go_to_sleep tool."""
    expected = {"status": "sleeping"}
    go_to_sleep = MagicMock(return_value=expected)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )

    result = app_lifecycle.run_go_to_sleep_tool(deps, MagicMock())

    assert result == expected
    go_to_sleep.assert_called_once_with()
