import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from my_conversation_app.moves import (
    NEUTRAL_ANTENNAS,
    STANDBY_ANTENNAS,
    STANDBY_HEAD_POSE,
    BUSY_SWAY_AMPLITUDE,
    MovementManager,
)
from my_conversation_app.dance_emotion_moves import EmotionQueueMove


class _FakeMove:
    """Minimal non-emotion Move stub returning a fixed head pose."""

    def __init__(self, head: np.ndarray) -> None:
        self._head = head
        self.duration = 10.0

    def evaluate(self, t: float):
        return (self._head, np.array([0.0, 0.0]), 0.0)


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = MagicMock()
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()


def test_standby_tucks_head_and_freezes_antennas() -> None:
    """Entering standby queues a level head retraction and suppresses breathing."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    manager = MovementManager(robot)

    manager._handle_command("set_standby", True, manager._now())

    assert manager._standby is True
    (tuck_move,) = manager.move_queue
    # Standby must stay distinct from the real sleep pose: the head stays level.
    assert np.allclose(tuck_move.target_head_pose[:3, :3], np.eye(3))
    assert tuck_move.target_antennas == STANDBY_ANTENNAS
    head, antennas, _body_yaw = tuck_move.evaluate(tuck_move.duration)
    assert np.allclose(head, STANDBY_HEAD_POSE)
    assert np.allclose(antennas, STANDBY_ANTENNAS)

    # Once the tuck has played (queue empty, idle again) breathing must not restart.
    manager.move_queue.clear()
    manager.state.last_activity_time = manager._now() - 10 * manager.idle_inactivity_delay
    manager._manage_breathing(manager._now())
    assert manager._breathing_active is False
    assert not manager.move_queue


def test_wake_lifts_head_back_to_neutral() -> None:
    """Leaving standby queues a goto that returns the head and antennas to neutral."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    manager = MovementManager(robot)
    manager._handle_command("set_standby", True, manager._now())

    manager._handle_command("set_standby", False, manager._now())

    assert manager._standby is False
    (wake_move,) = manager.move_queue
    assert np.allclose(wake_move.target_head_pose, np.eye(4))
    assert wake_move.target_antennas == NEUTRAL_ANTENNAS
    assert wake_move.start_body_yaw == wake_move.target_body_yaw


def test_head_tracking_follows_speaking() -> None:
    """Once enabled, tracking owns the head when idle and releases it while the assistant speaks."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    manager.start()
    try:
        # The head_tracking tool enables tracking with full weight.
        manager.set_head_tracking(True)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)

        # Speaking with a locked face captures the anchor and releases the head.
        manager.set_speaking(True)
        assert _wait_for(lambda: call(weight=0.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is not None)

        # Done speaking hands the head back to tracking.
        robot.start_head_tracking.reset_mock()
        manager.set_speaking(False)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is None)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


def test_busy_sway_wags_antennas_together_then_blends_back() -> None:
    """Busy sway starts from the live pose, wags both antennas together, then glides back."""
    robot = MagicMock()
    manager = MovementManager(robot)
    entry_pose = (np.eye(4, dtype=np.float32), (0.1, -0.1), 0.0)
    manager._last_commanded_pose = entry_pose

    manager._handle_command("set_busy_sway", True, manager._now())

    # At entry the sway equals its center: no jump when the tool call starts.
    entry_left, entry_right = manager._calculate_blended_antennas((0.0, 0.0))
    assert entry_left == pytest.approx(0.1, abs=0.05)
    assert entry_right == pytest.approx(-0.1, abs=0.05)

    # A quarter period later both antennas are offset by the same amplitude.
    manager._busy_sway_start -= 1.0 / (4 * 0.6)
    left, right = manager._calculate_blended_antennas((0.0, 0.0))
    assert left == pytest.approx(0.1 + BUSY_SWAY_AMPLITUDE, abs=1e-5)
    assert right == pytest.approx(-0.1 + BUSY_SWAY_AMPLITUDE, abs=1e-5)

    # After the tool call the antennas blend back toward the target instead of snapping.
    manager._handle_command("set_busy_sway", False, manager._now())
    assert manager._busy_sway is False
    assert manager._listening_antennas == (0.1, -0.1)
    assert manager._antenna_unfreeze_blend == 0.0


def test_speaking_anchor_composes_emotions_and_holds_dances_from_neutral() -> None:
    """While speaking: hold the anchor, compose emotions onto it, play dances from neutral."""
    robot = MagicMock()
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor

    # No move: the head holds the captured look-at anchor.
    manager.state.current_move = None
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, anchor)

    # Emotion: composed onto the anchor exactly like the daemon wobble.
    emotion_head = create_head_pose(0, 0, 0, 0, 0, 15, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(emotion_head)
    manager.state.current_move = EmotionQueueMove("happy", recorded)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, compose_world_offset(anchor, emotion_head))

    # Any other move (e.g. a dance) plays from its own neutral base, ignoring the anchor.
    dance_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    manager.state.current_move = _FakeMove(dance_head)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, dance_head)


def test_set_moving_state_marks_motion_until_the_deadline() -> None:
    """set_moving_state(duration) keeps is_moving() True only until the deadline."""
    robot = MagicMock()
    manager = MovementManager(robot)

    assert manager.is_moving() is False

    manager._handle_command("set_moving_state", 2.0, manager._now())
    manager._publish_shared_state()
    assert manager.is_moving() is True

    # A command whose window already elapsed does not extend the deadline.
    expired_manager = MovementManager(robot)
    expired_manager._handle_command("set_moving_state", 2.0, expired_manager._now() - 5.0)
    expired_manager._publish_shared_state()
    assert expired_manager.is_moving() is False


def test_clear_move_queue_releases_the_motion_deadline() -> None:
    """Cancelling the queue also cancels the settling wait of a camera capture."""
    robot = MagicMock()
    manager = MovementManager(robot)

    manager._handle_command("set_moving_state", 60.0, manager._now())
    manager._handle_command("clear_queue", None, manager._now())
    manager._publish_shared_state()

    assert manager.is_moving() is False
