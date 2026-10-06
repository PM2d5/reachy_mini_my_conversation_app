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
    STANDBY_MOVE_GRACE_S,
    BreathingMove,
    MovementManager,
)
from my_conversation_app.dance_emotion_moves import LEAD_IN_DURATION_S, GotoQueueMove, EmotionQueueMove


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
    # The fold must stay clear of vertical-down: at that gravity dead point the
    # pure-P motor hold limit-cycles across the gearbox backlash and the folded
    # antennas twitch until nudged.
    assert all(abs(a) < np.pi - np.deg2rad(10.0) for a in tuck_move.target_antennas)
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


def test_moves_queued_during_standby_are_dropped() -> None:
    """A late emotion move (goodbye play_emotion) must not pop the head out of the standby tuck."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    manager = MovementManager(robot)

    manager._handle_command("set_standby", True, manager._now())
    (tuck_move,) = manager.move_queue

    # A real wrapper move, not a bare stub: queue_move only accepts Move instances.
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(np.eye(4))
    manager._handle_command("queue_move", EmotionQueueMove("goodbye", recorded), manager._now())

    assert list(manager.move_queue) == [tuck_move]


def test_emotion_move_eases_in_from_the_captured_start_pose() -> None:
    """The recording's posed first frame must not step the head: it blends in over the lead-in."""
    recorded_head = create_head_pose(0, 0, 0, 0, 0, 14, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(recorded_head)
    move = EmotionQueueMove("goodbye", recorded)
    start_head = create_head_pose(0, 0, 0, 0, 0, -20, degrees=True)
    move.start_pose = (start_head, (0.1, -0.1), 0.05)

    head, antennas, body_yaw = move.evaluate(0.0)
    assert np.allclose(head, start_head)
    assert np.allclose(antennas, [0.1, -0.1])
    assert body_yaw == pytest.approx(0.05)

    # Smoothstep midpoint: halfway between the start pose and the recording.
    head, antennas, _body_yaw = move.evaluate(LEAD_IN_DURATION_S / 2)
    assert np.degrees(np.arctan2(head[1, 0], head[0, 0])) == pytest.approx(-3.0, abs=0.1)
    assert antennas[0] == pytest.approx(0.05)

    head, antennas, body_yaw = move.evaluate(LEAD_IN_DURATION_S + 0.01)
    assert np.allclose(head, recorded_head)
    assert np.allclose(antennas, [0.0, 0.0])
    assert body_yaw == pytest.approx(0.0)


def test_started_emotion_seeds_lead_in_from_the_commanded_pose() -> None:
    """Popping a queued emotion captures the commanded pose as the lead-in source."""
    robot = MagicMock()
    manager = MovementManager(robot)
    commanded_head = create_head_pose(0, 0, 0, 0, 0, -20, degrees=True)
    manager._last_commanded_pose = (commanded_head, (0.05, -0.05), 0.1)
    recorded = MagicMock()
    emotion = EmotionQueueMove("goodbye", recorded)
    manager.move_queue.append(emotion)

    manager._manage_move_queue(manager._now())

    assert manager.state.current_move is emotion
    assert emotion.start_pose is not None
    start_head, start_antennas, start_body_yaw = emotion.start_pose
    assert np.allclose(start_head, commanded_head)
    assert start_antennas == (0.05, -0.05)
    assert start_body_yaw == 0.1

    # Under a look-at anchor the blend starts from identity: the anchor
    # composition already places the head at the anchor.
    anchored = EmotionQueueMove("goodbye", recorded)
    manager._track_anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager.state.current_move = None
    manager.move_queue.append(anchored)
    manager._manage_move_queue(manager._now())
    assert anchored.start_pose is not None
    assert np.allclose(anchored.start_pose[0], np.eye(4))


def test_standby_gives_a_playing_move_a_capped_grace_before_tucking() -> None:
    """The goodbye emotion playing when the keyword lands keeps the floor; the tuck cuts in late."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    manager = MovementManager(robot)
    goodbye = _FakeMove(np.eye(4))
    goodbye.duration = 5.6
    now = manager._now()
    manager.state.current_move = goodbye
    manager.state.move_start_time = now - 0.2

    manager._handle_command("set_standby", True, now)

    # The move keeps playing; the tuck waits out a grace capped at STANDBY_MOVE_GRACE_S.
    assert manager.state.current_move is goodbye
    assert not manager.move_queue
    deadline = manager._standby_tuck_at
    assert deadline is not None
    assert deadline == pytest.approx(now + STANDBY_MOVE_GRACE_S)

    # Past the deadline the tuck replaces the lingering move from the live pose.
    manager._manage_move_queue(deadline + 0.05)
    assert manager._standby_tuck_at is None
    assert isinstance(manager.state.current_move, GotoQueueMove)
    assert not manager.move_queue


def test_standby_tucks_once_a_short_move_finishes_naturally() -> None:
    """A move outliving the keyword by less than the cap is left to finish on its own."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    manager = MovementManager(robot)
    farewell = _FakeMove(np.eye(4))
    farewell.duration = 0.5
    now = manager._now()
    manager.state.current_move = farewell
    manager.state.move_start_time = now - 0.2

    manager._handle_command("set_standby", True, now)

    deadline = manager._standby_tuck_at
    assert deadline is not None
    assert deadline == pytest.approx(now + 0.3)  # the remaining 0.3 s, under the cap

    # The move ends before its deadline; the tuck starts on that same tick.
    manager._manage_move_queue(now + 0.4)
    assert manager._standby_tuck_at is None
    assert isinstance(manager.state.current_move, GotoQueueMove)


def test_standby_tucks_immediately_while_only_breathing_is_active() -> None:
    """Idle standby (breathing, no real move) must not wait out a grace window."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    manager = MovementManager(robot)
    manager.state.current_move = BreathingMove(np.eye(4, dtype=np.float32), (0.0, 0.0))

    manager._handle_command("set_standby", True, manager._now())

    assert manager._standby_tuck_at is None
    (tuck_move,) = manager.move_queue
    assert manager.state.current_move is None
    assert np.allclose(tuck_move.target_head_pose[:3, :3], np.eye(3))


def test_wake_during_the_grace_window_cancels_the_deferred_tuck() -> None:
    """Waking while the goodbye emotion still plays lifts the head immediately, grace forgotten."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    manager = MovementManager(robot)
    goodbye = _FakeMove(np.eye(4))
    goodbye.duration = 5.6
    now = manager._now()
    manager.state.current_move = goodbye
    manager.state.move_start_time = now - 0.2

    manager._handle_command("set_standby", True, now)
    assert manager._standby_tuck_at is not None

    manager._handle_command("set_standby", False, manager._now())

    assert manager._standby_tuck_at is None
    assert manager.state.current_move is None
    (lift_move,) = manager.move_queue
    assert lift_move.target_antennas == NEUTRAL_ANTENNAS


def test_goodbye_sequence_never_steps_the_commanded_head_pose() -> None:
    """End-to-end over the working loop: emotion start + standby tuck command no single-tick jump."""
    robot = MagicMock()
    commanded_heads: list[np.ndarray] = []
    commanded_antennas: list[tuple[float, float]] = []

    def record_set_target(head: np.ndarray, antennas: tuple[float, float], body_yaw: object) -> None:
        commanded_heads.append(head.copy())
        commanded_antennas.append((float(antennas[0]), float(antennas[1])))

    robot.set_target.side_effect = record_set_target

    # A faithful robot follows the commanded stream closely (impedance control),
    # so its sensors report the last commanded pose back.
    def current_sensor_head() -> np.ndarray:
        return commanded_heads[-1] if commanded_heads else np.eye(4, dtype=np.float32)

    robot.get_current_head_pose.side_effect = current_sensor_head
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])

    manager = MovementManager(robot)
    manager.start()
    try:
        # The goodbye emotion recording starts with its head yawed 14° away.
        recorded = MagicMock()
        recorded.get.return_value = _FakeMove(create_head_pose(0, 0, 0, 0, 0, 14, degrees=True))
        manager.queue_move(EmotionQueueMove("goodbye", recorded))
        assert _wait_for(lambda: manager.state.current_move is not None)
        assert _wait_for(lambda: len(commanded_heads) >= 12)

        # The goodbye keyword lands while the emotion plays; the tuck takes over
        # once the (forced) grace deadline passes.
        manager.set_standby(True)
        assert _wait_for(lambda: manager._standby_tuck_at is not None)
        manager._standby_tuck_at = manager._now() - 0.01
        assert _wait_for(
            lambda: manager._standby_tuck_at is None and isinstance(manager.state.current_move, GotoQueueMove)
        )
        assert _wait_for(lambda: len(commanded_heads) >= 24)
    finally:
        manager.stop(reset_to_neutral=False)

    # Before the fix the recording's first frame stepped the commanded head by
    # its full ~14° in one 60 Hz tick; blending keeps every step in the low degrees.
    max_step_deg = max(
        np.degrees(np.arccos(np.clip((np.trace(prev[:3, :3].T @ cur[:3, :3]) - 1) / 2, -1, 1)))
        for prev, cur in zip(commanded_heads, commanded_heads[1:])
    )
    assert max_step_deg < 5.0

    # The tuck folds the antennas ~165° to the sleep position; before easing that
    # sweep stepped from rest to full speed in one tick, which read as a twitch.
    antenna_deg = [np.degrees([left, right]) for left, right in commanded_antennas]
    tick_speeds = [float(np.linalg.norm(cur - prev)) for prev, cur in zip(antenna_deg, antenna_deg[1:])]
    max_accel = max(abs(cur - prev) for prev, cur in zip(tick_speeds, tick_speeds[1:]))
    assert max_accel < 0.5  # linear snap ~1.37°/tick; eased onset ~0.07°/tick per tick


def test_eased_goto_starts_and_ends_from_rest() -> None:
    """An eased goto covers less ground than a linear one early on, and reaches the same target."""
    neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    common = {
        "target_head_pose": neutral,
        "start_head_pose": neutral,
        "target_antennas": (1.0, -1.0),
        "start_antennas": (0.0, 0.0),
        "duration": 2.0,
    }
    eased = GotoQueueMove(**common, ease=True)
    linear = GotoQueueMove(**common)

    _, eased_antennas, _ = eased.evaluate(0.5)
    _, linear_antennas, _ = linear.evaluate(0.5)
    # A quarter of the way in time, smoothstep has covered 0.15625, not 0.25.
    assert eased_antennas[0] == pytest.approx(0.15625)
    assert linear_antennas[0] == pytest.approx(0.25)

    # Both reach the same final pose.
    _, eased_final, _ = eased.evaluate(eased.duration)
    _, linear_final, _ = linear.evaluate(linear.duration)
    assert np.allclose(eased_final, linear_final)
    assert np.allclose(eased_final, [1.0, -1.0])
