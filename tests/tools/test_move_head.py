"""Tests for the move_head tool."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from my_conversation_app.tools.move_head import LOOK_HOLD_S, MoveHead
from my_conversation_app.tools.core_tools import ToolDependencies
from my_conversation_app.dance_emotion_moves import GotoQueueMove


def _deps(motion_duration_s: float = 1.0) -> tuple[ToolDependencies, MagicMock]:
    reachy_mini = MagicMock()
    reachy_mini.get_current_head_pose.return_value = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    reachy_mini.get_current_joint_positions.return_value = ([0.0], [0.1, -0.1])
    movement_manager = MagicMock()
    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=movement_manager,
        motion_duration_s=motion_duration_s,
    )
    return deps, movement_manager


def _queued_moves(movement_manager: MagicMock) -> list[GotoQueueMove]:
    return [call.args[0] for call in movement_manager.queue_move.call_args_list]


@pytest.mark.asyncio
async def test_move_head_turns_then_holds_the_gaze() -> None:
    """A turn is followed by a hold so a follow-up camera call still sees the view."""
    deps, movement_manager = _deps(motion_duration_s=1.5)

    result = await MoveHead()(deps, direction="left")

    assert result["status"] == "looking left"
    goto_move, hold_move = _queued_moves(movement_manager)
    assert goto_move.duration == 1.5
    left_pose = create_head_pose(0, 0, 0, 0, 0, 40, degrees=True)
    assert np.allclose(goto_move.target_head_pose, left_pose)
    assert hold_move.duration == LOOK_HOLD_S
    # The hold freezes the head exactly where the turn ended.
    assert np.allclose(hold_move.start_head_pose, left_pose)
    assert np.allclose(hold_move.target_head_pose, left_pose)
    # Only the turn counts as settling motion; the hold must not delay a camera capture.
    movement_manager.set_moving_state.assert_called_once_with(1.5)


def test_move_head_keeps_the_turn_to_chain_the_camera() -> None:
    """The model must get a response turn after moving, so it can call the camera next."""
    assert MoveHead.needs_response is True


@pytest.mark.asyncio
async def test_move_head_rejects_non_string_direction() -> None:
    """A malformed direction is reported as an error instead of raising."""
    deps, movement_manager = _deps()

    result = await MoveHead()(deps, direction=7)

    assert "error" in result
    movement_manager.queue_move.assert_not_called()


@pytest.mark.asyncio
async def test_move_head_reports_hardware_failure() -> None:
    """A robot I/O failure is returned as an error, not raised into the loop."""
    deps, movement_manager = _deps()
    deps.reachy_mini.get_current_head_pose.side_effect = OSError("pose read failed")

    result = await MoveHead()(deps, direction="left")

    assert "error" in result
    movement_manager.queue_move.assert_not_called()
