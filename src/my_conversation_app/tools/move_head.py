import logging
from typing import Any, Dict, Tuple, Literal

from reachy_mini.utils import create_head_pose
from my_conversation_app.tools.core_tools import Tool, ToolDependencies
from my_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

Direction = Literal["left", "right", "up", "down", "front"]

# Keeps the gaze in place after the turn so a follow-up camera call captures the
# view; without it breathing pulls the head back to neutral within ~1.3s.
LOOK_HOLD_S = 4.0


class MoveHead(Tool):
    """Move head in a given direction."""

    name = "move_head"
    description = (
        "Turn your head to look in a given direction: left, right, up, down or front. "
        "The head then stays pointed there for a few seconds before returning to center. "
        "When the user asks what you can see in a direction, call this first, then call "
        "the camera tool to capture and describe the view."
    )
    needs_response = True
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
            },
        },
        "required": ["direction"],
    }

    # mapping: direction -> args for create_head_pose
    DELTAS: Dict[str, Tuple[int, int, int, int, int, int]] = {
        "left": (0, 0, 0, 0, 0, 40),
        "right": (0, 0, 0, 0, 0, -40),
        "up": (0, 0, 0, 0, -30, 0),
        "down": (0, 0, 0, 0, 30, 0),
        "front": (0, 0, 0, 0, 0, 0),
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Move head in a given direction, then hold the gaze briefly."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str):
            return {"error": "direction must be a string"}
        direction: Direction = direction_raw  # type: ignore[assignment]
        logger.info("Tool call: move_head direction=%s", direction)

        deltas = self.DELTAS.get(direction, self.DELTAS["front"])
        target = create_head_pose(*deltas, degrees=True)

        # Use new movement manager
        try:
            movement_manager = deps.movement_manager

            # Get current state for interpolation
            current_head_pose = deps.reachy_mini.get_current_head_pose()
            _, current_antennas = deps.reachy_mini.get_current_joint_positions()

            # Create goto move
            goto_move = GotoQueueMove(
                target_head_pose=target,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),  # Reset antennas to default
                start_antennas=(
                    current_antennas[0],
                    current_antennas[1],
                ),  # Skip body_yaw
                target_body_yaw=0,  # Reset body yaw
                start_body_yaw=current_antennas[0],  # body_yaw is first in joint positions
                duration=deps.motion_duration_s,
            )
            hold_move = GotoQueueMove(
                target_head_pose=target,
                start_head_pose=target,
                target_antennas=(0, 0),
                start_antennas=(0, 0),
                target_body_yaw=0,
                start_body_yaw=0,
                duration=LOOK_HOLD_S,
            )

            movement_manager.queue_move(goto_move)
            movement_manager.queue_move(hold_move)
            # Only the goto settles the head; the hold must not delay a camera capture.
            movement_manager.set_moving_state(deps.motion_duration_s)

            return {
                "status": f"looking {direction}",
                "gaze_held_for_s": LOOK_HOLD_S,
                "next": "if the user asked what is there, call camera now to capture that view",
            }

        except Exception as e:
            logger.error("move_head failed: %s", e)
            return {"error": f"move_head failed: {type(e).__name__}: {e}"}
