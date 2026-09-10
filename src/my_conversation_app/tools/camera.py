import base64
import asyncio
import logging
from typing import Any, Dict

from my_conversation_app.config import config
from my_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# One control-loop tick of the movement manager, plus the poll interval while a
# commanded motion is still settling.
_MOTION_SETTLE_POLL_S = 0.05


class Camera(Tool):
    """Take a picture with the camera to see what is in front of the robot."""

    name = "camera"
    description = (
        "Take a picture with the camera to see what is in front of the robot. "
        "This tool is your only eyes — call it BEFORE answering any visual question; "
        "never claim you cannot see. "
        "Use this when the user asks you to look at something, see what they are holding, "
        "check their appearance, describe the scene, or comment on how they look. "
        "Also use it when the user asks what you can see or wants your visual opinion. "
        "The camera is live, each call captures the current moment, and it waits for any "
        "head motion to finish before capturing. "
        "If the user asks about a direction (left, right, up, down), call move_head first "
        "to point the head there, then call this tool. "
        "If the user asks you to look without saying at what, do not ask for clarification, call this tool and describe what you see. "
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What to observe or ask about in the picture. "
                    "Examples: what is the user holding, describe the user's outfit, "
                    "what do you see around you, how does the user look today."
                ),
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and return the base64-encoded JPEG."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", question[:120])

        if not deps.camera_enabled:
            logger.error("Camera is disabled")
            return {"error": "Camera is disabled"}

        # A move_head and a camera call can arrive in the same model turn and run as
        # parallel tasks; wait out any in-flight motion so the frame matches where the
        # head is pointing instead of catching it mid-turn.
        await asyncio.sleep(_MOTION_SETTLE_POLL_S)
        while deps.movement_manager.is_moving():
            await asyncio.sleep(_MOTION_SETTLE_POLL_S)

        jpeg_bytes = deps.reachy_mini.media.get_frame_jpeg()
        if jpeg_bytes is None:
            logger.error("No frame available from camera")
            return {"error": "No frame available"}

        result: Dict[str, Any] = {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8"), "question": question}
        # Piggyback recognition on this shot: it updates last-seen and tells the
        # model who is in frame — same capture moment, no extra framing delay.
        face_note = await self._recognize_visible_face(deps)
        if face_note is not None:
            result["face"] = face_note
        return result

    async def _recognize_visible_face(self, deps: ToolDependencies) -> Dict[str, Any] | None:
        """Recognize the dominant face of a fresh frame; None when unavailable."""
        recognizer = deps.face_recognizer
        if recognizer is None or not config.FACE_RECOGNITION_ENABLED or not recognizer.available:
            return None
        try:
            frame = await asyncio.to_thread(deps.reachy_mini.media.get_frame)
            if frame is None:
                return None
            outcome = await asyncio.to_thread(recognizer.recognize, frame)
        except Exception as exc:
            logger.warning("Face recognition on camera frame failed: %s", exc)
            return None
        if not outcome.face_detected:
            return None
        if outcome.face_id is None:
            return {"name": None, "relation": "unknown"}
        identity = deps.current_identity
        relation = "user" if identity is not None and identity.face_id == outcome.face_id else "other"
        return {"name": outcome.name, "relation": relation}
