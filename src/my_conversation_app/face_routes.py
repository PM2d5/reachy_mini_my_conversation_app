"""JSON-RPC methods for managing enrolled faces."""

import base64
import asyncio
import logging
from io import BytesIO
from typing import Any
from pathlib import Path
from itertools import count
from collections import deque
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from PIL import Image
from numpy.typing import NDArray

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from my_conversation_app.faces import (
    EnrolledFace,
    set_face_nicknames,
    list_enrolled_faces,
    normalize_face_name,
    remove_enrolled_face,
    rename_enrolled_face,
)
from my_conversation_app.config import config
from my_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _InspectedFrame:
    """One inspection frame, addressable by id for manual labels."""

    frame_id: int
    frame: NDArray[np.uint8]


# The client's freeze can land while one more inspection is already in flight
# (server-side it bumps the id after the client froze the previous payload), so
# labels resolve against a short window of recent frames, not just the newest.
_INSPECTED_FRAME_WINDOW = 8


_LABEL_ERRORS = {
    "face_not_found": "That face is not in the frozen frame.",
    "alignment_failed": "That face's landmarks could not be aligned.",
    "unavailable": "Face recognition is not available.",
    "duplicate_name": "That name is already enrolled.",
}


def _face_payload(face: EnrolledFace) -> dict[str, object]:
    return {
        "id": face.id,
        "name": face.name,
        "nicknames": list(face.nicknames),
        "embeddingCount": len(face.embeddings),
        "createdAt": face.created_at,
        "lastSeenAt": face.last_seen_at,
    }


def _encode_frame_jpeg(frame_bgr: NDArray[np.uint8]) -> str:
    """Encode the inspected frame itself, so overlay boxes match the picture."""
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=80)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def register_face_methods(
    rpc: JsonRpcServer,
    *,
    instance_path: str | Path | None,
    get_camera_deps: Callable[[], ToolDependencies | None] | None = None,
) -> None:
    """Register faces.list / rename / setNicknames / remove / inspectCamera / labelFace."""
    inspected_frames: deque[_InspectedFrame] = deque(maxlen=_INSPECTED_FRAME_WINDOW)
    frame_id_counter = count(1)

    async def _list_faces(_params: dict[str, Any]) -> dict[str, object]:
        faces = await asyncio.to_thread(list_enrolled_faces, instance_path)
        return {"faces": [_face_payload(face) for face in faces]}

    async def _rename_face(params: dict[str, Any]) -> dict[str, object]:
        face_id = str(params.get("id", "")).strip()
        name = normalize_face_name(str(params.get("name", "")))
        if not face_id:
            raise JsonRpcError("faces.rename requires 'id'", reason="invalid_params", code=-32602)
        if not name:
            raise JsonRpcError("A name is required.", reason="empty_name", code=-32602)
        try:
            face = await asyncio.to_thread(rename_enrolled_face, instance_path, face_id, name)
        except ValueError as exc:
            raise JsonRpcError("That name is already enrolled.", reason=str(exc)) from exc
        if face is None:
            raise JsonRpcError("That person is no longer enrolled.", reason="face_not_found")
        return {"ok": True, "face": _face_payload(face)}

    async def _set_nicknames(params: dict[str, Any]) -> dict[str, object]:
        face_id = str(params.get("id", "")).strip()
        raw_nicknames = params.get("nicknames")
        if not face_id:
            raise JsonRpcError("faces.setNicknames requires 'id'", reason="invalid_params", code=-32602)
        if not isinstance(raw_nicknames, list) or not all(isinstance(item, str) for item in raw_nicknames):
            raise JsonRpcError(
                "faces.setNicknames requires 'nicknames' as a list of strings.",
                reason="invalid_params",
                code=-32602,
            )
        try:
            face = await asyncio.to_thread(set_face_nicknames, instance_path, face_id, raw_nicknames)
        except ValueError as exc:
            raise JsonRpcError("That name is already enrolled.", reason=str(exc)) from exc
        if face is None:
            raise JsonRpcError("That person is no longer enrolled.", reason="face_not_found")
        return {"ok": True, "face": _face_payload(face)}

    async def _remove_face(params: dict[str, Any]) -> dict[str, object]:
        face_id = str(params.get("id", "")).strip()
        if not face_id:
            raise JsonRpcError("faces.remove requires 'id'", reason="invalid_params", code=-32602)
        face = await asyncio.to_thread(remove_enrolled_face, instance_path, face_id)
        if face is None:
            raise JsonRpcError("That person is no longer enrolled.", reason="face_not_found")
        return {"ok": True, "removed": face.name}

    async def _inspect_camera(_params: dict[str, Any]) -> dict[str, object]:
        """Read-only live match of one robot-camera frame against every enrolled person."""
        deps = get_camera_deps() if get_camera_deps is not None else None
        if deps is None:
            raise JsonRpcError("The robot is not connected.", reason="handler_unavailable")
        if not deps.camera_enabled:
            raise JsonRpcError("Camera is disabled.", reason="camera_disabled")
        recognizer = deps.face_recognizer
        if recognizer is None or not config.FACE_RECOGNITION_ENABLED:
            raise JsonRpcError("Face recognition is not available.", reason="face_recognition_unavailable")
        if not await asyncio.to_thread(recognizer.load_models):
            raise JsonRpcError("Face recognition is not available.", reason="face_recognition_unavailable")
        frame = await asyncio.to_thread(deps.reachy_mini.media.get_frame)
        if frame is None:
            raise JsonRpcError("No camera frame available.", reason="no_frame")
        inspections = await asyncio.to_thread(recognizer.inspect_faces, frame)
        image = await asyncio.to_thread(_encode_frame_jpeg, frame)
        frame_id = next(frame_id_counter)
        inspected_frames.append(_InspectedFrame(frame_id=frame_id, frame=frame))
        return {
            "frameId": frame_id,
            "image": image,
            "threshold": config.FACE_MATCH_THRESHOLD,
            "progressiveThreshold": config.FACE_MATCH_THRESHOLD + recognizer.PROGRESSIVE_MATCH_MARGIN,
            "faces": [
                {
                    "bbox": [round(value, 1) for value in inspection.bbox],
                    "scores": [
                        {"name": name, "similarity": round(similarity, 4)} for name, similarity in inspection.scores
                    ],
                }
                for inspection in inspections
            ],
        }

    async def _label_face(params: dict[str, Any]) -> dict[str, object]:
        """Attach one face of the frozen inspection frame to a person by name."""
        deps = get_camera_deps() if get_camera_deps is not None else None
        if deps is None:
            raise JsonRpcError("The robot is not connected.", reason="handler_unavailable")
        recognizer = deps.face_recognizer
        if recognizer is None:
            raise JsonRpcError("Face recognition is not available.", reason="face_recognition_unavailable")

        name = normalize_face_name(str(params.get("name", "")))
        if not name:
            raise JsonRpcError("A name is required.", reason="invalid_params", code=-32602)
        raw_frame_id = params.get("frameId")
        raw_face_index = params.get("faceIndex")
        if not isinstance(raw_frame_id, int) or not isinstance(raw_face_index, int):
            raise JsonRpcError("frameId and faceIndex must be integers.", reason="invalid_params", code=-32602)
        # Detection on a fixed frame is deterministic, so the cached frame keeps
        # the indexed face identical to the one the user froze and clicked.
        frame = next(
            (entry.frame for entry in inspected_frames if entry.frame_id == raw_frame_id),
            None,
        )
        if frame is None:
            raise JsonRpcError("The frozen frame is no longer available; freeze again.", reason="stale_frame")

        outcome = await asyncio.to_thread(recognizer.enroll_face_at, frame, raw_face_index, name)
        if outcome.face is None:
            reason = outcome.reason or "label_failed"
            raise JsonRpcError(_LABEL_ERRORS.get(reason, f"Labeling failed ({reason})."), reason=reason)
        return {"ok": True, "face": _face_payload(outcome.face)}

    rpc.register("faces.list", _list_faces)
    rpc.register("faces.rename", _rename_face)
    rpc.register("faces.setNicknames", _set_nicknames)
    rpc.register("faces.remove", _remove_face)
    rpc.register("faces.inspectCamera", _inspect_camera)
    rpc.register("faces.labelFace", _label_face)
