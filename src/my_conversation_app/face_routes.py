"""JSON-RPC methods for managing enrolled faces."""

import asyncio
import logging
from typing import Any
from pathlib import Path

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from my_conversation_app.faces import (
    EnrolledFace,
    list_enrolled_faces,
    normalize_face_name,
    remove_enrolled_face,
    rename_enrolled_face,
)


logger = logging.getLogger(__name__)


def _face_payload(face: EnrolledFace) -> dict[str, object]:
    return {
        "id": face.id,
        "name": face.name,
        "embeddingCount": len(face.embeddings),
        "createdAt": face.created_at,
        "lastSeenAt": face.last_seen_at,
    }


def register_face_methods(rpc: JsonRpcServer, *, instance_path: str | Path | None) -> None:
    """Register faces.list / faces.rename / faces.remove."""

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

    async def _remove_face(params: dict[str, Any]) -> dict[str, object]:
        face_id = str(params.get("id", "")).strip()
        if not face_id:
            raise JsonRpcError("faces.remove requires 'id'", reason="invalid_params", code=-32602)
        face = await asyncio.to_thread(remove_enrolled_face, instance_path, face_id)
        if face is None:
            raise JsonRpcError("That person is no longer enrolled.", reason="face_not_found")
        return {"ok": True, "removed": face.name}

    rpc.register("faces.list", _list_faces)
    rpc.register("faces.rename", _rename_face)
    rpc.register("faces.remove", _remove_face)
