"""Tests for the faces JSON-RPC routes."""

from typing import Any
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from my_conversation_app.faces import enroll_face
from my_conversation_app.face_routes import register_face_methods


def _client(instance_path: Path) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()
    register_face_methods(rpc, instance_path=instance_path)
    rpc.mount(app)
    return TestClient(app)


def _rpc_call(client: TestClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = websocket.receive_json()
        return response


def test_face_routes_list_rename_remove(tmp_path: Path) -> None:
    """The routes list enrolled faces with camelCase payloads, rename, and remove."""
    assert _rpc_call(_client(tmp_path), "faces.list")["result"] == {"faces": []}

    enrolled = enroll_face(tmp_path, "凯蕾", [[1.0, 0.0], [0.9, 0.1]])
    assert enrolled.face is not None

    listed = _rpc_call(_client(tmp_path), "faces.list")["result"]["faces"]
    assert len(listed) == 1
    assert listed[0]["name"] == "凯蕾"
    assert listed[0]["embeddingCount"] == 2
    assert "createdAt" in listed[0] and "lastSeenAt" in listed[0]

    renamed = _rpc_call(_client(tmp_path), "faces.rename", {"id": enrolled.face.id, "name": " 蕾蕾 "})
    assert renamed["result"]["face"]["name"] == "蕾蕾"

    removed = _rpc_call(_client(tmp_path), "faces.remove", {"id": enrolled.face.id})
    assert removed["result"] == {"ok": True, "removed": "蕾蕾"}
    assert _rpc_call(_client(tmp_path), "faces.list")["result"] == {"faces": []}


def test_face_routes_report_missing_and_duplicate_errors(tmp_path: Path) -> None:
    """Renaming onto an existing name and touching unknown ids return stable reasons."""
    first = enroll_face(tmp_path, "凯蕾", [[1.0, 0.0], [0.9, 0.1]])
    second = enroll_face(tmp_path, "李雷", [[0.0, 1.0], [0.1, 0.9]])
    assert first.face is not None
    assert second.face is not None
    client = _client(tmp_path)

    duplicate = _rpc_call(client, "faces.rename", {"id": first.face.id, "name": "李雷"})
    assert duplicate["error"]["data"]["reason"] == "duplicate_name"

    empty_name = _rpc_call(client, "faces.rename", {"id": first.face.id, "name": "   "})
    assert empty_name["error"]["data"]["reason"] == "empty_name"

    missing_rename = _rpc_call(client, "faces.rename", {"id": "missing", "name": "新名"})
    assert missing_rename["error"]["data"]["reason"] == "face_not_found"

    missing_remove = _rpc_call(client, "faces.remove", {"id": "missing"})
    assert missing_remove["error"]["data"]["reason"] == "face_not_found"
