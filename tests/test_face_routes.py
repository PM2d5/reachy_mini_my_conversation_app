"""Tests for the faces JSON-RPC routes."""

import base64
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock
from collections.abc import Callable

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini.vision.face_detector import Face
from my_conversation_app.faces import enroll_face
from my_conversation_app.config import config
from my_conversation_app.face_routes import register_face_methods
from my_conversation_app.face_recognition import FaceRecognitionService
from my_conversation_app.tools.core_tools import ToolDependencies


class StubDetector:
    """Always returns one configured face."""

    def __init__(self, faces: list[Face]) -> None:
        """Configure the faces every detect() returns."""
        self._faces = faces

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        """Return the configured faces."""
        return self._faces


class StubEmbedder:
    """Always returns one L2-normalized vector."""

    def __init__(self, vector: list[float]) -> None:
        """Pre-normalize the one vector every embed() returns."""
        raw = np.array(vector, dtype=np.float32)
        self._vector = raw / np.linalg.norm(raw)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Return the configured vector."""
        return self._vector


def _client(
    instance_path: Path,
    get_camera_deps: Callable[[], ToolDependencies | None] | None = None,
) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()
    register_face_methods(rpc, instance_path=instance_path, get_camera_deps=get_camera_deps)
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
    assert listed[0]["nicknames"] == []
    assert listed[0]["embeddingCount"] == 2
    assert "createdAt" in listed[0] and "lastSeenAt" in listed[0]

    renamed = _rpc_call(_client(tmp_path), "faces.rename", {"id": enrolled.face.id, "name": " 蕾蕾 "})
    assert renamed["result"]["face"]["name"] == "蕾蕾"

    removed = _rpc_call(_client(tmp_path), "faces.remove", {"id": enrolled.face.id})
    assert removed["result"] == {"ok": True, "removed": "蕾蕾"}
    assert _rpc_call(_client(tmp_path), "faces.list")["result"] == {"faces": []}


def test_face_routes_set_nicknames(tmp_path: Path) -> None:
    """SetNicknames replaces the pool, reports collisions, and validates input."""
    first = enroll_face(tmp_path, "凯蕾", [[1.0, 0.0]])
    second = enroll_face(tmp_path, "李雷", [[0.0, 1.0]])
    assert first.face is not None
    assert second.face is not None
    client = _client(tmp_path)

    set_result = _rpc_call(client, "faces.setNicknames", {"id": first.face.id, "nicknames": [" 老凯 ", "凯蕾"]})
    assert set_result["result"]["face"]["nicknames"] == ["老凯"]

    duplicate = _rpc_call(client, "faces.setNicknames", {"id": second.face.id, "nicknames": ["老凯"]})
    assert duplicate["error"]["data"]["reason"] == "duplicate_name"

    invalid = _rpc_call(client, "faces.setNicknames", {"id": first.face.id, "nicknames": "老凯"})
    assert invalid["error"]["data"]["reason"] == "invalid_params"

    missing = _rpc_call(client, "faces.setNicknames", {"id": "missing", "nicknames": []})
    assert missing["error"]["data"]["reason"] == "face_not_found"

    cleared = _rpc_call(client, "faces.setNicknames", {"id": first.face.id, "nicknames": []})
    assert cleared["result"]["face"]["nicknames"] == []


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


def test_face_routes_inspect_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """InspectCamera returns the inspected frame, thresholds, and per-person scores."""
    monkeypatch.setattr(config, "FACE_MATCH_THRESHOLD", 0.5)
    assert enroll_face(tmp_path, "凯蕾", [[1.0, 0.0], [1.0, 0.0]]).face is not None
    assert enroll_face(tmp_path, "李雷", [[0.0, 1.0], [0.0, 1.0]]).face is not None

    face = Face(bbox=(10.0, 10.0, 60.0, 70.0), right_eye=(28.0, 34.0), left_eye=(52.0, 34.0), nose=(40.0, 52.0))
    recognizer = FaceRecognitionService(tmp_path, detector=StubDetector([face]), embedder=StubEmbedder([0.8, 0.6]))
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_enabled=True,
        face_recognizer=recognizer,
    )
    deps.reachy_mini.media.get_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

    result = _rpc_call(_client(tmp_path, get_camera_deps=lambda: deps), "faces.inspectCamera")["result"]

    assert result["image"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(result["image"].split(",", 1)[1])[:2] == b"\xff\xd8"
    assert result["threshold"] == 0.5
    assert result["progressiveThreshold"] == pytest.approx(0.58)
    assert result["faces"][0]["bbox"] == [10.0, 10.0, 60.0, 70.0]
    scores = result["faces"][0]["scores"]
    assert scores[0] == {"name": "凯蕾", "similarity": pytest.approx(0.8)}
    assert scores[1] == {"name": "李雷", "similarity": pytest.approx(0.6)}


def test_face_routes_label_face(tmp_path: Path) -> None:
    """LabelFace enrolls a frozen frame's face: create new, append to existing, guard staleness."""
    assert enroll_face(tmp_path, "凯蕾", [[1.0, 0.0], [0.9, 0.1]]).face is not None

    face = Face(bbox=(10.0, 10.0, 60.0, 70.0), right_eye=(28.0, 34.0), left_eye=(52.0, 34.0), nose=(40.0, 52.0))
    recognizer = FaceRecognitionService(tmp_path, detector=StubDetector([face]), embedder=StubEmbedder([0.8, 0.6]))
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_enabled=True,
        face_recognizer=recognizer,
    )
    deps.reachy_mini.media.get_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    client = _client(tmp_path, get_camera_deps=lambda: deps)

    inspected = _rpc_call(client, "faces.inspectCamera")["result"]
    frame_id = inspected["frameId"]

    # Regression for the freeze race: one more inspection can land server-side
    # after the client froze the previous payload — that frame must stay labelable.
    for _ in range(3):
        _rpc_call(client, "faces.inspectCamera")

    created = _rpc_call(client, "faces.labelFace", {"frameId": frame_id, "faceIndex": 0, "name": "李雷"})["result"]
    assert created["ok"] is True
    assert created["face"]["name"] == "李雷"
    assert created["face"]["embeddingCount"] == 1

    appended = _rpc_call(client, "faces.labelFace", {"frameId": frame_id, "faceIndex": 0, "name": "凯蕾"})["result"]
    assert appended["ok"] is True
    assert appended["face"]["name"] == "凯蕾"
    assert appended["face"]["embeddingCount"] == 3

    stale = _rpc_call(client, "faces.labelFace", {"frameId": 999999, "faceIndex": 0, "name": "王五"})
    assert stale["error"]["data"]["reason"] == "stale_frame"

    invalid_index = _rpc_call(client, "faces.labelFace", {"frameId": frame_id, "faceIndex": 7, "name": "王五"})
    assert invalid_index["error"]["data"]["reason"] == "face_not_found"

    invalid_name = _rpc_call(client, "faces.labelFace", {"frameId": frame_id, "faceIndex": 0, "name": "   "})
    assert invalid_name["error"]["data"]["reason"] == "invalid_params"

    non_integer = _rpc_call(client, "faces.labelFace", {"frameId": "latest", "faceIndex": 0, "name": "李雷"})
    assert non_integer["error"]["data"]["reason"] == "invalid_params"


def test_face_routes_inspect_camera_reports_disabled_states(tmp_path: Path) -> None:
    """A missing handler, a disabled camera, and missing recognition map to stable reasons."""
    offline = _rpc_call(_client(tmp_path, get_camera_deps=lambda: None), "faces.inspectCamera")
    assert offline["error"]["data"]["reason"] == "handler_unavailable"

    camera_off = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_enabled=False)
    disabled = _rpc_call(_client(tmp_path, get_camera_deps=lambda: camera_off), "faces.inspectCamera")
    assert disabled["error"]["data"]["reason"] == "camera_disabled"

    no_recognizer = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_enabled=True)
    unavailable = _rpc_call(_client(tmp_path, get_camera_deps=lambda: no_recognizer), "faces.inspectCamera")
    assert unavailable["error"]["data"]["reason"] == "face_recognition_unavailable"
