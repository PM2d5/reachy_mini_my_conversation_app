"""Tests for the remember_face tool and its local enrollment matcher."""

from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import my_conversation_app.tools.remember_face as remember_face_mod
from my_conversation_app.faces import EnrolledFace
from my_conversation_app.config import config
from my_conversation_app.face_recognition import EnrollmentOutcome
from my_conversation_app.tools.core_tools import ToolDependencies
from my_conversation_app.tools.remember_face import RememberFace, match_face_enrollment_command


class FakeRecognizer:
    """Stands in for FaceRecognitionService with a canned enroll outcome."""

    def __init__(self, outcome: EnrollmentOutcome) -> None:
        """Store the canned outcome for every enroll() call."""
        self.available = True
        self._outcome = outcome
        self.enroll_calls: list[tuple[str, int]] = []

    def load_models(self) -> bool:
        """Report the fake service as ready."""
        return True

    def enroll(self, name: str, frames: list[np.ndarray]) -> EnrollmentOutcome:
        """Record the call and return the canned outcome."""
        self.enroll_calls.append((name, len(frames)))
        return self._outcome


def _ok_outcome() -> EnrollmentOutcome:
    face = EnrolledFace(
        id="f_1",
        name="凯蕾",
        embeddings=((1.0, 0.0), (0.9, 0.1)),
        created_at=1,
        last_seen_at=1,
    )
    return EnrollmentOutcome(face=face, reason=None)


@pytest.fixture(autouse=True)
def _fast_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the inter-frame delay and reset the dedup window between tests."""
    monkeypatch.setattr(remember_face_mod, "_ENROLL_FRAME_GAP_S", 0.0)
    RememberFace._last_enrollment = None


def _deps(recognizer: Any | None) -> ToolDependencies:
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    movement_manager = MagicMock()
    movement_manager.is_moving.return_value = False
    return ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=movement_manager,
        camera_enabled=True,
        face_recognizer=recognizer,
    )


@pytest.mark.asyncio
async def test_remember_face_enrolls_from_captured_frames(tmp_path: Path) -> None:
    """A happy enrollment captures three frames and reports the stored name."""
    recognizer = FakeRecognizer(_ok_outcome())

    result = await RememberFace()(_deps(recognizer), name=" 凯蕾 ")

    assert result["saved"] == "凯蕾"
    assert result["face_id"] == "f_1"
    assert recognizer.enroll_calls == [("凯蕾", 3)]


@pytest.mark.asyncio
async def test_remember_face_rejects_empty_name_and_disabled_camera(tmp_path: Path) -> None:
    """Empty names and a disabled camera fail before any frame is captured."""
    recognizer = FakeRecognizer(_ok_outcome())

    empty_name = await RememberFace()(_deps(recognizer), name="   ")
    assert "error" in empty_name

    deps = _deps(recognizer)
    deps.camera_enabled = False
    disabled = await RememberFace()(deps, name="凯蕾")
    assert disabled["error"] == "Camera is disabled"
    assert recognizer.enroll_calls == []


@pytest.mark.asyncio
async def test_remember_face_reports_unavailable_service(tmp_path: Path) -> None:
    """Without a recognizer the tool degrades to an error, never raises."""
    unavailable = await RememberFace()(_deps(None), name="凯蕾")
    assert unavailable["error"] == "Face recognition is not available"

    failing = FakeRecognizer(_ok_outcome())
    failing.load_models = lambda: False  # type: ignore[method-assign]
    assert (await RememberFace()(_deps(failing), name="凯蕾"))["error"] == "Face recognition is not available"


@pytest.mark.asyncio
async def test_remember_face_collapses_duplicate_calls_in_the_dedup_window(tmp_path: Path) -> None:
    """A local trigger plus a model call for the same utterance enroll only once."""
    recognizer = FakeRecognizer(_ok_outcome())
    deps = _deps(recognizer)

    first = await RememberFace()(deps, name="凯蕾")
    second = await RememberFace()(deps, name="凯蕾")

    assert first["saved"] == "凯蕾"
    assert second["status"] == "already_enrolled"
    assert len(recognizer.enroll_calls) == 1


@pytest.mark.asyncio
async def test_remember_face_maps_store_failures_to_guidance(tmp_path: Path) -> None:
    """Store and capture failures become actionable errors for the model to relay."""
    recognizer = FakeRecognizer(EnrollmentOutcome(face=None, reason="too_few_faces"))
    too_few = await RememberFace()(_deps(recognizer), name="凯蕾")
    assert "clear face" in too_few["error"]

    recognizer_duplicate = FakeRecognizer(EnrollmentOutcome(face=None, reason="duplicate_name"))
    duplicate = await RememberFace()(_deps(recognizer_duplicate), name="凯蕾")
    assert duplicate["error"] == "this name is already enrolled"


def test_remember_face_availability_follows_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool disappears from the model when recognition is switched off."""
    monkeypatch.setattr(config, "FACE_RECOGNITION_ENABLED", True)
    assert RememberFace().is_available() is True

    monkeypatch.setattr(config, "FACE_RECOGNITION_ENABLED", False)
    assert RememberFace().is_available() is False


@pytest.mark.parametrize(
    "transcript",
    [
        "我叫凯蕾，记住我",
        "我叫凯蕾记住我",
        "请记住我，我叫凯蕾",
        "记住我，我的名字是李雷。",
        "叫我老王吧，记住我的脸",
        "我叫凯蕾，帮我记住我的脸",
        "我叫凯蕾，请帮我记住这张脸",
        "记住我，我叫凯蕾吧",
        "记住我这个人，我叫老王",
        "my name is Anna, remember me",
        "Remember me, my name is Anna Smith",
        "I'm Bob, please remember my face",
        "remember me, I'm Carol",
    ],
)
def test_matcher_extracts_names(transcript: str) -> None:
    """Introduction + remember-me phrases yield the introduced name."""
    name = match_face_enrollment_command(transcript)
    assert name is not None
    assert name not in ("", "我", "你")


@pytest.mark.parametrize(
    "transcript",
    [
        "",
        "记住我",
        "记住我喜欢喝咖啡",
        "我叫凯蕾，记住我喜欢喝咖啡",
        "我叫凯蕾",
        "帮我记住这个电话号码",
        "帮我记住这件事，我叫凯蕾",
        "remember the milk",
        "i'm leaving, remember me",
        "I'm tired, remember me",
        "讲个笑话",
    ],
)
def test_matcher_ignores_non_enrollments(transcript: str) -> None:
    """Bare commands without a name, facts, and unrelated text never trigger."""
    assert match_face_enrollment_command(transcript) is None
