"""Tests for the remember_face tool and its local enrollment matcher."""

from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import my_conversation_app.tools.remember_face as remember_face_mod
from my_conversation_app.faces import EnrolledFace
from my_conversation_app.config import config
from my_conversation_app.face_recognition import EnrollmentOutcome, RecognitionOutcome
from my_conversation_app.tools.core_tools import ToolDependencies
from my_conversation_app.tools.remember_face import RememberFace, match_face_enrollment_command


class FakeRecognizer:
    """Stands in for FaceRecognitionService with canned enroll/recognize outcomes."""

    def __init__(
        self,
        outcome: EnrollmentOutcome,
        recognition: RecognitionOutcome | None = None,
    ) -> None:
        """Store the canned outcomes for every enroll()/recognize() call."""
        self.available = True
        self._outcome = outcome
        self._recognition = recognition or RecognitionOutcome(
            name=None, face_id=None, similarity=0.0, face_detected=True
        )
        self.enroll_calls: list[tuple[str, int]] = []
        self.recognize_calls: list[np.ndarray] = []

    def load_models(self) -> bool:
        """Report the fake service as ready."""
        return True

    def enroll(self, name: str, frames: list[np.ndarray]) -> EnrollmentOutcome:
        """Record the call and return the canned outcome."""
        self.enroll_calls.append((name, len(frames)))
        return self._outcome

    def recognize(self, frame: np.ndarray) -> RecognitionOutcome:
        """Record the call and return the canned outcome."""
        self.recognize_calls.append(frame)
        return self._recognition


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


class FakeSpeakerRecognizer:
    """Stands in for SpeakerRecognitionService with a fixed enrollment embedding."""

    def __init__(self, embedding: np.ndarray) -> None:
        """Store the canned embedding for every embed_for_enrollment call."""
        self._embedding = embedding
        self.enrolled_samples: list[np.ndarray] = []

    def load_models(self) -> bool:
        """Report the fake service as ready."""
        return True

    def embed_for_enrollment(self, samples: np.ndarray) -> np.ndarray:
        """Record the call and return the canned embedding."""
        self.enrolled_samples.append(samples)
        return self._embedding


@pytest.mark.asyncio
async def test_remember_face_enrolls_the_utterances_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful face enrollment attaches the just-spoken utterance as voiceprint."""
    appended: list[tuple[object, str, list[list[float]]]] = []
    monkeypatch.setattr(
        remember_face_mod,
        "append_voice_embeddings",
        lambda instance, face_id, embeddings: (
            appended.append((instance, face_id, embeddings))
            or EnrolledFace(id=face_id, name="凯蕾", embeddings=(), created_at=1, last_seen_at=1)
        ),
    )
    recognizer = FakeRecognizer(_ok_outcome())
    speaker = FakeSpeakerRecognizer(np.array([0.25, 1.0], dtype=np.float32))
    deps = _deps(recognizer)
    deps.instance_path = tmp_path
    deps.speaker_recognizer = speaker  # type: ignore[assignment]
    utterance = np.ones(32000, dtype=np.int16)  # 2 s at 16 kHz
    deps.get_last_user_speech = lambda: (16000, utterance)

    result = await RememberFace()(deps, name="凯蕾")

    assert result["voice_enrolled"] is True
    assert appended == [(tmp_path, "f_1", [[0.25, 1.0]])]
    assert speaker.enrolled_samples[0].shape == (32000,)


@pytest.mark.asyncio
async def test_remember_face_skips_voice_when_utterance_missing(tmp_path: Path) -> None:
    """No captured utterance or no speaker service means voice_enrolled is False."""
    recognizer = FakeRecognizer(_ok_outcome())
    speaker = FakeSpeakerRecognizer(np.array([0.25, 1.0], dtype=np.float32))

    no_speech = _deps(recognizer)
    no_speech.instance_path = tmp_path
    no_speech.speaker_recognizer = speaker  # type: ignore[assignment]
    no_speech.get_last_user_speech = lambda: None
    assert (await RememberFace()(no_speech, name="凯蕾"))["voice_enrolled"] is False

    RememberFace._last_enrollment = None  # the 60 s dedupe window spans the test
    no_service = _deps(recognizer)
    no_service.instance_path = tmp_path
    no_service.get_last_user_speech = lambda: (16000, np.ones(32000, dtype=np.int16))
    assert (await RememberFace()(no_service, name="凯蕾"))["voice_enrolled"] is False
    assert speaker.enrolled_samples == []


def _existing_person() -> EnrolledFace:
    return EnrolledFace(id="f_1", name="凯蕾", embeddings=((1.0, 0.0),), created_at=1, last_seen_at=1)


@pytest.mark.asyncio
async def test_reenrollment_attaches_voice_to_matching_existing_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-saying 我叫X，记住我 attaches the voice when the face confirms the name."""
    appended: list[tuple[object, str, list[list[float]]]] = []
    monkeypatch.setattr(
        remember_face_mod,
        "append_voice_embeddings",
        lambda instance, face_id, embeddings: (
            appended.append((instance, face_id, embeddings))
            or EnrolledFace(id=face_id, name="凯蕾", embeddings=(), created_at=1, last_seen_at=1)
        ),
    )
    monkeypatch.setattr(remember_face_mod, "list_enrolled_faces", lambda _instance: [_existing_person()])
    recognizer = FakeRecognizer(
        EnrollmentOutcome(face=None, reason="duplicate_name"),
        recognition=RecognitionOutcome(name="凯蕾", face_id="f_1", similarity=0.9, face_detected=True),
    )
    speaker = FakeSpeakerRecognizer(np.array([0.25, 1.0], dtype=np.float32))
    deps = _deps(recognizer)
    deps.instance_path = tmp_path
    deps.speaker_recognizer = speaker  # type: ignore[assignment]
    deps.get_last_user_speech = lambda: (16000, np.ones(32000, dtype=np.int16))

    result = await RememberFace()(deps, name="凯蕾")

    assert result["saved"] == "凯蕾"
    assert result["face_id"] == "f_1"
    assert result["voice_enrolled"] is True
    assert appended == [(tmp_path, "f_1", [[0.25, 1.0]])]


@pytest.mark.asyncio
async def test_reenrollment_rejects_a_face_that_is_not_that_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stranger re-saying someone's name must not glue their voice onto the record."""
    appended: list[tuple[object, str, list[list[float]]]] = []
    monkeypatch.setattr(
        remember_face_mod,
        "append_voice_embeddings",
        lambda instance, face_id, embeddings: (
            appended.append((instance, face_id, embeddings))
            or EnrolledFace(id=face_id, name="凯蕾", embeddings=(), created_at=1, last_seen_at=1)
        ),
    )
    monkeypatch.setattr(remember_face_mod, "list_enrolled_faces", lambda _instance: [_existing_person()])
    recognizer = FakeRecognizer(
        EnrollmentOutcome(face=None, reason="duplicate_name"),
        recognition=RecognitionOutcome(name="老婆", face_id="f_other", similarity=0.9, face_detected=True),
    )
    deps = _deps(recognizer)
    deps.instance_path = tmp_path
    deps.speaker_recognizer = FakeSpeakerRecognizer(np.array([0.25, 1.0], dtype=np.float32))
    deps.get_last_user_speech = lambda: (16000, np.ones(32000, dtype=np.int16))

    result = await RememberFace()(deps, name="凯蕾")

    assert result["error"] == "this name is already enrolled"
    assert appended == []
