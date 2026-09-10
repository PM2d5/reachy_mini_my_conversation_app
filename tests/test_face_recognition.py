"""Tests for the local face-recognition service."""

import time
from typing import Any
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import constants as hub_constants

from reachy_mini.vision.face_detector import Face
import my_conversation_app.face_recognition as face_recognition_mod
from my_conversation_app.faces import MAX_EMBEDDINGS_PER_FACE, list_enrolled_faces
from my_conversation_app.config import config
from my_conversation_app.face_recognition import (
    FaceRecognitionService,
    align_face,
)


class FakeDetector:
    """Yields configured faces per call and records the frames it saw."""

    def __init__(self, faces_per_frame: list[list[Face]] | None = None) -> None:
        """Configure which faces each successive detect() returns."""
        self._faces_per_frame = faces_per_frame or []
        self.seen_frames: list[np.ndarray] = []

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        """Return the configured faces and record the frame."""
        self.seen_frames.append(frame_bgr)
        index = min(len(self.seen_frames) - 1, len(self._faces_per_frame) - 1) if self._faces_per_frame else 0
        if not self._faces_per_frame:
            return []
        return self._faces_per_frame[index]


class FakeEmbedder:
    """Returns queued L2-normalized vectors in order, repeating the last one."""

    def __init__(self, vectors: list[list[float]]) -> None:
        """Queue one embedding vector per embed() call."""
        self._vectors = [np.array(vector, dtype=np.float32) for vector in vectors]
        self._last = self._vectors[0]
        self.inputs: list[np.ndarray] = []

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Return the next queued vector, L2-normalized."""
        self.inputs.append(aligned_bgr)
        if self._vectors:
            self._last = self._vectors.pop(0)
        return self._last / np.linalg.norm(self._last)


def _face(bbox: tuple[float, float, float, float] = (90.0, 90.0, 60.0, 70.0)) -> Face:
    x, y, w, h = bbox
    return Face(
        bbox=bbox,
        right_eye=(x + w * 0.3, y + h * 0.35),
        left_eye=(x + w * 0.7, y + h * 0.35),
        nose=(x + w * 0.5, y + h * 0.6),
    )


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _service(tmp_path: Path, detector: FakeDetector, embedder: FakeEmbedder) -> FaceRecognitionService:
    return FaceRecognitionService(tmp_path, detector=detector, embedder=embedder)


def test_load_models_failure_is_sticky(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed model load must disable recognition permanently, never raising."""

    def _raise() -> Any:
        raise RuntimeError("model download failed")

    monkeypatch.setattr(face_recognition_mod, "FaceDetector", _raise)
    service = FaceRecognitionService(None)

    assert service.load_models() is False
    assert service.available is False
    assert service.load_models() is False

    outcome = service.recognize(_frame())
    assert outcome.face_detected is False
    assert outcome.name is None


def test_recognize_matches_best_reference_and_reports_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recognition picks the best cosine across all people and references."""
    monkeypatch.setattr(config, "FACE_MATCH_THRESHOLD", 0.8)
    detector = FakeDetector([[_face()]])
    embedder = FakeEmbedder(
        [
            [1.0, 0.0],
            [1.0, 0.0],  # enroll 凯蕾
            [0.0, 1.0],
            [0.0, 1.0],  # enroll 李雷
            [0.99, 0.05],  # recognize → close to 凯蕾
            [0.707, 0.707],  # recognize → below threshold for both
        ]
    )
    service = _service(tmp_path, detector, embedder)

    enrolled_first = service.enroll("凯蕾", [_frame(), _frame()])
    enrolled_second = service.enroll("李雷", [_frame(), _frame()])
    assert enrolled_first.face is not None
    assert enrolled_second.face is not None

    matched = service.recognize(_frame())
    assert matched.face_detected is True
    assert matched.name == "凯蕾"
    assert matched.face_id == enrolled_first.face.id
    assert matched.similarity == pytest.approx(0.9987, abs=1e-3)

    unknown = service.recognize(_frame())
    assert unknown.face_detected is True
    assert unknown.name is None
    # Best cosine runs against every reference — including the one the strong
    # match just added to 凯蕾 (cos ≈ 0.742), which is the new best.
    assert unknown.similarity == pytest.approx(0.7418, abs=1e-3)

    kaili = next(face for face in list_enrolled_faces(tmp_path) if face.name == "凯蕾")
    assert kaili.last_seen_at >= enrolled_first.face.last_seen_at


def test_recognize_reports_no_face_when_detector_finds_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame without faces must be distinguishable from an unmatched face."""
    monkeypatch.setattr(config, "FACE_MATCH_THRESHOLD", 0.5)
    detector = FakeDetector([[_face()], [_face()], []])
    embedder = FakeEmbedder([[1.0, 0.0], [1.0, 0.0]])
    service = _service(tmp_path, detector, embedder)

    assert service.enroll("凯蕾", [_frame(), _frame()]).face is not None

    empty = service.recognize(_frame())
    assert empty.face_detected is False
    assert empty.name is None
    assert empty.similarity == 0.0


def test_strong_match_progressively_grows_the_reference_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A match well above the threshold feeds the embedding back, capped per person."""
    monkeypatch.setattr(config, "FACE_MATCH_THRESHOLD", 0.5)
    detector = FakeDetector([[_face()]])
    embedder = FakeEmbedder(
        [
            [1.0, 0.0],
            [1.0, 0.0],  # enroll with two references
            [0.99, 0.05],  # strong match (cos ≈ 0.999) → progressive add
            [0.52, 0.84],  # above 0.5, below 0.5+margin vs every reference → no add
        ]
    )
    service = _service(tmp_path, detector, embedder)

    enrolled = service.enroll("凯蕾", [_frame(), _frame()])
    assert enrolled.face is not None

    strong = service.recognize(_frame())
    assert strong.name == "凯蕾"
    assert len(list_enrolled_faces(tmp_path)[0].embeddings) == 3

    mild = service.recognize(_frame())
    assert mild.name == "凯蕾"
    assert len(list_enrolled_faces(tmp_path)[0].embeddings) == 3

    for _ in range(MAX_EMBEDDINGS_PER_FACE + 2):
        embedder._vectors.insert(0, [0.999, 0.01])
        service.recognize(_frame())
    assert len(list_enrolled_faces(tmp_path)[0].embeddings) == MAX_EMBEDDINGS_PER_FACE


def test_enroll_requires_two_clean_frames_and_maps_store_reasons(tmp_path: Path) -> None:
    """Enrollment skips faceless frames and surfaces the store's rejection reasons."""
    detector = FakeDetector([[_face()], [], [_face()]])
    embedder = FakeEmbedder([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    service = _service(tmp_path, detector, embedder)

    enrolled = service.enroll("凯蕾", [_frame(), _frame(), _frame()])
    assert enrolled.face is not None

    duplicate = service.enroll("凯蕾", [_frame(), _frame()])
    assert duplicate.reason == "duplicate_name"

    empty_detector = FakeDetector([])
    empty_service = _service(tmp_path, empty_detector, FakeEmbedder([[1.0, 0.0]]))
    too_few = empty_service.enroll("新人物", [_frame(), _frame()])
    assert too_few.reason == "too_few_faces"


def test_select_face_prefers_largest_then_most_central() -> None:
    """The interlocutor is the biggest face; ties go to the most central one."""
    small = _face((100.0, 100.0, 30.0, 30.0))
    large = _face((0.0, 0.0, 90.0, 90.0))
    assert FaceRecognitionService._select_face(_frame(), [small, large]) is large

    left = _face((0.0, 100.0, 50.0, 50.0))
    center = _face((135.0, 95.0, 50.0, 50.0))
    assert FaceRecognitionService._select_face(_frame(), [left, center]) is center
    assert FaceRecognitionService._select_face(_frame(), []) is None


def test_align_face_is_identity_for_template_landmarks_and_survives_swapped_eyes() -> None:
    """Landmarks already on the template must reproduce the crop, whatever the eye naming."""
    frame = np.arange(112 * 112 * 3, dtype=np.uint8).reshape(112, 112, 3)
    template = face_recognition_mod._ALIGNMENT_TEMPLATE
    right_eye = (float(template[0][0]), float(template[0][1]))
    left_eye = (float(template[1][0]), float(template[1][1]))
    nose = (float(template[2][0]), float(template[2][1]))

    straight = align_face(
        frame, Face(bbox=(30.0, 40.0, 52.0, 52.0), right_eye=right_eye, left_eye=left_eye, nose=nose)
    )
    swapped = align_face(frame, Face(bbox=(30.0, 40.0, 52.0, 52.0), right_eye=left_eye, left_eye=right_eye, nose=nose))

    assert straight.shape == (112, 112, 3)
    np.testing.assert_allclose(straight, frame, atol=1)
    np.testing.assert_allclose(swapped, frame, atol=1)


def test_seed_detector_model_cache_copies_bundled_model(tmp_path: Path) -> None:
    """Seeding places the bundled YuNet exactly where the HF cache expects it."""
    target = face_recognition_mod.seed_detector_model_cache(tmp_path)

    assert target == face_recognition_mod._detector_snapshot_path(tmp_path)
    assert target.is_file()
    assert target.read_bytes() == face_recognition_mod.BUNDLED_DETECTOR_MODEL.read_bytes()

    # Idempotent: a second seed never rewrites the file.
    target.write_bytes(b"placeholder")
    face_recognition_mod.seed_detector_model_cache(tmp_path)
    assert target.read_bytes() == b"placeholder"


def test_construct_detector_loads_bundled_model_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction serves the bundled model from a fresh cache, with no network."""
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    offline_before = hub_constants.HF_HUB_OFFLINE

    detector = face_recognition_mod._construct_detector()

    # The offline flag toggled only inside the construction window.
    assert hub_constants.HF_HUB_OFFLINE == offline_before
    assert face_recognition_mod._detector_snapshot_path(tmp_path / "hub").is_file()
    assert detector.detect(np.zeros((240, 320, 3), dtype=np.uint8)) == []


def test_run_with_timeout_surfaces_hangs() -> None:
    """A hanging load must surface as TimeoutError instead of wedging shutdown."""

    def _hang() -> None:
        time.sleep(5.0)

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        face_recognition_mod._run_with_timeout(_hang, timeout_s=0.05, what="hang probe")
    assert time.monotonic() - start < 2.0

    def _fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        face_recognition_mod._run_with_timeout(_fail, timeout_s=5.0, what="fail probe")
