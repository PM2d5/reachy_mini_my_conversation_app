"""Local face recognition: detect, align, embed, and match enrolled people."""

import shutil
import logging
import threading
from typing import TypeVar
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from huggingface_hub import constants as hub_constants
from huggingface_hub import try_to_load_from_cache

from reachy_mini.vision.face_detector import Face, FaceDetector
from my_conversation_app.faces import (
    EnrolledFace,
    enroll_face,
    mark_face_seen,
    list_enrolled_faces,
)
from my_conversation_app.config import config


logger = logging.getLogger(__name__)

BUNDLED_EMBEDDING_MODEL = Path(__file__).parent / "vision_models" / "face_embedding_w600k_mbf.onnx"
_EMBEDDING_INPUT_SIZE = 112
# An enrollment needs at least two clean frames; one alone matches too loosely.
_MIN_ENROLL_EMBEDDINGS = 2

# Mirror of the SDK FaceDetector's pinned model (reachy_mini.vision.face_detector):
# bundled so the first run needs no network, and a flaky connection cannot hang
# the session loop in hf_hub_download's retry chain.
_DETECTOR_REPO = "pollen-robotics/face_detection_yunet_2026may"
_DETECTOR_FILENAME = "face_detection_yunet_2026may.onnx"
_DETECTOR_REVISION = "2b8e922362946a0db67e861bae0f77826980effd"
BUNDLED_DETECTOR_MODEL = Path(__file__).parent / "vision_models" / _DETECTOR_FILENAME

# Bounds even a worst-case hub retry chain (~65 s) so a dead network degrades
# face recognition instead of wedging shutdown for minutes.
_MODEL_LOAD_TIMEOUT_S = 75.0

# ArcFace reference landmarks on the aligned 112x112 crop: image-left eye,
# image-right eye, nose. The standard template's mouth corners stay unused —
# the SDK detector only provides eyes and nose.
_ALIGNMENT_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SessionIdentity:
    """The person the current conversation belongs to."""

    name: str
    face_id: str
    source: str = "face"


@dataclass(frozen=True)
class RecognitionOutcome:
    """Result of matching one frame against the enrolled faces."""

    name: str | None
    face_id: str | None
    similarity: float
    face_detected: bool


@dataclass(frozen=True)
class EnrollmentOutcome:
    """Result of enrolling a person from captured frames."""

    face: EnrolledFace | None
    reason: str | None  # None on success; "too_few_faces" or the store's reason


def _estimate_affine(src_points: NDArray[np.float64], dst_points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve the exact 3-point affine mapping src -> dst as a 2x3 matrix."""
    coefficients = np.column_stack((src_points, np.ones(3)))
    return np.linalg.solve(coefficients, dst_points).T


def _warp_affine(
    frame_bgr: NDArray[np.uint8],
    matrix: NDArray[np.float64],
    size: int = _EMBEDDING_INPUT_SIZE,
) -> NDArray[np.uint8]:
    """Sample an inverse-mapped size x size crop with bilinear interpolation."""
    height, width = frame_bgr.shape[:2]
    inverse = np.linalg.inv(np.vstack((matrix, (0.0, 0.0, 1.0))))[:2, :]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    ones = np.ones_like(xs)
    src_x, src_y = inverse @ np.stack((xs.ravel(), ys.ravel(), ones.ravel()))

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    weight_x = src_x - x0
    weight_y = src_y - y0
    inside = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x0 + 1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y0 + 1, 0, height - 1)

    frame = frame_bgr.astype(np.float64)
    top = frame[y0c, x0c] * (1.0 - weight_x)[:, np.newaxis] + frame[y0c, x1c] * weight_x[:, np.newaxis]
    bottom = frame[y1c, x0c] * (1.0 - weight_x)[:, np.newaxis] + frame[y1c, x1c] * weight_x[:, np.newaxis]
    sampled = top * (1.0 - weight_y)[:, np.newaxis] + bottom * weight_y[:, np.newaxis]
    sampled[~inside] = 0.0
    return np.asarray(np.clip(sampled, 0.0, 255.0).round().reshape(size, size, 3), dtype=np.uint8)


def align_face(frame_bgr: NDArray[np.uint8], face: Face) -> NDArray[np.uint8]:
    """Warp a detected face onto the canonical eye/nose template."""
    # Order the eyes by image x so the detector's naming convention never matters.
    left_first = sorted((face.right_eye, face.left_eye), key=lambda point: point[0])
    src_points = np.array([left_first[0], left_first[1], face.nose], dtype=np.float64)
    matrix = _estimate_affine(src_points, _ALIGNMENT_TEMPLATE)
    return _warp_affine(frame_bgr, matrix)


class FaceEmbedder:
    """Bundle-shipped MobileFaceNet-class ONNX embedding model."""

    def __init__(self, model_path: str | Path) -> None:
        """Load the embedding ONNX confined to a single CPU thread."""
        options = ort.SessionOptions()
        # One thread, like the SDK's YuNet detector: never starve the audio loop.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, aligned_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return the L2-normalized embedding of one aligned 112x112 BGR crop."""
        blob = (aligned_bgr.astype(np.float32).transpose(2, 0, 1)[np.newaxis] - 127.5) / 127.5
        embedding = np.asarray(self._session.run(None, {self._input_name: blob})[0][0], dtype=np.float32)
        return np.asarray(embedding / np.linalg.norm(embedding), dtype=np.float32)


_T = TypeVar("_T")


def _run_with_timeout(work: Callable[[], _T], timeout_s: float, what: str) -> _T:
    """Run work in a daemon thread; a hang surfaces as TimeoutError instead of wedging."""
    outcome: list[_T] = []
    failure: list[BaseException] = []
    finished = threading.Event()

    def _worker() -> None:
        try:
            outcome.append(work())
        except BaseException as exc:
            failure.append(exc)
        finally:
            finished.set()

    # Daemon on purpose: even a timeout-abandoned attempt must never delay exit.
    threading.Thread(target=_worker, name="face-model-load", daemon=True).start()
    if not finished.wait(timeout_s):
        raise TimeoutError(f"{what} did not finish within {timeout_s:.0f}s (is the network unreachable?)")
    if failure:
        raise failure[0]
    return outcome[0]


def _detector_snapshot_path(cache_dir: Path) -> Path:
    """Return the HF-cache snapshot path the pinned YuNet revision occupies."""
    return (
        cache_dir
        / f"models--{_DETECTOR_REPO.replace('/', '--')}"
        / "snapshots"
        / _DETECTOR_REVISION
        / _DETECTOR_FILENAME
    )


def seed_detector_model_cache(cache_dir: Path) -> Path:
    """Copy the bundled YuNet into the HF cache; idempotent, returns the snapshot path."""
    snapshot_path = _detector_snapshot_path(cache_dir)
    if not snapshot_path.is_file():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BUNDLED_DETECTOR_MODEL, snapshot_path)
    return snapshot_path


def _construct_detector() -> FaceDetector:
    """Build the YuNet detector, served from the local cache instead of the network."""
    try:
        seed_detector_model_cache(Path(hub_constants.HF_HUB_CACHE))
    except OSError as exc:
        logger.warning("Could not seed the HF cache with the bundled detector: %s", exc)

    cached_path = try_to_load_from_cache(_DETECTOR_REPO, _DETECTOR_FILENAME, revision=_DETECTOR_REVISION)
    if not isinstance(cached_path, str):
        # Nothing usable on disk (read-only cache?) — the SDK's own download is
        # the last resort and is bounded by _MODEL_LOAD_TIMEOUT_S at the caller.
        logger.warning("Bundled YuNet not found in the HF cache; letting the SDK fetch it")
        return FaceDetector()

    # Flip the hub's request-time offline flag just for the construction window:
    # hf_hub_download then serves the cached snapshot instantly, and the flaky
    # network cannot add a minute of HEAD retries. No other hub traffic runs at
    # session start, so the process-wide window is harmless; always restored.
    was_offline = hub_constants.HF_HUB_OFFLINE
    hub_constants.HF_HUB_OFFLINE = True
    try:
        return FaceDetector()
    finally:
        hub_constants.HF_HUB_OFFLINE = was_offline


class FaceRecognitionService:
    """Recognize the facing person against the enrolled-faces store."""

    # A match this much stronger than the threshold also feeds the frame's
    # embedding back as an extra reference, so haircuts and glasses never make
    # an enrollment go stale.
    _PROGRESSIVE_MATCH_MARGIN = 0.08

    def __init__(
        self,
        instance_path: str | Path | None,
        *,
        detector: FaceDetector | None = None,
        embedder: FaceEmbedder | None = None,
    ) -> None:
        """Bind the faces-store path; models may be injected for tests."""
        self._instance_path = instance_path
        self._detector = detector
        self._embedder = embedder
        self._models_ready = detector is not None and embedder is not None
        self._load_failed = False
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        """Whether models are loaded (or injected) and recognition can run."""
        return self._models_ready and not self._load_failed

    def load_models(self) -> bool:
        """Load the detector and embedder once; False (sticky) when unavailable."""
        if self.available:
            return True
        with self._load_lock:
            if self.available:
                return True
            if self._load_failed:
                return False
            try:
                detector = (
                    self._detector
                    if self._detector is not None
                    else _run_with_timeout(_construct_detector, _MODEL_LOAD_TIMEOUT_S, "face-detector load")
                )
                embedder = self._embedder if self._embedder is not None else FaceEmbedder(BUNDLED_EMBEDDING_MODEL)
                # Smoke-run the embedder so a broken model fails here, never mid-conversation.
                embedder.embed(np.zeros((_EMBEDDING_INPUT_SIZE, _EMBEDDING_INPUT_SIZE, 3), dtype=np.uint8))
            except Exception as exc:
                logger.warning("Face recognition unavailable: %s", exc)
                self._load_failed = True
                return False
            self._detector = detector
            self._embedder = embedder
            self._models_ready = True
            return True

    @staticmethod
    def _select_face(frame_bgr: NDArray[np.uint8], faces: Sequence[Face]) -> Face | None:
        """Pick the largest face, breaking ties by closeness to the frame center."""
        if not faces:
            return None
        height, width = frame_bgr.shape[:2]

        def rank(face: Face) -> tuple[float, float]:
            x, y, w, h = face.bbox
            center_x = x + w / 2 - width / 2
            center_y = y + h / 2 - height / 2
            return (-w * h, center_x * center_x + center_y * center_y)

        return min(faces, key=rank)

    def _embedding_for_frame(self, frame_bgr: NDArray[np.uint8]) -> NDArray[np.float32] | None:
        if self._detector is None or self._embedder is None:
            return None
        faces = self._detector.detect(frame_bgr)
        face = self._select_face(frame_bgr, faces)
        if face is None:
            return None
        try:
            return self._embedder.embed(align_face(frame_bgr, face))
        except np.linalg.LinAlgError:
            logger.debug("Skipping a face whose landmarks cannot be aligned", exc_info=True)
            return None

    def embeddings_from_frames(self, frames: Sequence[NDArray[np.uint8] | None]) -> list[NDArray[np.float32]]:
        """Embed the dominant face of each frame, skipping frames without one."""
        embeddings: list[NDArray[np.float32]] = []
        for frame in frames:
            if frame is None:
                continue
            embedding = self._embedding_for_frame(frame)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings

    def enroll(self, name: str, frames: Sequence[NDArray[np.uint8] | None]) -> EnrollmentOutcome:
        """Register a person from captured frames."""
        embeddings = self.embeddings_from_frames(frames)
        if len(embeddings) < _MIN_ENROLL_EMBEDDINGS:
            return EnrollmentOutcome(face=None, reason="too_few_faces")
        result = enroll_face(self._instance_path, name, [vector.tolist() for vector in embeddings])
        if result.face is None:
            return EnrollmentOutcome(face=None, reason=result.reason)
        logger.info("Enrolled face %r with %d embeddings", result.face.name, len(embeddings))
        return EnrollmentOutcome(face=result.face, reason=None)

    def recognize(self, frame_bgr: NDArray[np.uint8]) -> RecognitionOutcome:
        """Match the dominant face of one frame against every enrolled reference."""
        if not self.available:
            return RecognitionOutcome(name=None, face_id=None, similarity=0.0, face_detected=False)
        embedding = self._embedding_for_frame(frame_bgr)
        if embedding is None:
            return RecognitionOutcome(name=None, face_id=None, similarity=0.0, face_detected=False)

        best_face: EnrolledFace | None = None
        best_similarity = 0.0
        for face in list_enrolled_faces(self._instance_path):
            for reference in face.embeddings:
                reference_vector = np.asarray(reference, dtype=np.float32)
                reference_norm = np.linalg.norm(reference_vector)
                if reference_norm == 0.0:
                    continue
                similarity = float(np.dot(embedding, reference_vector / reference_norm))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_face = face

        threshold = config.FACE_MATCH_THRESHOLD
        if best_face is None or best_similarity < threshold:
            return RecognitionOutcome(name=None, face_id=None, similarity=best_similarity, face_detected=True)

        progressive_embedding = (
            embedding.tolist() if best_similarity >= threshold + self._PROGRESSIVE_MATCH_MARGIN else None
        )
        mark_face_seen(self._instance_path, best_face.id, progressive_embedding)
        return RecognitionOutcome(
            name=best_face.name, face_id=best_face.id, similarity=best_similarity, face_detected=True
        )
