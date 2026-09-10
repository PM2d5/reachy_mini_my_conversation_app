import os
import json
import time
import random
import string
import logging
import threading
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Sequence


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_FACES = 10
MAX_EMBEDDINGS_PER_FACE = 5
MAX_NAME_CHARS = 24
FACES_FILENAME = "faces.v1.json"

_STORE_LOCK = threading.Lock()


@dataclass(frozen=True)
class EnrolledFace:
    """One enrolled person: a name plus face-embedding references."""

    id: str
    name: str
    embeddings: tuple[tuple[float, ...], ...]
    created_at: int
    last_seen_at: int

    def to_json(self) -> dict[str, object]:
        """Return the persisted JSON shape."""
        return {
            "id": self.id,
            "name": self.name,
            "embeddings": [[round(value, 4) for value in embedding] for embedding in self.embeddings],
            "createdAt": self.created_at,
            "lastSeenAt": self.last_seen_at,
        }


@dataclass(frozen=True)
class EnrollFaceResult:
    """Outcome of an enrollment attempt."""

    face: EnrolledFace | None
    reason: str | None  # None on success; "duplicate_name" | "store_full" | "empty_name" | "no_embeddings"


def faces_path_for_instance(instance_path: str | Path | None = None) -> Path:
    """Return the enrolled-faces JSON path for this app instance."""
    if instance_path is not None:
        return Path(instance_path).expanduser() / FACES_FILENAME

    data_home = os.getenv("XDG_DATA_HOME")
    data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return data_root / "my_conversation_app" / FACES_FILENAME


def normalize_face_name(name: str) -> str:
    """Collapse whitespace and enforce the name length cap."""
    return " ".join(name.split()).strip()[:MAX_NAME_CHARS]


def _make_id() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"f_{int(time.time() * 1000)}_{suffix}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _face_from_json(value: object) -> EnrolledFace | None:
    if not isinstance(value, Mapping):
        return None

    face_id = value.get("id")
    name = value.get("name")
    embeddings_value = value.get("embeddings")
    created_at = value.get("createdAt")
    last_seen_at = value.get("lastSeenAt")

    if not isinstance(face_id, str) or not isinstance(name, str):
        return None
    if not isinstance(embeddings_value, list) or not embeddings_value:
        return None
    if not isinstance(created_at, (int, float)) or not isinstance(last_seen_at, (int, float)):
        return None

    embeddings: list[tuple[float, ...]] = []
    for embedding_value in embeddings_value:
        if not isinstance(embedding_value, Sequence) or isinstance(embedding_value, (str, bytes)):
            return None
        try:
            embeddings.append(tuple(float(component) for component in embedding_value))
        except (TypeError, ValueError):
            return None
    if not all(embedding for embedding in embeddings):
        return None

    normalized = normalize_face_name(name)
    if not normalized:
        return None

    return EnrolledFace(
        id=face_id,
        name=normalized,
        embeddings=tuple(embeddings[:MAX_EMBEDDINGS_PER_FACE]),
        created_at=int(created_at),
        last_seen_at=int(last_seen_at),
    )


def _read_faces_file(path: Path) -> list[EnrolledFace]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Failed to read faces store at %s: %s", path, exc)
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse faces store at %s: %s", path, exc)
        return []

    if not isinstance(parsed, Mapping):
        return []

    faces_value = parsed.get("faces")
    if not isinstance(faces_value, list):
        return []

    faces: list[EnrolledFace] = []
    for item in faces_value:
        face = _face_from_json(item)
        if face is not None:
            faces.append(face)
    return faces[:MAX_FACES]


def _write_faces_file(path: Path, faces: list[EnrolledFace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCHEMA_VERSION,
        "faces": [face.to_json() for face in faces[:MAX_FACES]],
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def list_enrolled_faces(instance_path: str | Path | None = None) -> list[EnrolledFace]:
    """Return enrolled faces, newest first."""
    with _STORE_LOCK:
        return list(_read_faces_file(faces_path_for_instance(instance_path)))


def enroll_face(
    instance_path: str | Path | None,
    name: str,
    embeddings: Sequence[Sequence[float]],
) -> EnrollFaceResult:
    """Store one person with fresh embedding references."""
    normalized = normalize_face_name(name)
    if not normalized or not embeddings:
        return EnrollFaceResult(face=None, reason="empty_name" if not normalized else "no_embeddings")

    path = faces_path_for_instance(instance_path)
    with _STORE_LOCK:
        faces = _read_faces_file(path)
        if any(face.name.lower() == normalized.lower() for face in faces):
            return EnrollFaceResult(face=None, reason="duplicate_name")
        if len(faces) >= MAX_FACES:
            return EnrollFaceResult(face=None, reason="store_full")

        now = _now_ms()
        face = EnrolledFace(
            id=_make_id(),
            name=normalized,
            embeddings=tuple(tuple(float(value) for value in embedding) for embedding in embeddings),
            created_at=now,
            last_seen_at=now,
        )
        _write_faces_file(path, [face, *faces][:MAX_FACES])
        return EnrollFaceResult(face=face, reason=None)


def rename_enrolled_face(instance_path: str | Path | None, face_id: str, name: str) -> EnrolledFace | None:
    """Rename one enrolled face; raises ValueError on a duplicate name."""
    normalized = normalize_face_name(name)
    if not normalized:
        raise ValueError("empty_name")

    path = faces_path_for_instance(instance_path)
    with _STORE_LOCK:
        faces = _read_faces_file(path)
        if any(face.name.lower() == normalized.lower() and face.id != face_id for face in faces):
            raise ValueError("duplicate_name")
        renamed = [face for face in faces if face.id == face_id]
        if not renamed:
            return None
        updated = EnrolledFace(
            id=renamed[0].id,
            name=normalized,
            embeddings=renamed[0].embeddings,
            created_at=renamed[0].created_at,
            last_seen_at=renamed[0].last_seen_at,
        )
        _write_faces_file(path, [updated if face.id == face_id else face for face in faces])
        return updated


def remove_enrolled_face(instance_path: str | Path | None, face_id: str) -> EnrolledFace | None:
    """Remove one enrolled face."""
    path = faces_path_for_instance(instance_path)
    with _STORE_LOCK:
        faces = _read_faces_file(path)
        removed = [face for face in faces if face.id == face_id]
        if not removed:
            return None
        _write_faces_file(path, [face for face in faces if face.id != face_id])
        return removed[0]


def mark_face_seen(
    instance_path: str | Path | None,
    face_id: str,
    extra_embedding: Sequence[float] | None = None,
) -> None:
    """Refresh lastSeenAt and optionally append a high-confidence reference embedding."""
    path = faces_path_for_instance(instance_path)
    with _STORE_LOCK:
        faces = _read_faces_file(path)
        if not any(face.id == face_id for face in faces):
            return

        updated_faces: list[EnrolledFace] = []
        for face in faces:
            if face.id != face_id:
                updated_faces.append(face)
                continue
            embeddings = face.embeddings
            if extra_embedding is not None and len(extra_embedding) > 0:
                # Newest reference wins when the per-person cap is hit, so looks drift in.
                embeddings = (*embeddings, tuple(float(value) for value in extra_embedding))[-MAX_EMBEDDINGS_PER_FACE:]
            updated_faces.append(
                EnrolledFace(
                    id=face.id,
                    name=face.name,
                    embeddings=embeddings,
                    created_at=face.created_at,
                    last_seen_at=_now_ms(),
                )
            )
        _write_faces_file(path, updated_faces)
