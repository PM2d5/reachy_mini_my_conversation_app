"""Tests for the enrolled-faces store."""

import json
from pathlib import Path

import pytest

from my_conversation_app.faces import (
    MAX_FACES,
    MAX_EMBEDDINGS_PER_FACE,
    MAX_VOICE_EMBEDDINGS_PER_FACE,
    enroll_face,
    mark_face_seen,
    list_enrolled_faces,
    normalize_face_name,
    remove_enrolled_face,
    rename_enrolled_face,
    append_voice_embeddings,
    faces_path_for_instance,
)


def test_faces_store_enrolls_dedupes_and_caps(tmp_path: Path) -> None:
    """Enrollment should normalize names, reject duplicates, and cap the store."""
    first = enroll_face(tmp_path, "  凯蕾  ", [[1.0, 0.0], [0.9, 0.1]])

    assert first.face is not None
    assert first.face.name == "凯蕾"
    assert len(first.face.embeddings) == 2

    duplicate = enroll_face(tmp_path, "凯蕾", [[0.0, 1.0]])
    assert duplicate.face is None
    assert duplicate.reason == "duplicate_name"

    enroll_face(tmp_path, "kailei", [[0.0, 1.0]])
    case_duplicate = enroll_face(tmp_path, "KAILEI", [[0.0, 1.0]])
    assert case_duplicate.reason == "duplicate_name"

    last_result = None
    for index in range(MAX_FACES + 3):
        last_result = enroll_face(tmp_path, f"Person {index}", [[0.1, 0.2]])
    assert last_result is not None
    assert last_result.reason == "store_full"

    faces = list_enrolled_faces(tmp_path)
    assert len(faces) == MAX_FACES
    # The store rejects instead of silently dropping older people: the newest
    # successful enrollment sits in front, the overflow was refused.
    assert faces[0].name == f"Person {MAX_FACES - 3}"
    assert "凯蕾" in [face.name for face in faces]


def test_faces_store_rejects_empty_names_and_embeddings(tmp_path: Path) -> None:
    """Blank names or empty embeddings must be refused."""
    assert enroll_face(tmp_path, "   ", [[1.0]]).reason == "empty_name"
    assert enroll_face(tmp_path, "Amy", []).reason == "no_embeddings"

    assert normalize_face_name("  A  B  ") == "A B"


def test_faces_store_rename_and_remove(tmp_path: Path) -> None:
    """Rename should keep the identity, reject duplicates, and remove should be idempotent."""
    enrolled = enroll_face(tmp_path, "凯蕾", [[1.0, 0.0]])
    assert enrolled.face is not None
    other = enroll_face(tmp_path, "李雷", [[0.0, 1.0]])
    assert other.face is not None

    renamed = rename_enrolled_face(tmp_path, enrolled.face.id, " 蕾蕾 ")
    assert renamed is not None
    assert renamed.name == "蕾蕾"
    assert renamed.created_at == enrolled.face.created_at

    with pytest.raises(ValueError, match="duplicate_name"):
        rename_enrolled_face(tmp_path, enrolled.face.id, "李雷")

    with pytest.raises(ValueError, match="empty_name"):
        rename_enrolled_face(tmp_path, enrolled.face.id, "   ")

    assert remove_enrolled_face(tmp_path, enrolled.face.id) is not None
    assert remove_enrolled_face(tmp_path, enrolled.face.id) is None
    assert [face.name for face in list_enrolled_faces(tmp_path)] == ["李雷"]
    assert rename_enrolled_face(tmp_path, "missing", "新名") is None


def test_faces_store_json_shape_roundtrip(tmp_path: Path) -> None:
    """The persisted envelope keeps camelCase keys and tolerates bad entries."""
    enroll_face(tmp_path, "凯蕾", [[1.0, 0.0], [0.0, 1.0]])
    raw = json.loads(faces_path_for_instance(tmp_path).read_text(encoding="utf-8"))

    assert raw["version"] == 1
    entry = raw["faces"][0]
    assert set(entry) == {"id", "name", "embeddings", "createdAt", "lastSeenAt"}
    assert entry["embeddings"] == [[1.0, 0.0], [0.0, 1.0]]

    faces_path_for_instance(tmp_path).write_text(
        json.dumps(
            {
                "version": 1,
                "faces": [
                    {"id": "f_1", "name": "李雷", "embeddings": [[0.5, 0.5]], "createdAt": 1, "lastSeenAt": 2},
                    {"id": "bad"},
                    {"id": "f_2", "name": "", "embeddings": [[1.0]], "createdAt": 1, "lastSeenAt": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    faces = list_enrolled_faces(tmp_path)
    assert [face.id for face in faces] == ["f_1"]


def test_mark_face_seen_updates_timestamp_and_grows_references(tmp_path: Path) -> None:
    """Seen updates refresh lastSeenAt; extra embeddings append up to the cap."""
    enrolled = enroll_face(tmp_path, "凯蕾", [[1.0, 0.0], [0.9, 0.1]])
    assert enrolled.face is not None
    face_id = enrolled.face.id
    original_last_seen = enrolled.face.last_seen_at

    mark_face_seen(tmp_path, face_id)
    seen_once = list_enrolled_faces(tmp_path)[0]
    assert seen_once.last_seen_at >= original_last_seen
    assert len(seen_once.embeddings) == 2

    mark_face_seen(tmp_path, face_id, [0.5, 0.5])
    seen_twice = list_enrolled_faces(tmp_path)[0]
    assert len(seen_twice.embeddings) == 3

    for _ in range(MAX_EMBEDDINGS_PER_FACE + 2):
        mark_face_seen(tmp_path, face_id, [0.1, 0.2])
    final = list_enrolled_faces(tmp_path)[0]
    assert len(final.embeddings) == MAX_EMBEDDINGS_PER_FACE
    assert final.embeddings[-1] == (0.1, 0.2)

    mark_face_seen(tmp_path, "missing", [1.0])
    assert len(list_enrolled_faces(tmp_path)) == 1


def test_append_voice_embeddings_roundtrip_and_backward_compat(tmp_path: Path) -> None:
    """Voice references persist under voiceEmbeddings; old stores read back empty."""
    enrolled = enroll_face(tmp_path, "凯蕾", [[1.0, 0.0]])
    assert enrolled.face is not None

    assert append_voice_embeddings(tmp_path, enrolled.face.id, [[0.5, 0.5], []]) is not None
    raw = json.loads(faces_path_for_instance(tmp_path).read_text(encoding="utf-8"))
    assert raw["faces"][0]["voiceEmbeddings"] == [[0.5, 0.5]]

    assert list_enrolled_faces(tmp_path)[0].voice_embeddings == ((0.5, 0.5),)

    # A pre-voice store (no voiceEmbeddings key) still loads, with no references.
    faces_path_for_instance(tmp_path).write_text(
        json.dumps(
            {
                "version": 1,
                "faces": [
                    {"id": "f_old", "name": "旧数据", "embeddings": [[0.9, 0.1]], "createdAt": 1, "lastSeenAt": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert list_enrolled_faces(tmp_path)[0].voice_embeddings == ()


def test_append_voice_embeddings_caps_and_ignores_unknown_face(tmp_path: Path) -> None:
    """Newest voice references win at the cap; unknown face ids change nothing."""
    enrolled = enroll_face(tmp_path, "凯蕾", [[1.0, 0.0]])
    assert enrolled.face is not None
    face_id = enrolled.face.id

    for index in range(MAX_VOICE_EMBEDDINGS_PER_FACE + 2):
        assert append_voice_embeddings(tmp_path, face_id, [[float(index), 1.0]]) is not None

    voices = list_enrolled_faces(tmp_path)[0].voice_embeddings
    assert len(voices) == MAX_VOICE_EMBEDDINGS_PER_FACE
    assert voices[0] == (2.0, 1.0)  # the two oldest references fell off the cap
    assert voices[-1] == (float(MAX_VOICE_EMBEDDINGS_PER_FACE + 1), 1.0)

    assert append_voice_embeddings(tmp_path, "missing", [[0.1, 0.2]]) is None
    assert append_voice_embeddings(tmp_path, face_id, [[]]) is None


def test_rename_and_mark_face_seen_keep_voice_embeddings(tmp_path: Path) -> None:
    """Face-only updates must not drop the attached voice references."""
    enrolled = enroll_face(tmp_path, "凯蕾", [[1.0, 0.0]])
    assert enrolled.face is not None
    face_id = enrolled.face.id
    assert append_voice_embeddings(tmp_path, face_id, [[0.5, 0.5]]) is not None

    renamed = rename_enrolled_face(tmp_path, face_id, "凯磊")
    assert renamed is not None
    assert renamed.voice_embeddings == ((0.5, 0.5),)

    mark_face_seen(tmp_path, face_id, [0.9, 0.1])
    updated = list_enrolled_faces(tmp_path)[0]
    assert updated.name == "凯磊"
    assert updated.voice_embeddings == ((0.5, 0.5),)
