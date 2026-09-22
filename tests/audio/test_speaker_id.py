"""Tests for the local speaker-identification frontend and service."""

from pathlib import Path

import numpy as np
import pytest

from my_conversation_app.faces import enroll_face, list_enrolled_faces, append_voice_embeddings
from my_conversation_app.config import config
from my_conversation_app.audio.speaker_id import (
    BUNDLED_SPEAKER_MODEL,
    SpeakerEmbedder,
    SpeakerRecognitionService,
    kaldi_fbank,
)


def _speech_like(seconds: float = 3.0) -> np.ndarray:
    """Deterministic amplitude-modulated buzz standing in for an utterance."""
    t = np.arange(int(16000 * seconds)) / 16000
    buzz = np.sin(2 * np.pi * 110.0 * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (buzz * envelope * 0.5).astype(np.float32)


class _FakeEmbedder:
    """Stand-in for the ONNX embedder: positive-leading input maps to unit vector 0."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def embed(self, samples: np.ndarray) -> np.ndarray:
        self.calls.append(samples)
        direction = np.zeros(4, dtype=np.float32)
        direction[int(samples[0] <= 0)] = 1.0
        return direction


def _service_with_voice(tmp_path: Path) -> tuple[SpeakerRecognitionService, _FakeEmbedder]:
    embedder = _FakeEmbedder()
    service = SpeakerRecognitionService(tmp_path, embedder=embedder)
    result = enroll_face(tmp_path, "凯蕾", [[1.0, 0.0, 0.0, 0.0]])
    assert result.face is not None
    assert append_voice_embeddings(tmp_path, result.face.id, [[1.0, 0.0, 0.0, 0.0]]) is not None
    return service, embedder


def _positive_utterance(seconds: float = 1.5) -> np.ndarray:
    samples = np.zeros(int(16000 * seconds), dtype=np.float32)
    samples[0] = 1.0
    return samples


def _negative_utterance(seconds: float = 1.5) -> np.ndarray:
    samples = np.zeros(int(16000 * seconds), dtype=np.float32)
    samples[0] = -1.0
    return samples


def test_kaldi_fbank_frame_count_and_finiteness() -> None:
    """snip_edges=false framing: round(num_samples / shift) frames of 80 mel bins."""
    fbank = kaldi_fbank(np.zeros(16000, dtype=np.float32))  # exactly 1 s -> 100 frames
    assert fbank.shape == (100, 80)
    assert np.isfinite(fbank).all()
    # silence floors at log(eps) in every bin, never -inf
    assert fbank.min() > -20.0


def test_bundled_embedder_is_deterministic_and_unit_norm() -> None:
    """The bundled ONNX loads offline, embeds 192 dims, unit norm, repeatably."""
    embedder = SpeakerEmbedder(BUNDLED_SPEAKER_MODEL)
    first = embedder.embed(_speech_like())
    second = embedder.embed(_speech_like())
    assert first.shape == (192,)
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert np.array_equal(first, second)


def test_recognize_matches_enrolled_voice_above_threshold(tmp_path: Path) -> None:
    """A match over the threshold names the person; under it stays unknown."""
    service, _ = _service_with_voice(tmp_path)

    matched = service.recognize(_positive_utterance())
    assert matched.name == "凯蕾"
    assert matched.face_id is not None
    assert matched.similarity == pytest.approx(1.0)

    assert service.recognize(_negative_utterance()).name is None


def test_recognize_skips_short_utterances(tmp_path: Path) -> None:
    """Under 1.5 s the utterance is not attributed at all."""
    service, embedder = _service_with_voice(tmp_path)

    assert service.recognize(_positive_utterance(1.0)).face_id is None
    assert embedder.calls == []


def test_enrollment_minimum_is_looser_than_recognition(tmp_path: Path) -> None:
    """A 1 s utterance is too short to attribute but long enough to enroll."""
    service, _ = _service_with_voice(tmp_path)

    assert service.embed_for_enrollment(_positive_utterance(1.0)) is not None
    assert service.embed_for_enrollment(_positive_utterance(0.5)) is None


def test_strong_match_progressively_grows_the_reference_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A high-margin match feeds the utterance back as an extra reference."""
    monkeypatch.setattr(config, "SPEAKER_MATCH_THRESHOLD", 0.5)
    service, _ = _service_with_voice(tmp_path)

    assert service.recognize(_positive_utterance()).name == "凯蕾"
    assert len(list_enrolled_faces(tmp_path)[0].voice_embeddings) == 2


def test_no_voice_references_means_no_match(tmp_path: Path) -> None:
    """Face-only enrollment never trips speaker identification."""
    assert enroll_face(tmp_path, "face-only", [[1.0, 0.0, 0.0, 0.0]]).face is not None
    service = SpeakerRecognitionService(tmp_path, embedder=_FakeEmbedder())

    outcome = service.recognize(_positive_utterance())
    assert outcome.face_id is None
    assert outcome.similarity == 0.0


def test_load_models_failure_is_sticky(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed model load is remembered; later calls do not retry."""

    class _ExplodingEmbedder:
        def __init__(self, _model_path: object) -> None:
            raise RuntimeError("no model")

    monkeypatch.setattr("my_conversation_app.audio.speaker_id.SpeakerEmbedder", _ExplodingEmbedder)
    service = SpeakerRecognitionService(tmp_path)

    assert service.load_models() is False
    monkeypatch.setattr("my_conversation_app.audio.speaker_id.SpeakerEmbedder", None)
    assert service.load_models() is False
