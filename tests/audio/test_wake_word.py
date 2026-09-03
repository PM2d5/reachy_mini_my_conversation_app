"""Tests for the wake word detector and its configuration helpers."""

from typing import Any

import numpy as np
import pytest

from my_conversation_app import config
from my_conversation_app.audio import wake_word
from my_conversation_app.audio.wake_word import WakeWordDetector, _resample_to_16k


class _FakeModel:
    """Stand-in for the openWakeWord model with scripted scores."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.reset_called = False
        self.last_samples: np.ndarray | None = None

    def predict(self, samples: Any) -> dict[str, float]:
        self.last_samples = samples
        return dict(self._scores)

    def reset(self) -> None:
        self.reset_called = True


def _detector_with_scores(scores: dict[str, float]) -> WakeWordDetector:
    detector = WakeWordDetector(models=("hey_mycroft",), threshold=0.5)
    detector._model = _FakeModel(scores)
    return detector


def test_predict_returns_hit_and_resets_the_model() -> None:
    """A score at or above the threshold is reported once and the model state resets."""
    detector = _detector_with_scores({"hey_mycroft": 0.7})

    assert detector.predict(16000, np.zeros(1280, dtype=np.int16)) == "hey_mycroft"
    assert detector._model is not None and detector._model.reset_called


def test_predict_below_threshold_returns_none() -> None:
    """Scores under the threshold never report a wake word."""
    detector = _detector_with_scores({"hey_mycroft": 0.3})

    assert detector.predict(16000, np.zeros(1280, dtype=np.int16)) is None


def test_predict_converts_float32_frames_to_int16() -> None:
    """Recorder frames arriving as float32 are converted before reaching the model."""
    detector = _detector_with_scores({"hey_mycroft": 0.0})

    assert detector.predict(16000, np.zeros(1280, dtype=np.float32)) is None
    model = detector._model
    assert model is not None and model.last_samples is not None and model.last_samples.dtype == np.int16


def test_dump_records_exactly_what_the_detector_hears(tmp_path: Any) -> None:
    """With a dump path, standby mic audio is written as 16 kHz mono PCM."""
    import wave

    dump_path = tmp_path / "standby.wav"
    detector = WakeWordDetector(models=("hey_mycroft",), dump_path=dump_path)

    detector.predict(48000, np.zeros((1024, 2), dtype=np.int16))
    detector.close_dump()

    with wave.open(str(dump_path)) as dump:
        assert dump.getframerate() == 16000
        assert dump.getnchannels() == 1
        assert dump.getnframes() == 341  # 1024 samples at 48 kHz -> 16 kHz mono


def test_load_failure_degrades_to_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that cannot load disables detection instead of raising."""

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("no model")

    monkeypatch.setattr(wake_word, "OpenWakeWordModel", _boom)

    detector = WakeWordDetector(models=("nope",))

    assert detector.available is False
    assert detector.predict(16000, np.zeros(8, dtype=np.int16)) is None


def test_predict_resamples_stereo_frames_from_other_rates() -> None:
    """Stereo frames at a non-16 kHz rate still feed the detector without raising."""
    detector = _detector_with_scores({"hey_mycroft": 0.0})
    stereo_frame = np.zeros((512, 2), dtype=np.int16)

    assert detector.predict(48000, stereo_frame) is None


def test_resample_to_16k_scales_length() -> None:
    """Linear resampling converts by the sample-rate ratio."""
    samples = np.arange(16000, dtype=np.int16)

    resampled = _resample_to_16k(samples, 48000)

    assert resampled.size == round(16000 * 16000 / 48000)
    assert _resample_to_16k(samples, 16000) is samples


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("0.8", 0.8),
        ("", config.DEFAULT_WAKE_WORD_THRESHOLD),
        ("bogus", config.DEFAULT_WAKE_WORD_THRESHOLD),
        ("2.5", 1.0),  # clamped to the valid range
        ("-3", 0.0),
    ],
)
def test_resolve_wake_word_threshold(monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: float) -> None:
    """The threshold parses, falls back to the default, or clamps into 0..1."""
    monkeypatch.setenv(config.WAKE_WORD_THRESHOLD_ENV, raw_value)

    assert config.resolve_wake_word_threshold() == expected


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("120", 120.0),
        ("", config.DEFAULT_WAKE_WORD_ACTIVE_TIMEOUT_S),
        ("bogus", config.DEFAULT_WAKE_WORD_ACTIVE_TIMEOUT_S),
        ("0", 0.0),  # disabled
        ("-5", 0.0),
    ],
)
def test_resolve_wake_word_active_timeout_s(monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: float) -> None:
    """The active-listening timeout parses, falls back, or disables on non-positive."""
    monkeypatch.setenv(config.WAKE_WORD_ACTIVE_TIMEOUT_S_ENV, raw_value)

    assert config.resolve_wake_word_active_timeout_s() == expected


def test_default_wake_word_model_is_bundled() -> None:
    """The default wake word points at the bundled, shipped model file."""
    from pathlib import Path

    for entry in config.DEFAULT_WAKE_WORD_MODELS:
        assert Path(entry).is_file(), f"bundled wake word model missing: {entry}"


def test_normalize_wake_word_models_and_goodbye_keywords() -> None:
    """Model lists and goodbye keywords split on commas, trimming blanks."""
    assert config._normalize_wake_word_models(" a.onnx, hey_mycroft , ") == ("a.onnx", "hey_mycroft")
    assert config._normalize_wake_word_models("") == config.DEFAULT_WAKE_WORD_MODELS
    assert config._normalize_goodbye_keywords(" 再见 , GoodNight ") == ("再见", "goodnight")
    assert config._normalize_goodbye_keywords(None) == config.DEFAULT_GOODBYE_KEYWORDS
