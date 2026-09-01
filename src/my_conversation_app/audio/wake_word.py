import time
import wave
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openwakeword.model import Model as OpenWakeWordModel
from openwakeword.utils import download_models

from my_conversation_app.config import config
from my_conversation_app.streaming import AudioArray, audio_to_int16


logger = logging.getLogger(__name__)

WAKE_WORD_SAMPLE_RATE = 16000

# How often to log a standby heartbeat with the best score and mic level, so a
# silent mic or a too-high threshold is visible in the logs.
SCORE_LOG_INTERVAL_S = 15.0


def _to_mono_int16(frame: AudioArray) -> NDArray[np.int16]:
    """Collapse a recorder frame to mono int16; recorder frames may arrive as float32."""
    if frame.ndim == 2:
        if frame.shape[1] > frame.shape[0]:
            frame = frame.T
        if frame.shape[1] > 1:
            frame = frame[:, 0]
    return audio_to_int16(frame.reshape(-1))


def _resample_to_16k(samples: NDArray[np.int16], sample_rate: int) -> NDArray[np.int16]:
    """Linearly resample mono int16 audio to the 16 kHz the wake word models expect."""
    if sample_rate == WAKE_WORD_SAMPLE_RATE or samples.size == 0:
        return samples
    target_length = int(round(samples.size * WAKE_WORD_SAMPLE_RATE / sample_rate))
    source_positions = np.linspace(0.0, samples.size - 1.0, target_length)
    return np.interp(source_positions, np.arange(samples.size), samples.astype(np.float64)).astype(np.int16)


class WakeWordDetector:
    """Offline wake word detection over openWakeWord models."""

    def __init__(
        self,
        models: tuple[str, ...] | None = None,
        threshold: float | None = None,
        dump_path: Path | None = None,
    ) -> None:
        """Load the configured models; sets ``available`` False instead of raising."""
        self._model_names = models if models is not None else config.WAKE_WORD_MODELS
        self._threshold = config.WAKE_WORD_THRESHOLD if threshold is None else threshold
        self._model: OpenWakeWordModel | None = None
        self.available = False
        self._last_score_log_time = 0.0
        self._window_best_score = 0.0
        self._window_best_name = ""
        self._window_peak_rms = 0.0
        self._dump_file = self._open_dump(dump_path)
        self._load()

    def _load(self) -> None:
        try:
            self._model = self._build_model()
        except Exception as e:
            logger.warning(
                "Wake word models %s failed to load, wake word detection disabled: %s", self._model_names, e
            )
            self._model = None
            return
        self.available = True
        logger.info("Wake word detector ready: models=%s threshold=%.2f", self._model_names, self._threshold)

    def _build_model(self) -> OpenWakeWordModel:
        try:
            return OpenWakeWordModel(wakeword_models=list(self._model_names), inference_framework="onnx")
        except Exception:
            # First run on a fresh install: pretrained model files are not shipped in the wheel.
            download_models()
            return OpenWakeWordModel(wakeword_models=list(self._model_names), inference_framework="onnx")

    def _open_dump(self, dump_path: Path | None) -> wave.Wave_write | None:
        """Open a 16 kHz mono PCM wav capturing exactly what the detector hears."""
        if dump_path is None:
            return None
        try:
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_file = wave.open(str(dump_path), "wb")
            dump_file.setnchannels(1)
            dump_file.setsampwidth(2)
            dump_file.setframerate(WAKE_WORD_SAMPLE_RATE)
            logger.info("Dumping standby mic audio to %s", dump_path)
            return dump_file
        except Exception as e:
            logger.warning("Cannot open wake word dump file %s: %s", dump_path, e)
            return None

    def close_dump(self) -> None:
        """Flush and close the debug dump file, if one is open."""
        if self._dump_file is not None:
            self._dump_file.close()
            self._dump_file = None

    def predict(self, sample_rate: int, frame: AudioArray) -> str | None:
        """Feed one mic frame and return the detected wake word name, if any."""
        model = self._model
        if model is None or frame.size == 0:
            return None
        samples = _resample_to_16k(_to_mono_int16(frame), sample_rate)
        if samples.size == 0:
            return None
        if self._dump_file is not None:
            self._dump_file.writeframes(samples.tobytes())
        try:
            scores = model.predict(samples)
        except Exception as e:
            logger.warning("Wake word prediction failed: %s", e)
            return None
        now = time.monotonic()
        best_name, best_score = max(scores.items(), key=lambda item: float(item[1]))
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
        self._window_best_score = max(self._window_best_score, float(best_score))
        self._window_peak_rms = max(self._window_peak_rms, rms)
        if not self._window_best_name or float(best_score) >= self._window_best_score:
            self._window_best_name = best_name
        if now - self._last_score_log_time >= SCORE_LOG_INTERVAL_S:
            logger.info(
                "Wake word standby: best=%s peak score=%.2f threshold=%.2f peak mic RMS=%.0f",
                self._window_best_name,
                self._window_best_score,
                self._threshold,
                self._window_peak_rms,
            )
            self._last_score_log_time = now
            self._window_best_score = 0.0
            self._window_best_name = ""
            self._window_peak_rms = 0.0
        for name, score in scores.items():
            if float(score) >= self._threshold:
                model.reset()
                return str(name)
        return None
