"""Local speaker identification: kaldi-style fbank + bundled CAM++ ONNX embedding.

The frontend replicates kaldi-native-fbank as configured by sherpa-onnx for the
3D-Speaker CAM++ zh model (80 mel bins, 25/10 ms, povey window, preemphasis 0.97,
snip_edges=false, low 20 Hz, high 7600 Hz, samples in [-1, 1]) plus the model's
own global-mean feature normalization. The model file ships in ``audio/models``.
"""

import logging
import threading
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from my_conversation_app.faces import list_enrolled_faces, append_voice_embeddings
from my_conversation_app.config import config


logger = logging.getLogger(__name__)

BUNDLED_SPEAKER_MODEL = Path(__file__).parent / "models" / "speaker_embedding_campplus_zh.onnx"
SPEAKER_SAMPLE_RATE = 16000

_FRAME_SHIFT = 160
_FRAME_LENGTH = 400
_PADDED_WINDOW = 512
_NUM_MEL_BINS = 80
_LOW_FREQ_HZ = 20.0
_HIGH_FREQ_HZ = 7600.0  # sherpa-onnx convention: nyquist 8000 + high_freq -400

# An utterance shorter than this is not attributed: a syllable or two gives an
# embedding too noisy to bet an identity switch on.
_MIN_RECOGNIZE_SAMPLES = 24000  # 1.5 s
# Enrollment accepts shorter audio than recognition: a bare name (「凯蕾」) is a
# weak but usable seed, and progressive matches grow the reference set later.
_MIN_ENROLL_SAMPLES = 16000  # 1.0 s


def _mel_scale(freq: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return 1127.0 * np.log1p(np.asarray(freq, dtype=np.float64) / 700.0)


def _inverse_mel_scale(mel: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return 700.0 * np.expm1(np.asarray(mel, dtype=np.float64) / 1127.0)


def _build_mel_filters() -> NDArray[np.float64]:
    num_fft_bins = _PADDED_WINDOW // 2
    fft_bin_hz = np.arange(num_fft_bins + 1, dtype=np.float64) * (SPEAKER_SAMPLE_RATE / _PADDED_WINDOW)

    mel_low = _mel_scale(_LOW_FREQ_HZ)
    mel_high = _mel_scale(_HIGH_FREQ_HZ)
    mel_delta = (mel_high - mel_low) / (_NUM_MEL_BINS + 1)

    filters = np.zeros((_NUM_MEL_BINS, num_fft_bins + 1), dtype=np.float64)
    for bin_index in range(_NUM_MEL_BINS):
        left_hz = _inverse_mel_scale(mel_low + bin_index * mel_delta)
        center_hz = _inverse_mel_scale(mel_low + (bin_index + 1) * mel_delta)
        right_hz = _inverse_mel_scale(mel_low + (bin_index + 2) * mel_delta)
        inside = (fft_bin_hz > left_hz) & (fft_bin_hz < right_hz)
        ramp_up = (fft_bin_hz - left_hz) / (center_hz - left_hz)
        ramp_down = (right_hz - fft_bin_hz) / (right_hz - center_hz)
        filters[bin_index, inside] = np.where(fft_bin_hz[inside] <= center_hz, ramp_up[inside], ramp_down[inside])
    return filters


_MEL_FILTERS = _build_mel_filters()
_POVEY_WINDOW = np.power(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(_FRAME_LENGTH) / (_FRAME_LENGTH - 1)), 0.85)


def kaldi_fbank(samples: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return (T, 80) log-mel fbank frames for 16 kHz mono samples in [-1, 1]."""
    num_frames = (samples.size + _FRAME_SHIFT // 2) // _FRAME_SHIFT
    frame_starts = _FRAME_SHIFT * np.arange(num_frames) + _FRAME_SHIFT // 2 - _FRAME_LENGTH // 2
    sample_index = frame_starts[:, None] + np.arange(_FRAME_LENGTH)[None, :]
    # Reflect out-of-range samples around the utterance edges, like kaldi's
    # snip_edges=false framing; repeated reflection only kicks in on tiny inputs.
    while True:
        out_of_range = (sample_index < 0) | (sample_index >= samples.size)
        if not out_of_range.any():
            break
        sample_index = np.where(sample_index < 0, -sample_index - 1, sample_index)
        sample_index = np.where(sample_index >= samples.size, 2 * samples.size - 1 - sample_index, sample_index)

    frames = samples[sample_index].astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1]
    frames[:, 0] -= 0.97 * frames[:, 0]
    frames *= _POVEY_WINDOW

    spectrum = np.fft.rfft(frames, n=_PADDED_WINDOW, axis=1)
    power = np.asarray(spectrum.real**2 + spectrum.imag**2, dtype=np.float64)
    mel_energy = np.asarray(power @ _MEL_FILTERS.T, dtype=np.float64)
    return np.asarray(np.log(np.maximum(mel_energy, np.finfo(np.float32).eps)), dtype=np.float64)


@dataclass(frozen=True)
class SpeakerMatchOutcome:
    """Result of matching one utterance against the enrolled voiceprints."""

    name: str | None
    face_id: str | None
    similarity: float


class SpeakerEmbedder:
    """Bundle-shipped CAM++ speaker-embedding ONNX model."""

    def __init__(self, model_path: str | Path) -> None:
        """Load the embedding ONNX confined to a single CPU thread."""
        options = ort.SessionOptions()
        # One thread, like the wake word and face models: never starve the audio loop.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the L2-normalized embedding of one 16 kHz mono utterance."""
        fbank = kaldi_fbank(samples)
        # The CAM++ model was exported expecting global-mean normalized features.
        normalized = fbank - fbank.mean(axis=0, keepdims=True)
        embedding = np.asarray(
            self._session.run(None, {self._input_name: normalized[np.newaxis].astype(np.float32)})[0][0],
            dtype=np.float32,
        )
        return np.asarray(embedding / np.linalg.norm(embedding), dtype=np.float32)


class SpeakerRecognitionService:
    """Identify the current speaker against the enrolled-faces store's voiceprints."""

    # A match this much stronger than the threshold also feeds the utterance's
    # embedding back as an extra reference, so a scratchy first enrollment
    # tightens as the person keeps talking. Same mechanics as the face service.
    _PROGRESSIVE_MATCH_MARGIN = 0.08

    def __init__(
        self,
        instance_path: str | Path | None,
        *,
        embedder: SpeakerEmbedder | None = None,
    ) -> None:
        """Bind the faces-store path; the embedder may be injected for tests."""
        self._instance_path = instance_path
        self._embedder = embedder
        self._load_failed = False
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        """Whether the embedder is loaded (or injected) and identification can run."""
        return self._embedder is not None and not self._load_failed

    def load_models(self) -> bool:
        """Load the embedder once; False (sticky) when unavailable."""
        if self.available:
            return True
        with self._load_lock:
            if self.available:
                return True
            if self._load_failed:
                return False
            try:
                embedder = SpeakerEmbedder(BUNDLED_SPEAKER_MODEL)
                # Smoke-run so a broken model fails here, never mid-conversation.
                embedder.embed(np.zeros(SPEAKER_SAMPLE_RATE, dtype=np.float32))
            except Exception as exc:
                logger.warning("Speaker identification unavailable: %s", exc)
                self._load_failed = True
                return False
            self._embedder = embedder
            return True

    def embed_for_enrollment(self, samples: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """Return an embedding usable as a reference, or None when too short."""
        embedder = self._embedder
        if embedder is None or self._load_failed:
            return None
        if samples.size < _MIN_ENROLL_SAMPLES:
            logger.info(
                "Voice enrollment skipped: %.1f s of speech is under the %.1f s minimum",
                samples.size / SPEAKER_SAMPLE_RATE,
                _MIN_ENROLL_SAMPLES / SPEAKER_SAMPLE_RATE,
            )
            return None
        return embedder.embed(samples)

    def embed(self, samples: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """Return the utterance's embedding, or None when it cannot be computed."""
        embedder = self._embedder
        if embedder is None or self._load_failed:
            return None
        if samples.size < _MIN_RECOGNIZE_SAMPLES:
            return None
        return embedder.embed(samples)

    def recognize(self, samples: NDArray[np.float32]) -> SpeakerMatchOutcome:
        """Match one utterance against every enrolled person's voice references."""
        embedding = self.embed(samples)
        if embedding is None:
            return SpeakerMatchOutcome(name=None, face_id=None, similarity=0.0)

        best_face_id: str | None = None
        best_name: str | None = None
        best_similarity = 0.0
        for face in list_enrolled_faces(self._instance_path):
            for reference in face.voice_embeddings:
                reference_vector = np.asarray(reference, dtype=np.float32)
                reference_norm = np.linalg.norm(reference_vector)
                if reference_norm == 0.0:
                    continue
                similarity = float(np.dot(embedding, reference_vector / reference_norm))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_name = face.name
                    best_face_id = face.id

        threshold = config.SPEAKER_MATCH_THRESHOLD
        if best_face_id is None or best_similarity < threshold:
            return SpeakerMatchOutcome(name=None, face_id=None, similarity=best_similarity)

        if best_similarity >= threshold + self._PROGRESSIVE_MATCH_MARGIN:
            append_voice_embeddings(self._instance_path, best_face_id, [embedding.tolist()])
        return SpeakerMatchOutcome(name=best_name, face_id=best_face_id, similarity=best_similarity)
