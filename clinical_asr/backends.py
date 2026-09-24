"""Streaming ASR backends.

Two local backends, both free of the old rolling-window re-decode that caused
missing and repeated words:

- :class:`NemoCacheAwareStreamingASR` — true streaming. Each audio frame is
  processed exactly once; the encoder reuses cached activations
  (``conformer_stream_step``). No overlap, no text-merge heuristics, so words
  can neither be dropped at a window edge nor committed twice. Runs locally on
  GPU (or CPU) via NeMo. Recommended models (all local after first download):

  - ``nvidia/nemotron-speech-streaming-en-0.6b`` (default, 600M, PnC)
  - ``nvidia/parakeet_realtime_eou_120m-v1`` (120M, EOU turn-taking, no PnC)
  - ``nvidia/nemotron-3.5-asr-streaming-0.6b`` (multilingual streaming)

- :class:`VadSegmentedOfflineASR` — fallback for offline-only ``.nemo`` files
  such as ``parakeet-tdt-0.6b-v3.nemo``. Audio is cut at pauses and every
  segment is decoded exactly once with ``transcribe()``. No sliding window,
  so again no repeats/drops by construction. No partials mid-segment.

Protocol (unchanged, see ``clinical_asr/main.py``):

- ``push_audio`` returns the new grey tail (delta) or ``None``.
- ``pending_commit`` / ``consume_pending_commit`` expose the white text.
- ``finalize`` returns the full cumulative transcript.
"""

import importlib
import logging
import re
import struct
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _load_nemo_model(model_name: str, model_path: str, device: str):
    """Load a NeMo ASR model once; caller decides streaming vs offline use."""
    nemo = importlib.import_module("nemo.collections.asr.models")
    asr_model = nemo.ASRModel
    if model_path:
        model = asr_model.restore_from(restore_path=Path(model_path), map_location="cpu")
    else:
        model = asr_model.from_pretrained(model_name=model_name)
    if device == "cuda" or (device == "auto" and _cuda_available()):
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()
    return model


def _repo_slug(model_name: str) -> str:
    """Last path component of a repo id: ``nvidia/foo`` -> ``foo``."""
    return model_name.strip().strip("/").split("/")[-1] or "model"


def _largest_nemo(directory: Path) -> Path | None:
    candidates = [path for path in directory.glob("*.nemo") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def _download_checkpoint(repo_id: str, local_dir: Path) -> Path:
    """Fetch the ``*.nemo`` checkpoint of a repo into ``local_dir``."""
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download model weights "
            "(pip install huggingface_hub), or place the .nemo file under "
            f"{local_dir} manually."
        ) from exc
    logger.info(f"Downloading {repo_id} weights to {local_dir} ...")
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        hub.snapshot_download(
            repo_id=repo_id, local_dir=str(local_dir), allow_patterns=["*.nemo"]
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to download weights for '{repo_id}': {exc}. "
            "Pre-place the .nemo file manually to run fully offline."
        ) from exc
    checkpoint = _largest_nemo(local_dir)
    if checkpoint is None:
        raise RuntimeError(
            f"Repo '{repo_id}' contains no .nemo checkpoint in {local_dir}."
        )
    logger.info(f"Weights ready: {checkpoint}")
    return checkpoint


def ensure_local_checkpoint(
    model_name: str, model_path: str, models_dir: str | Path
) -> Path:
    """Resolve a local ``.nemo`` file, downloading into ``models_dir`` if needed.

    Resolution order:

    1. Explicit ``model_path`` that exists -> use it.
    2. ``model_name`` that is itself an existing file -> use it.
    3. ``<slug>.nemo`` anywhere under ``models_dir`` (covers hand-placed
       layouts like ``models/parakeet/parakeet-tdt-0.6b-v3.nemo``).
    4. ``*.nemo`` already inside ``models_dir/<slug>/`` -> use it.
    5. Otherwise download the repo's ``*.nemo`` into ``models_dir/<slug>/``.
       An explicit-but-missing ``model_path`` under the models tree is used as
       the download target directory.
    """
    models_root = Path(models_dir)
    if model_path:
        explicit = Path(model_path)
        if explicit.is_file():
            return explicit
        if model_name and "/" in model_name:
            logger.warning(f"Configured weight not found: {explicit}; downloading.")
            return _download_checkpoint(model_name, explicit.parent)
        raise FileNotFoundError(
            f"Configured weight not found: {explicit}. Set ASR_MODEL_PATH to an "
            "existing .nemo file or leave it empty to auto-download."
        )
    if model_name:
        literal = Path(model_name)
        if literal.is_file():
            return literal
        if "/" in model_name:
            slug = _repo_slug(model_name)
            recursive = sorted(models_root.rglob(f"{slug}.nemo"))
            existing_recursive = [path for path in recursive if path.is_file()]
            if existing_recursive:
                return existing_recursive[0]
            local_dir = models_root / slug
            checkpoint = _largest_nemo(local_dir) if local_dir.is_dir() else None
            if checkpoint is not None:
                return checkpoint
            return _download_checkpoint(model_name, local_dir)
    raise ValueError(
        "No model configured: set ASR_MODEL_NAME to a repo id "
        "(e.g. nvidia/nemotron-speech-streaming-en-0.6b) or ASR_MODEL_PATH "
        "to a local .nemo file."
    )


def _pcm16_to_float32(audio_bytes: bytes) -> np.ndarray:
    sample_count = len(audio_bytes) // SAMPLE_WIDTH
    if sample_count == 0:
        return np.zeros(0, dtype=np.float32)
    samples = struct.unpack(f"<{sample_count}h", audio_bytes[: sample_count * SAMPLE_WIDTH])
    return np.asarray(samples, dtype=np.float32) / 32768.0


def _rms(audio_chunk: bytes) -> float:
    sample_count = len(audio_chunk) // SAMPLE_WIDTH
    if sample_count == 0:
        return 0.0
    samples = struct.unpack(
        f"<{sample_count}h", audio_chunk[: sample_count * SAMPLE_WIDTH]
    )
    mean_square = sum(sample * sample for sample in samples) / sample_count
    return mean_square**0.5


def _extract_text(hyps) -> str:
    """Hypotheses may be RNNT Hypothesis objects (.text) or plain strings."""
    parts = []
    for hyp in hyps or []:
        text = getattr(hyp, "text", hyp)
        text = str(text).strip() if text is not None else ""
        if text:
            parts.append(text)
    return " ".join(parts).strip()


_EOU_TOKEN_RE = re.compile(r"</?(eou|eob)\s*/?>", re.IGNORECASE)


def _strip_eou_markers(text: str) -> tuple[str, bool]:
    """Remove inline end-of-utterance markers; report if any were present."""
    cleaned, count = _EOU_TOKEN_RE.subn("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, count > 0


def _strip_committed_prefix(committed: str, hypothesis: str, max_words: int = 20) -> str:
    """Display-only tail: drop the already-white prefix from a hypothesis.

    Exact string prefix first; if the streaming decoder revised recent words,
    fall back to a bounded word-overlap so the grey tail never repeats white
    text. Never touches the cumulative transcript.
    """
    committed = committed.strip()
    hypothesis = hypothesis.strip()
    if not committed:
        return hypothesis
    if not hypothesis:
        return ""
    if hypothesis.startswith(committed):
        return hypothesis[len(committed):].strip()
    committed_words = committed.split()
    hypothesis_words = hypothesis.split()
    normalize = lambda word: word.strip(".,!?;:").lower()
    committed_norm = [normalize(word) for word in committed_words]
    hypothesis_norm = [normalize(word) for word in hypothesis_words]
    limit = min(max_words, len(committed_norm), len(hypothesis_norm))
    for size in range(limit, 0, -1):
        if committed_norm[-size:] == hypothesis_norm[:size]:
            return " ".join(hypothesis_words[size:])
    return hypothesis


class StreamingASR(ABC):
    @abstractmethod
    async def start_session(self) -> None: ...
    @abstractmethod
    async def push_audio(self, audio_chunk: bytes) -> str | None: ...
    @abstractmethod
    async def push_demo_text(self, text: str) -> str | None: ...
    @abstractmethod
    async def finalize(self) -> str: ...
    @abstractmethod
    async def reset(self) -> None: ...


class MockStreamingASR(StreamingASR):
    def __init__(self) -> None:
        self.text = ""
        self.audio_bytes = 0

    async def start_session(self) -> None:
        self.text = ""
        self.audio_bytes = 0

    async def push_audio(self, audio_chunk: bytes) -> str | None:
        self.audio_bytes += len(audio_chunk)
        return self.text or None

    async def push_demo_text(self, text: str) -> str | None:
        self.text = text.strip()
        return self.text or None

    async def finalize(self) -> str:
        return self.text

    async def reset(self) -> None:
        self.text = ""
        self.audio_bytes = 0


class NemoCacheAwareStreamingASR(StreamingASR):
    """True cache-aware streaming over a locally loaded NeMo streaming model.

    Feed-forward only: PCM16 -> per-session ``CacheAwareStreamingAudioBuffer``
    -> ``conformer_stream_step`` with persistent encoder caches and RNNT
    hypotheses. The decoder hypothesis is already cumulative, so committing is
    just promotion (silence pause or EOU token) — nothing is re-decoded.
    """

    _model = None
    _model_key: tuple | None = None
    _model_lock = threading.Lock()
    _step_lock = threading.Lock()

    # att_context_size right-context frames -> chunk latency (80 ms frames).
    RIGHT_CONTEXT_LATENCY_MS = {0: 80, 1: 160, 6: 560, 13: 1120}

    SILENCE_RMS_THRESHOLD = 500.0
    SILENCE_COMMIT_SECONDS = 0.6

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        model_path: str = "",
        device: str = "auto",
        right_context: int = 6,
        models_dir: str | Path = "models",
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.right_context = right_context
        self.models_dir = Path(models_dir)
        self._buffer = None
        self._reset_session_state()

    def _reset_session_state(self) -> None:
        self._buffer = None
        self._cache_last_channel = None
        self._cache_last_time = None
        self._cache_last_channel_len = None
        self._previous_hypotheses = None
        self._previous_pred_out = None
        self._step_num = 0
        self._hypothesis = ""
        self._committed = ""
        self._pending_commit = False
        self._silent_audio_bytes = 0
        self._gap_committed = False

    async def start_session(self) -> None:
        self._ensure_model()
        streaming_utils = importlib.import_module(
            "nemo.collections.asr.parts.utils.streaming_utils"
        )
        model = self.__class__._model
        encoder = model.encoder
        if self.right_context is not None and hasattr(
            encoder, "set_default_att_context_size"
        ):
            try:
                encoder.set_default_att_context_size(
                    att_context_size=[70, self.right_context]
                )
            except Exception as exc:
                logger.warning(f"Could not set att_context_size: {exc}")
        self._reset_session_state()
        self._buffer = streaming_utils.CacheAwareStreamingAudioBuffer(model)
        cache = encoder.get_initial_cache_state(batch_size=1)
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = cache

    def _ensure_model(self) -> None:
        checkpoint = ensure_local_checkpoint(
            self.model_name, self.model_path, self.models_dir
        )
        key = (str(checkpoint), self.device)
        if self.__class__._model is not None and self.__class__._model_key == key:
            return
        with self.__class__._model_lock:
            if (
                self.__class__._model is not None
                and self.__class__._model_key == key
            ):
                return
            try:
                model = _load_nemo_model("", str(checkpoint), self.device)
            except Exception as exc:
                raise RuntimeError(f"Unable to load streaming model: {exc}") from exc
            if not hasattr(model, "conformer_stream_step"):
                raise RuntimeError(
                    f"Model '{checkpoint}' has no conformer_stream_step — it is "
                    "offline-only. Use VadSegmentedOfflineASR for offline .nemo "
                    "files, or a streaming checkpoint "
                    "(nemotron-speech-streaming-en-0.6b, "
                    "parakeet_realtime_eou_120m-v1)."
                )
            self.__class__._model = model
            self.__class__._model_key = key

    async def push_audio(self, audio_chunk: bytes) -> str | None:
        if self._buffer is None:
            raise RuntimeError("Session not started; call start_session() first.")
        if not audio_chunk:
            return None
        if _rms(audio_chunk) < self.SILENCE_RMS_THRESHOLD:
            self._silent_audio_bytes += len(audio_chunk)
        else:
            self._silent_audio_bytes = 0
            self._gap_committed = False

        waveform = _pcm16_to_float32(audio_chunk)
        with self.__class__._step_lock:
            self._buffer.append_audio(waveform)
            self._drain_new_chunks()

        utterance_ended = self._consume_eou_boundary()
        if utterance_ended and not self._gap_committed:
            self._promote_commit()
            self._gap_committed = True
        elif self._silence_boundary_reached() and not self._gap_committed:
            self._promote_commit()
            self._gap_committed = True
        return _strip_committed_prefix(self._committed, self._hypothesis) or None

    async def push_demo_text(self, text: str) -> str | None:
        raise RuntimeError("mock_text is available only with ASR_BACKEND=mock")

    async def finalize(self) -> str:
        if self._buffer is None:
            raise RuntimeError("Session not started; call start_session() first.")
        with self.__class__._step_lock:
            self._drain_new_chunks()
        self._promote_commit()
        return self._committed

    async def reset(self) -> None:
        self._reset_session_state()

    @property
    def pending_commit(self) -> bool:
        return self._pending_commit

    @property
    def cumulative_text(self) -> str:
        return self._committed

    def consume_pending_commit(self) -> str:
        self._pending_commit = False
        return self._committed

    @property
    def chunk_latency_ms(self) -> int | None:
        return self.RIGHT_CONTEXT_LATENCY_MS.get(self.right_context)

    def _drain_new_chunks(self) -> None:
        """Run cache-aware steps for newly appended audio only."""
        import torch

        model = self.__class__._model
        for chunk_audio, chunk_lengths in self._buffer:
            drop_extra = (
                0
                if self._step_num == 0
                else model.encoder.streaming_cfg.drop_extra_pre_encoded
            )
            with torch.inference_mode(), torch.no_grad():
                (
                    pred_out,
                    transcribed,
                    cache_channel,
                    cache_time,
                    cache_channel_len,
                    best_hyp,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self._cache_last_channel,
                    cache_last_time=self._cache_last_time,
                    cache_last_channel_len=self._cache_last_channel_len,
                    keep_all_outputs=self._buffer.is_buffer_empty(),
                    previous_hypotheses=self._previous_hypotheses,
                    previous_pred_out=self._previous_pred_out,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )
            self._cache_last_channel = cache_channel
            self._cache_last_time = cache_time
            self._cache_last_channel_len = cache_channel_len
            self._previous_pred_out = pred_out
            # RNNT returns Hypothesis objects; keep them for the next step.
            try:
                self._previous_hypotheses = list(best_hyp) if best_hyp else None
            except TypeError:
                self._previous_hypotheses = None
            text = _extract_text(transcribed)
            if text:
                self._hypothesis = text
            self._step_num += 1

    def _consume_eou_boundary(self) -> bool:
        cleaned, found = _strip_eou_markers(self._hypothesis)
        if found:
            self._hypothesis = cleaned
        return found

    def _silence_boundary_reached(self) -> bool:
        silence_bytes_needed = int(
            SAMPLE_RATE * SAMPLE_WIDTH * self.SILENCE_COMMIT_SECONDS
        )
        return (
            self._silent_audio_bytes >= silence_bytes_needed
            and bool(self._hypothesis.strip())
        )

    def _promote_commit(self) -> None:
        if self._hypothesis != self._committed:
            self._committed = self._hypothesis
            self._pending_commit = True


class VadSegmentedOfflineASR(StreamingASR):
    """Fallback for offline-only local ``.nemo`` files (e.g. parakeet-tdt).

    Speech is cut at pauses; each segment is decoded exactly once via
    ``transcribe()`` and appended. No window, no overlap, no merge — repeats
    and boundary drops are impossible by construction. Trade-off: no partials
    mid-segment; partials appear per completed segment.
    """

    _model = None
    _model_key: tuple | None = None
    _model_lock = threading.Lock()

    SILENCE_RMS_THRESHOLD = 500.0
    SILENCE_COMMIT_SECONDS = 0.6
    MIN_SEGMENT_SECONDS = 0.4
    MAX_SEGMENT_SECONDS = 20.0

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        model_path: str = "",
        device: str = "auto",
        models_dir: str | Path = "models",
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.models_dir = Path(models_dir)
        self._segment = bytearray()
        self._speech_bytes = 0
        self._silent_audio_bytes = 0
        self._committed = ""
        self._pending_commit = False

    async def start_session(self) -> None:
        self._ensure_model()
        self._segment.clear()
        self._speech_bytes = 0
        self._silent_audio_bytes = 0
        self._committed = ""
        self._pending_commit = False

    def _ensure_model(self) -> None:
        checkpoint = ensure_local_checkpoint(
            self.model_name, self.model_path, self.models_dir
        )
        key = (str(checkpoint), self.device)
        if self.__class__._model is not None and self.__class__._model_key == key:
            return
        with self.__class__._model_lock:
            if (
                self.__class__._model is not None
                and self.__class__._model_key == key
            ):
                return
            try:
                model = _load_nemo_model("", str(checkpoint), self.device)
            except Exception as exc:
                raise RuntimeError(f"Unable to load offline model: {exc}") from exc
            self.__class__._model = model
            self.__class__._model_key = key

    async def push_audio(self, audio_chunk: bytes) -> str | None:
        if self.__class__._model is None:
            raise RuntimeError("Session not started; call start_session() first.")
        if not audio_chunk:
            return None
        self._segment.extend(audio_chunk)
        if _rms(audio_chunk) < self.SILENCE_RMS_THRESHOLD:
            self._silent_audio_bytes += len(audio_chunk)
        else:
            self._speech_bytes += len(audio_chunk)
            self._silent_audio_bytes = 0

        segment_seconds = len(self._segment) / (SAMPLE_RATE * SAMPLE_WIDTH)
        silence_seconds = self._silent_audio_bytes / (SAMPLE_RATE * SAMPLE_WIDTH)
        speech_seconds = self._speech_bytes / (SAMPLE_RATE * SAMPLE_WIDTH)
        if speech_seconds >= self.MIN_SEGMENT_SECONDS and (
            silence_seconds >= self.SILENCE_COMMIT_SECONDS
            or segment_seconds >= self.MAX_SEGMENT_SECONDS
        ):
            return self._decode_segment()
        return None

    async def push_demo_text(self, text: str) -> str | None:
        raise RuntimeError("mock_text is available only with ASR_BACKEND=mock")

    async def finalize(self) -> str:
        if self.__class__._model is None:
            raise RuntimeError("Session not started; call start_session() first.")
        if self._speech_bytes / (SAMPLE_RATE * SAMPLE_WIDTH) >= 0.2:
            self._decode_segment()
        else:
            self._segment.clear()
            self._speech_bytes = 0
            self._silent_audio_bytes = 0
        return self._committed

    async def reset(self) -> None:
        self._segment.clear()
        self._speech_bytes = 0
        self._silent_audio_bytes = 0
        self._committed = ""
        self._pending_commit = False

    @property
    def pending_commit(self) -> bool:
        return self._pending_commit

    @property
    def cumulative_text(self) -> str:
        return self._committed

    def consume_pending_commit(self) -> str:
        self._pending_commit = False
        return self._committed

    def _decode_segment(self) -> str | None:
        """Decode the current segment exactly once and append it."""
        waveform = _pcm16_to_float32(bytes(self._segment))
        self._segment.clear()
        self._speech_bytes = 0
        self._silent_audio_bytes = 0
        if waveform.size == 0:
            return None
        with self.__class__._model_lock:
            hyps = self.__class__._model.transcribe([waveform])
        text = _extract_text(hyps)
        if not text:
            return None
        previous = self._committed
        self._committed = f"{previous} {text}".strip() if previous else text
        self._pending_commit = self._committed != previous
        return text
