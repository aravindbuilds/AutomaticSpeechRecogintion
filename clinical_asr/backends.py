import asyncio
import logging
import struct
import threading
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from pathlib import Path

import importlib

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _load_parakeet_model(model_name: str, model_path: str, device: str):
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


class ParakeetStreamingASR(StreamingASR):
    """
    Streaming ASR with cumulative transcript tracking.

    Protocol:
    - partial_transcript: text = the complete latest audio-window hypothesis
    - committed_transcript: text = full cumulative text to render as white
    - final_transcript: text = full cumulative text, everything goes white

    The client renders:
    - committed text (from committed_transcript) as white
    - partial text (from partial_transcript) as grey delta appended after
    """

    _model = None
    _model_lock = threading.Lock()

    SAMPLE_RATE = 16000
    SAMPLE_WIDTH = 2
    MAX_AUDIO_SECONDS = 10
    MAX_AUDIO_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * MAX_AUDIO_SECONDS
    # Keep a sizeable overlap so the pre/post-trim text can be aligned.
    TRIM_TARGET_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * 8
    MIN_INFERENCE_BYTES = 16000
    # A sustained low-energy interval is treated as a sentence boundary. The
    # audio is deliberately kept in the rolling buffer; this only promotes
    # the current hypothesis so resumed speech can continue in the same
    # session without losing the first words after the pause.
    SILENCE_RMS_THRESHOLD = 500.0
    SILENCE_COMMIT_SECONDS = 0.6
    SILENCE_COMMIT_BYTES = int(SAMPLE_RATE * SAMPLE_WIDTH * SILENCE_COMMIT_SECONDS)
    # Same-word timestamps within this interval are treated as one boundary
    # token. Words beginning more than one second apart remain distinct.
    DUPLICATE_WORD_START_TOLERANCE = 1.0

    def __init__(self, model_name: str, model_path: str = "", device: str = "auto") -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.audio = bytearray()
        self.last_inference_size = 0
        self._inference_count = 0
        self._last_error: str | None = None
        self._cumulative_text = ""      # full cumulative text (all white)
        self._cumulative_words: list[dict] = []
        self._last_buffer_text = ""     # last model output for current buffer
        self._last_buffer_words: list[dict] = []
        self._word_evidence: list[dict] = []
        self._pending_commit = False    # True when we need to send committed_transcript
        self._silent_audio_bytes = 0
        self._gap_committed = False
        self._audio_offset_seconds = 0.0

    async def start_session(self) -> None:
        if self.__class__._model is None:
            with self.__class__._model_lock:
                if self.__class__._model is None:
                    try:
                        self.__class__._model = _load_parakeet_model(
                            self.model_name, self.model_path, self.device
                        )
                    except Exception as exc:
                        raise RuntimeError(f"Unable to load Parakeet: {exc}") from exc
        self.audio.clear()
        self.last_inference_size = 0
        self._inference_count = 0
        self._last_error = None
        self._cumulative_text = ""
        self._cumulative_words = []
        self._last_buffer_text = ""
        self._last_buffer_words = []
        self._word_evidence = []
        self._pending_commit = False
        self._silent_audio_bytes = 0
        self._gap_committed = False
        self._audio_offset_seconds = 0.0

    async def push_audio(self, audio_chunk: bytes) -> str | None:
        self.audio.extend(audio_chunk)

        if self._audio_is_silent(audio_chunk):
            self._silent_audio_bytes += len(audio_chunk)
        elif audio_chunk:
            self._silent_audio_bytes = 0
            self._gap_committed = False

        # --- Before trimming: commit current transcription ---
        if len(self.audio) > self.MAX_AUDIO_BYTES:
            # Capture the complete hypothesis for the old window.
            await self._safe_transcribe(force=True)
            before_trim_text = self._last_buffer_text

            trim_target = min(self.TRIM_TARGET_BYTES, self.MAX_AUDIO_BYTES)
            excess = max(0, len(self.audio) - trim_target)
            cutoff_seconds = self._audio_offset_seconds + excess / (self.SAMPLE_RATE * self.SAMPLE_WIDTH)

            # With timestamps, commit only words that will actually leave
            # the rolling window. The overlap stays grey after trimming.
            if self._last_buffer_words:
                self._commit_words_until(cutoff_seconds)
            else:
                self._commit_current_buffer(before_trim_text)

            # Trim by a larger chunk, then infer the actual new window. This
            # gives us the new grey active-window hypothesis.
            self.audio = self.audio[excess:]
            self._audio_offset_seconds += excess / (self.SAMPLE_RATE * self.SAMPLE_WIDTH)
            self._word_evidence = [
                word for word in self._word_evidence
                if float(word.get("end", word.get("start", 0.0))) > self._audio_offset_seconds - 0.05
            ]
            self.last_inference_size = 0
            await self._safe_transcribe(force=True)
            after_trim_text = self._last_buffer_text

            # If the overflow happened during a pause, promote the new
            # post-trim hypothesis too. This leaves the buffer intact for
            # the next words instead of ending/restarting the ASR session.
            if self._silence_boundary_reached() and not self._gap_committed:
                self._commit_current_buffer(
                    after_trim_text,
                    self._committable_words(include_unstable=True) or self._last_buffer_words,
                )
                self._gap_committed = True
            return self._active_tail(
                self._cumulative_text, after_trim_text, self._last_buffer_words
            ) or None

        # Commit once per sustained pause. Do not clear audio or reset the
        # model context: the next non-silent chunk is still appended to the
        # same rolling window and can be reconciled against this white text.
        if self._silence_boundary_reached() and not self._gap_committed:
            await self._safe_transcribe(force=True)
            self._commit_current_buffer(
                self._last_buffer_text,
                self._committable_words(include_unstable=True) or self._last_buffer_words,
            )
            self._gap_committed = True
            return self._active_tail(
                self._cumulative_text, self._last_buffer_text, self._last_buffer_words
            ) or None

        unprocessed = len(self.audio) - self.last_inference_size
        if unprocessed >= self.MIN_INFERENCE_BYTES:
            await self._safe_transcribe()
            return self._active_tail(
                self._cumulative_text, self._last_buffer_text, self._last_buffer_words
            ) or None
        return None

    async def push_demo_text(self, text: str) -> str | None:
        raise RuntimeError("mock_text is available only with ASR_BACKEND=mock")

    async def finalize(self) -> str:
        await self._safe_transcribe(force=True)
        self._commit_current_buffer(
            self._last_buffer_text,
            self._committable_words(include_unstable=True) or self._last_buffer_words,
        )
        return self._cumulative_text

    async def reset(self) -> None:
        self.audio.clear()
        self.last_inference_size = 0
        self._inference_count = 0
        self._last_error = None
        self._cumulative_text = ""
        self._cumulative_words = []
        self._last_buffer_text = ""
        self._last_buffer_words = []
        self._word_evidence = []
        self._pending_commit = False
        self._silent_audio_bytes = 0
        self._gap_committed = False
        self._audio_offset_seconds = 0.0

    @property
    def pending_commit(self) -> bool:
        return self._pending_commit

    @property
    def cumulative_text(self) -> str:
        return self._cumulative_text

    def consume_pending_commit(self) -> str:
        """Get cumulative text and clear pending flag."""
        self._pending_commit = False
        return self._cumulative_text

    def _commit_current_buffer(self, text: str, words: list[dict] | None = None) -> None:
        """Promote the current hypothesis without dropping its audio."""
        previous = self._cumulative_text
        if words:
            merged_words = self._merge_timestamped_words(self._cumulative_words, words)
            self._cumulative_words = merged_words
            self._cumulative_text = self._dedupe_adjacent_text(
                " ".join(word["word"] for word in merged_words).strip()
            )
        else:
            self._cumulative_text = self._dedupe_adjacent_text(
                self._append_segment(previous, text)
            )
        self._pending_commit = self._pending_commit or self._cumulative_text != previous

    def _commit_words_until(self, cutoff_seconds: float) -> None:
        """Commit observed words that have left the rolling window."""
        words = self._committable_words(cutoff_seconds)
        eligible = [
            word for word in words
            if float(word.get("end", word.get("start", 0.0))) <= cutoff_seconds + 0.05
        ]
        if eligible:
            self._commit_current_buffer("", eligible)

    def _committable_words(
        self,
        cutoff_seconds: float | None = None,
        include_unstable: bool = False,
    ) -> list[dict]:
        """Return timestamped words with enough evidence to retain.

        A word seen in two hypotheses is stable. Words from the latest
        hypothesis are also allowed through so the first inference of a new
        phrase is not delayed; older hypotheses are retained when the latest
        decode temporarily drops a word.
        """
        if not self._word_evidence:
            return []
        latest = self._inference_count
        words = [
            word for word in self._word_evidence
            if (
                include_unstable
                or word.get("seen_count", 0) >= 2
                or word.get("last_seen") == latest
            )
            and (
                cutoff_seconds is None
                or float(word.get("end", word.get("start", 0.0))) <= cutoff_seconds + 0.05
            )
        ]
        return self._collapse_adjacent_timestamp_duplicates(words)

    def _observe_words(self, words: list[dict]) -> None:
        """Accumulate short-lived timestamp evidence across ASR updates."""
        for incoming in self._collapse_adjacent_timestamp_duplicates(words):
            start = float(incoming.get("start", 0.0))
            end = float(incoming.get("end", start))
            match = None
            best_distance = float("inf")
            for existing in self._word_evidence:
                existing_start = float(existing.get("start", 0.0))
                existing_end = float(existing.get("end", existing_start))
                overlap = max(0.0, min(end, existing_end) - max(start, existing_start))
                distance = abs(start - existing_start)
                if overlap > 0.04 or distance <= 0.22:
                    if distance < best_distance:
                        match = existing
                        best_distance = distance
            if match is None:
                self._word_evidence.append({
                    **incoming,
                    "seen_count": 1,
                    "last_seen": self._inference_count,
                })
            else:
                # Keep the newest spelling/capitalization while preserving
                # the original timing anchor for stable alignment.
                match["word"] = incoming["word"]
                match["seen_count"] = match.get("seen_count", 0) + 1
                match["last_seen"] = self._inference_count
        self._word_evidence = self._collapse_adjacent_timestamp_duplicates(self._word_evidence)

    @classmethod
    def _same_boundary_word(cls, left: dict, right: dict) -> bool:
        left_word = str(left.get("word", "")).strip(".,!?;:").lower()
        right_word = str(right.get("word", "")).strip(".,!?;:").lower()
        if not left_word or left_word != right_word:
            return False
        left_start = float(left.get("start", 0.0))
        right_start = float(right.get("start", 0.0))
        left_end = float(left.get("end", left_start))
        right_end = float(right.get("end", right_start))
        return (
            abs(left_start - right_start) <= cls.DUPLICATE_WORD_START_TOLERANCE
            or min(left_end, right_end) - max(left_start, right_start) > 0.04
        )

    @classmethod
    def _collapse_adjacent_timestamp_duplicates(cls, words: list[dict]) -> list[dict]:
        """Collapse stuttered copies produced at a window boundary."""
        ordered = sorted(words, key=lambda word: float(word.get("start", 0.0)))
        collapsed: list[dict] = []
        for word in ordered:
            if collapsed and cls._same_boundary_word(collapsed[-1], word):
                # Preserve the most recent spelling while keeping the wider
                # timing interval for future boundary comparisons.
                previous = collapsed[-1]
                previous["word"] = word.get("word", previous.get("word", ""))
                previous["end"] = max(
                    float(previous.get("end", previous.get("start", 0.0))),
                    float(word.get("end", word.get("start", 0.0))),
                )
                previous["seen_count"] = max(
                    previous.get("seen_count", 0), word.get("seen_count", 0)
                )
                previous["last_seen"] = max(
                    previous.get("last_seen", 0), word.get("last_seen", 0)
                )
                continue
            collapsed.append(word)
        return collapsed

    @staticmethod
    def _merge_timestamped_words(base: list[dict], incoming: list[dict]) -> list[dict]:
        """Merge words by absolute audio time, keeping each word once."""
        if not base:
            return ParakeetStreamingASR._collapse_adjacent_timestamp_duplicates(list(incoming))

        merged = ParakeetStreamingASR._collapse_adjacent_timestamp_duplicates(list(base))
        boundary = max(float(word.get("end", word.get("start", 0.0))) for word in base)
        for word in ParakeetStreamingASR._collapse_adjacent_timestamp_duplicates(list(incoming)):
            if merged and ParakeetStreamingASR._same_boundary_word(merged[-1], word):
                continue
            end = float(word.get("end", word.get("start", 0.0)))
            if end <= boundary + 0.05:
                continue
            merged.append(word)
            boundary = end
        return ParakeetStreamingASR._collapse_adjacent_timestamp_duplicates(merged)

    def _silence_boundary_reached(self) -> bool:
        return (
            self._silent_audio_bytes >= self.SILENCE_COMMIT_BYTES
            and bool(self._last_buffer_text.strip())
        )

    @classmethod
    def _audio_is_silent(cls, audio_chunk: bytes) -> bool:
        """Return whether a PCM16 chunk is below the pause threshold."""
        sample_count = len(audio_chunk) // cls.SAMPLE_WIDTH
        if sample_count == 0:
            return True

        samples = struct.unpack(
            f"<{sample_count}h",
            audio_chunk[:sample_count * cls.SAMPLE_WIDTH],
        )
        mean_square = sum(sample * sample for sample in samples) / sample_count
        return mean_square ** 0.5 < cls.SILENCE_RMS_THRESHOLD

    @staticmethod
    def _append_segment(base: str, segment: str) -> str:
        """Append a newly committed/window segment without boundary repeats."""
        base = base.strip()
        segment = segment.strip()
        if not base:
            return segment
        if not segment:
            return base
        if segment.startswith(base):
            return segment

        base_words = base.split()
        segment_words = segment.split()
        # A rolling hypothesis can restart at an older phrase already inside
        # the white transcript, rather than at its final word. Treat that
        # prefix as context so a pause cannot commit the same window again.
        leading_overlap = ParakeetStreamingASR._leading_overlap(base_words, segment_words)
        if leading_overlap:
            segment_words = segment_words[leading_overlap:]
            if not segment_words:
                return base

        overlap = ParakeetStreamingASR._boundary_overlap(base_words, segment_words)
        if overlap:
            suffix = " ".join(segment_words[overlap:])
            return base if not suffix else f"{base} {suffix}"
        return f"{base} {' '.join(segment_words)}"

    @staticmethod
    def _dedupe_adjacent_text(text: str) -> str:
        """Remove exact adjacent stutters in text-only fallback output."""
        words = text.split()
        result = []
        for word in words:
            normalized = word.strip(".,!?;:").lower()
            if result and normalized == result[-1].strip(".,!?;:").lower():
                continue
            result.append(word)
        return " ".join(result)

    @staticmethod
    def _leading_overlap(base_words: list[str], incoming_words: list[str]) -> int:
        """Return a repeated incoming prefix found anywhere in the base."""
        if not base_words or not incoming_words:
            return 0

        normalize = lambda word: word.strip(".,!?;:").lower()
        base_norm = [normalize(word) for word in base_words]
        incoming_norm = [normalize(word) for word in incoming_words]

        # Prefer the longest exact prefix match. Requiring two words avoids
        # treating a common one-word opener such as "the" as alignment.
        for size in range(min(len(base_norm), len(incoming_norm)), 1, -1):
            prefix = incoming_norm[:size]
            if any(
                base_norm[start:start + size] == prefix
                for start in range(len(base_norm) - size + 1)
            ):
                return size
        return 0

    @staticmethod
    def _boundary_overlap(base_words: list[str], incoming_words: list[str]) -> int:
        """Return how many incoming words overlap the end of the base."""
        if not base_words or not incoming_words:
            return 0

        normalize = lambda word: word.strip(".,!?;:").lower()
        base_norm = [normalize(word) for word in base_words]
        incoming_norm = [normalize(word) for word in incoming_words]
        for size in range(min(len(base_norm), len(incoming_norm)), 1, -1):
            if base_norm[-size:] == incoming_norm[:size]:
                return size

        # Permit small ASR revisions while requiring the matching block to
        # be near the old transcript's end and the new hypothesis's start.
        matches = SequenceMatcher(None, base_norm, incoming_norm, autojunk=False).get_matching_blocks()
        candidates = [
            match for match in matches
            if match.size >= 2 and match.a + match.size >= len(base_norm) * 0.55
            and match.b <= max(4, len(incoming_norm) // 4)
        ]
        if candidates:
            match = max(candidates, key=lambda item: (item.size, item.a + item.size))
            return match.b + match.size
        return 0

    def _active_tail(
        self,
        committed: str,
        hypothesis: str,
        hypothesis_words: list[dict] | None = None,
    ) -> str:
        """Remove only the confirmed white/grey boundary overlap."""
        if hypothesis_words and self._word_evidence:
            boundary = (
                max(
                    float(word.get("end", word.get("start", 0.0)))
                    for word in self._cumulative_words
                )
                if self._cumulative_words
                else float("-inf")
            )
            # Render the union of recent timestamp evidence, not only the
            # newest decode. A transient omission in one hypothesis should
            # not make an already observed grey word disappear.
            observed_words = self._committable_words(include_unstable=True)
            active_words = [
                word for word in observed_words
                if float(word.get("end", word.get("start", 0.0))) > boundary + 0.05
            ]
            return self._dedupe_adjacent_text(
                " ".join(word["word"] for word in active_words).strip()
            )

        hypothesis_words = hypothesis.strip().split()
        if not committed.strip() or not hypothesis_words:
            return self._dedupe_adjacent_text(hypothesis.strip())
        committed_words = committed.strip().split()
        overlap = self._leading_overlap(committed_words, hypothesis_words)
        if overlap:
            return self._dedupe_adjacent_text(" ".join(hypothesis_words[overlap:]))

        overlap = self._boundary_overlap(committed_words, hypothesis_words)
        tail = " ".join(hypothesis_words[overlap:]) if overlap else hypothesis.strip()
        return self._dedupe_adjacent_text(tail)

    async def _safe_transcribe(self, force: bool = False) -> str | None:
        if not self.audio:
            return None

        sample_count = len(self.audio) // 2
        if not force and sample_count < self.MIN_INFERENCE_BYTES // 2:
            return None

        self.last_inference_size = len(self.audio)
        self._inference_count += 1

        try:
            result = self._transcribe_bytes(self.audio)
            if isinstance(result, tuple):
                result, words = result
            else:
                # Keep compatibility with test doubles and model adapters
                # that only return plain text.
                words = []
            self._last_error = None
            self._last_buffer_text = result
            self._last_buffer_words = words
            if words:
                self._observe_words(words)
            # Every inference is a revised hypothesis for the current audio
            # window. Return it as a replacement, never as an append-only
            # delta; only committed/final text is allowed to accumulate.
            return result
        except Exception as exc:
            error_msg = f"Inference #{self._inference_count} failed: {exc}"
            logger.error(error_msg, exc_info=True)
            self._last_error = error_msg
            self.audio.clear()
            self.last_inference_size = 0
            raise RuntimeError(error_msg) from exc

    def _transcribe_bytes(self, audio_bytes: bytes) -> tuple[str, list[dict]]:
        import numpy as np

        sample_count = len(audio_bytes) // 2
        samples = struct.unpack(f"<{sample_count}h", audio_bytes[:sample_count * 2])
        waveform = np.asarray(samples, dtype=np.float32) / 32768.0

        # Parakeet exposes word timestamps through Hypothesis.timestamp.
        # They let us align rolling windows by audio time instead of guessing
        # overlap from text that may be revised or repeated during silence.
        try:
            result = self.__class__._model.transcribe(
                [waveform], batch_size=1, timestamps=True
            )
        except TypeError:
            # Older NeMo builds may not accept timestamps in this call path.
            result = self.__class__._model.transcribe([waveform], batch_size=1)
        output = result[0] if isinstance(result, list) else result
        text = getattr(output, "text", str(output)).strip()

        timestamp_data = getattr(output, "timestamp", None)
        raw_words = timestamp_data.get("word", []) if isinstance(timestamp_data, dict) else []
        words = []
        for stamp in raw_words or []:
            if not isinstance(stamp, dict):
                continue
            word = str(stamp.get("word", stamp.get("text", ""))).strip()
            if not word:
                continue
            start = self._timestamp_float(stamp.get("start"))
            end = self._timestamp_float(stamp.get("end", stamp.get("start")))
            if start is None or end is None:
                continue
            words.append({
                "word": word,
                "start": self._audio_offset_seconds + start,
                "end": self._audio_offset_seconds + max(start, end),
            })
        return text, words

    @staticmethod
    def _timestamp_float(value) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
