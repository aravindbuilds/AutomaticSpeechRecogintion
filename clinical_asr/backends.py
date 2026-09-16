from abc import ABC, abstractmethod
import importlib
import struct
import threading
from pathlib import Path


def _load_parakeet_model(model_name: str, model_path: str, device: str):
    """Load once per process; model files must already be cached for offline use."""
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


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


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
    """Deterministic backend for contract and UI testing without model weights."""

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
    """Parakeet adapter using NeMo inference over the current utterance buffer."""

    _model = None
    _model_lock = threading.Lock()

    def __init__(self, model_name: str, model_path: str = "", device: str = "auto") -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.audio = bytearray()
        self.min_inference_bytes = 16000
        self.last_inference_size = 0

    async def start_session(self) -> None:
        if self.__class__._model is None:
            with self.__class__._model_lock:
                if self.__class__._model is None:
                    try:
                        self.__class__._model = _load_parakeet_model(self.model_name, self.model_path, self.device)
                    except Exception as exc:
                        raise RuntimeError(
                            "Unable to load Parakeet. Install nemo_toolkit[asr] and provide a cached "
                            f"model via ASR_MODEL_PATH, or allow the first download: {exc}"
                        ) from exc
        self.audio.clear()
        self.last_inference_size = 0

    async def push_audio(self, audio_chunk: bytes) -> str | None:
        self.audio.extend(audio_chunk)
        if len(self.audio) - self.last_inference_size >= self.min_inference_bytes:
            return self._transcribe()
        return None

    async def push_demo_text(self, text: str) -> str | None:
        raise RuntimeError("mock_text is available only with ASR_BACKEND=mock")

    async def finalize(self) -> str:
        return self._transcribe() or ""

    async def reset(self) -> None:
        self.audio.clear()
        self.last_inference_size = 0

    def _transcribe(self) -> str:
        if not self.audio:
            return ""
        self.last_inference_size = len(self.audio)
        import numpy as np
        samples = struct.unpack("<%dh" % (len(self.audio) // 2), self.audio[: len(self.audio) // 2 * 2])
        waveform = np.asarray(samples, dtype=np.float32) / 32768.0
        result = self.__class__._model.transcribe([waveform], batch_size=1)
        output = result[0] if isinstance(result, list) else result
        return getattr(output, "text", str(output)).strip()
