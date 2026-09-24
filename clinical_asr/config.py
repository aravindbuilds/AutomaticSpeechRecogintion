from dataclasses import dataclass
import os


def _right_context() -> int:
    try:
        return int(os.getenv("ASR_CHUNK_RIGHT_CONTEXT", "6"))
    except ValueError:
        return 6


@dataclass(frozen=True)
class Settings:
    # Backends: "nemotron" (cache-aware streaming, default), "offline"
    # (VAD-segmented decode of an offline-only .nemo file), "mock".
    # Legacy values "parakeet" and "onnx" map to "offline".
    backend: str = os.getenv("ASR_BACKEND", "mock")
    # Streaming checkpoint (local after first download, then fully offline):
    #   nvidia/nemotron-speech-streaming-en-0.6b (default, PnC) |
    #   nvidia/parakeet_realtime_eou_120m-v1 (EOU turn-taking) |
    #   nvidia/nemotron-3.5-asr-streaming-0.6b (multilingual)
    model_name: str = os.getenv("ASR_MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b")
    # Optional local .nemo path. For "offline" this is the offline model file;
    # for "nemotron" it is an optional local streaming checkpoint.
    model_path: str = os.getenv("ASR_MODEL_PATH", "")
    # Root folder for weights. Missing weights are auto-downloaded here
    # (models/<repo-slug>/*.nemo) on first use, then run fully offline.
    models_dir: str = os.getenv("ASR_MODELS_DIR", "models")
    device: str = os.getenv("ASR_DEVICE", "auto")
    # Streaming chunk latency: right-context frames {0, 1, 6, 13} -> chunk
    # {80, 160, 560, 1120} ms. 6 is the balanced default; 1 for low latency.
    chunk_right_context: int = _right_context()
    host: str = os.getenv("ASR_HOST", "127.0.0.1")
    port: int = int(os.getenv("ASR_PORT", "8000"))
    sample_rate: int = 16000
    channels: int = 1
    min_term_confidence: float = 0.88
    review_term_confidence: float = 0.65
