from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    backend: str = os.getenv("ASR_BACKEND", "mock")
    model_name: str = os.getenv("ASR_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")
    model_path: str = os.getenv("ASR_MODEL_PATH", "models/parakeet/parakeet-tdt-0.6b-v3.nemo")
    device: str = os.getenv("ASR_DEVICE", "auto")
    host: str = os.getenv("ASR_HOST", "127.0.0.1")
    port: int = int(os.getenv("ASR_PORT", "8000"))
    sample_rate: int = 16000
    channels: int = 1
    min_term_confidence: float = 0.88
    review_term_confidence: float = 0.65
