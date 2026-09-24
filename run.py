import os

import uvicorn

if __name__ == "__main__":
    # Cache-aware streaming (local after first download, then fully offline):
    #   nvidia/nemotron-speech-streaming-en-0.6b (default, PnC) |
    #   nvidia/parakeet_realtime_eou_120m-v1 (lowest latency, EOU turn-taking)
    os.environ.setdefault("ASR_BACKEND", "nemotron")
    os.environ.setdefault("ASR_DEVICE", "cuda")
    os.environ.setdefault("ASR_MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b")
    # ASR_MODEL_PATH optionally points at a local streaming .nemo checkpoint.
    # Missing weights auto-download into ASR_MODELS_DIR (default "models/") on
    # first use, then run fully offline.
    # Chunk latency: ASR_CHUNK_RIGHT_CONTEXT in {0,1,6,13} -> {80,160,560,1120} ms.
    os.environ.setdefault("ASR_CHUNK_RIGHT_CONTEXT", "6")
    # Fallback for the existing offline file (VAD-segmented, no repeats):
    #   ASR_BACKEND=offline + ASR_MODEL_PATH=models/parakeet/parakeet-tdt-0.6b-v3.nemo
    uvicorn.run("clinical_asr.main:app", host="127.0.0.1", port=8000, reload=False)
