import os

import uvicorn

if __name__ == "__main__":
    os.environ.setdefault("ASR_BACKEND", "parakeet")
    os.environ.setdefault("ASR_DEVICE", "cuda")
    os.environ.setdefault("ASR_MODEL_PATH", "models/parakeet/parakeet-tdt-0.6b-v3.nemo")
    uvicorn.run("clinical_asr.main:app", host="127.0.0.1", port=8000, reload=False)
