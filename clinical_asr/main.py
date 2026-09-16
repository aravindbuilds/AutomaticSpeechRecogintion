import asyncio
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .backends import MockStreamingASR, ParakeetStreamingASR
from .config import Settings
from .events import TranscriptEvent
from .resolver import TerminologyResolver

settings = Settings()
app = FastAPI(title="Offline Clinical ASR", version="0.1.0")
root = Path(__file__).parent.parent
resolver = TerminologyResolver(root / "vocabulary.json", settings.min_term_confidence, settings.review_term_confidence)
app.mount("/static", StaticFiles(directory=root / "static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(root / "static" / "index.html")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "backend": settings.backend,
        "model": settings.model_path or settings.model_name,
        "device": settings.device,
    }


def make_backend():
    return ParakeetStreamingASR(settings.model_name, settings.model_path, settings.device) if settings.backend == "parakeet" else MockStreamingASR()


async def run_backend(operation, *args):
    return await asyncio.to_thread(lambda: asyncio.run(operation(*args)))


@app.websocket("/v1/transcribe/stream")
async def transcribe(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = str(uuid.uuid4())
    backend = make_backend()
    started = time.monotonic()
    speech_started = False
    audio_bytes = 0
    audio_chunks = 0
    await websocket.send_json(TranscriptEvent(type="session_started", session_id=session_id).to_dict())
    await websocket.send_json(TranscriptEvent(type="model_loading", session_id=session_id).to_dict())
    try:
        await run_backend(backend.start_session)
    except Exception as exc:
        await websocket.send_json(TranscriptEvent(type="error", error=str(exc), session_id=session_id).to_dict())
        await websocket.close()
        return
    await websocket.send_json(TranscriptEvent(type="model_ready", session_id=session_id).to_dict())
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                chunk = message["bytes"]
                audio_bytes += len(chunk)
                audio_chunks += 1
                if chunk and not speech_started:
                    speech_started = True
                    await websocket.send_json(TranscriptEvent(type="speech_started", session_id=session_id).to_dict())
                text = await run_backend(backend.push_audio, chunk)
                if audio_chunks == 1 or audio_chunks % 10 == 0:
                    await websocket.send_json(TranscriptEvent(type="audio_received", text=f"{audio_bytes} bytes in {audio_chunks} chunks", session_id=session_id).to_dict())
            else:
                payload = message.get("text")
                if payload is None:
                    continue
                import json
                command = json.loads(payload)
                if command.get("type") == "mock_text":
                    speech_started = True
                    text = await run_backend(backend.push_demo_text, command.get("text", ""))
                    await websocket.send_json(TranscriptEvent(type="speech_started", session_id=session_id).to_dict())
                elif command.get("type") == "finalize":
                    text = await run_backend(backend.finalize)
                    resolved, terms = resolver.apply(text)
                    if speech_started:
                        await websocket.send_json(TranscriptEvent(type="speech_ended", session_id=session_id).to_dict())
                    await websocket.send_json(TranscriptEvent(type="final_transcript", text=resolved, confidence=0.9, terms=terms, session_id=session_id).to_dict())
                    await websocket.send_json(TranscriptEvent(type="session_finished", session_id=session_id).to_dict())
                    break
                else:
                    continue
            if text:
                resolved, terms = resolver.apply(text)
                await websocket.send_json(TranscriptEvent(type="partial_transcript", text=resolved, confidence=0.9, terms=terms, session_id=session_id, start_ms=0, end_ms=int((time.monotonic() - started) * 1000)).to_dict())
                if terms:
                    await websocket.send_json(TranscriptEvent(type="terminology_update", terms=terms, session_id=session_id).to_dict())
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json(TranscriptEvent(type="error", error=str(exc), session_id=session_id).to_dict())
