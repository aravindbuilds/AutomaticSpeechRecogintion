import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .backends import MockStreamingASR, ParakeetStreamingASR
from .config import Settings
from .events import TranscriptEvent
from .resolver import TerminologyResolver

logger = logging.getLogger(__name__)
settings = Settings()

asr_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr-inference")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    asr_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Offline Clinical ASR", version="0.2.0", lifespan=lifespan)
root = Path(__file__).parent.parent
resolver = TerminologyResolver(root / "vocabulary.json", settings.min_term_confidence, settings.review_term_confidence)
app.mount("/static", StaticFiles(directory=root / "static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(root / "static" / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend": settings.backend, "model": settings.model_path or settings.model_name, "device": settings.device}


@app.get("/voice")
async def voice() -> FileResponse:
    return FileResponse(root / "static" / "voice-ui.html")


def make_backend():
    return ParakeetStreamingASR(settings.model_name, settings.model_path, settings.device) if settings.backend == "parakeet" else MockStreamingASR()


async def run_backend(operation, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(asr_executor, lambda: asyncio.run(operation(*args)))


async def send_json(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


@app.websocket("/v1/transcribe/stream")
async def transcribe(websocket: WebSocket) -> None:
    SESSION_TIMEOUT = 600
    MAX_CONSECUTIVE_ERRORS = 3

    await websocket.accept()
    session_id = str(uuid.uuid4())
    backend = make_backend()
    started = time.monotonic()
    speech_started = False
    audio_bytes = 0
    audio_chunks = 0
    last_audio_time = time.monotonic()
    consecutive_errors = 0
    last_partial_key = None
    last_committed_event = ""

    if not await send_json(websocket, TranscriptEvent(type="session_started", session_id=session_id).to_dict()):
        return

    if not await send_json(websocket, TranscriptEvent(type="model_loading", session_id=session_id).to_dict()):
        return

    try:
        await run_backend(backend.start_session)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.error(f"Model load failed: {exc}", exc_info=True)
        await send_json(websocket, TranscriptEvent(type="error", error=f"Model load failed: {exc}", session_id=session_id).to_dict())
        return

    if not await send_json(websocket, TranscriptEvent(type="model_ready", session_id=session_id).to_dict()):
        return

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=SESSION_TIMEOUT)
            except asyncio.TimeoutError:
                await send_json(websocket, TranscriptEvent(type="error", error="Session timeout due to inactivity", session_id=session_id).to_dict())
                return
            except (WebSocketDisconnect, RuntimeError):
                return

            if message.get("bytes") is not None:
                chunk = message["bytes"]
                audio_bytes += len(chunk)
                audio_chunks += 1
                last_audio_time = time.monotonic()
                if chunk and not speech_started:
                    speech_started = True
                    if not await send_json(websocket, TranscriptEvent(type="speech_started", session_id=session_id).to_dict()):
                        return

                pending_commit_text = None
                try:
                    text = await run_backend(backend.push_audio, chunk)
                    if getattr(backend, "pending_commit", False):
                        pending_commit_text = backend.consume_pending_commit()
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    logger.error(f"Inference error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {exc}", exc_info=True)
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        await send_json(websocket, TranscriptEvent(type="error", error=f"Model failed after {MAX_CONSECUTIVE_ERRORS} consecutive errors", session_id=session_id).to_dict())
                        return
                    await send_json(websocket, TranscriptEvent(type="warning", text=f"Inference error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}). Recovering...", session_id=session_id).to_dict())
                    await run_backend(backend.reset)
                    speech_started = False
                    text = None

                if audio_chunks == 1 or audio_chunks % 10 == 0:
                    if not await send_json(websocket, TranscriptEvent(type="audio_received", text=f"{audio_bytes} bytes in {audio_chunks} chunks", session_id=session_id).to_dict()):
                        return

                if text:
                    resolved, terms = resolver.apply(text)
                    committed, _ = resolver.apply(getattr(backend, "cumulative_text", ""))
                    partial_key = (resolved, committed)
                    if partial_key != last_partial_key:
                        if not await send_json(websocket, TranscriptEvent(type="partial_transcript", text=resolved, committed=committed, active_only=True, confidence=0.9, terms=terms, session_id=session_id, start_ms=0, end_ms=int((time.monotonic() - started) * 1000)).to_dict()):
                            return
                        last_partial_key = partial_key
                        last_committed_event = committed
                    if terms:
                        if not await send_json(websocket, TranscriptEvent(type="terminology_update", terms=terms, session_id=session_id).to_dict()):
                            return
                elif pending_commit_text:
                    resolved_commit, _ = resolver.apply(pending_commit_text)
                    if resolved_commit != last_committed_event:
                        if not await send_json(websocket, TranscriptEvent(type="committed_transcript", text=resolved_commit, session_id=session_id).to_dict()):
                            return
                        last_committed_event = resolved_commit
            else:
                payload = message.get("text")
                if payload is None:
                    continue
                import json
                command = json.loads(payload)
                if command.get("type") == "mock_text":
                    speech_started = True
                    text = await run_backend(backend.push_demo_text, command.get("text", ""))
                    if not await send_json(websocket, TranscriptEvent(type="speech_started", session_id=session_id).to_dict()):
                        return
                    if text:
                        resolved, terms = resolver.apply(text)
                        if not await send_json(websocket, TranscriptEvent(type="partial_transcript", text=resolved, confidence=0.9, terms=terms, session_id=session_id, start_ms=0, end_ms=int((time.monotonic() - started) * 1000)).to_dict()):
                            return
                        if terms:
                            if not await send_json(websocket, TranscriptEvent(type="terminology_update", terms=terms, session_id=session_id).to_dict()):
                                return
                elif command.get("type") == "finalize":
                    try:
                        text = await run_backend(backend.finalize)
                    except Exception as exc:
                        logger.error(f"Finalize error: {exc}", exc_info=True)
                        text = ""
                        await send_json(websocket, TranscriptEvent(type="warning", text=f"Finalize error: {exc}", session_id=session_id).to_dict())

                    resolved, terms = resolver.apply(text)
                    if speech_started:
                        if not await send_json(websocket, TranscriptEvent(type="speech_ended", session_id=session_id).to_dict()):
                            return
                    if not await send_json(websocket, TranscriptEvent(type="final_transcript", text=resolved, confidence=0.9, terms=terms, session_id=session_id).to_dict()):
                        return
                    if not await send_json(websocket, TranscriptEvent(type="session_finished", session_id=session_id).to_dict()):
                        return
                    await run_backend(backend.reset)
                    speech_started = False
                    audio_bytes = 0
                    audio_chunks = 0
                    consecutive_errors = 0
                    last_partial_key = None
                    last_committed_event = ""
                    continue
                else:
                    continue
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.error(f"Unexpected session error: {exc}", exc_info=True)
        await send_json(websocket, TranscriptEvent(type="error", error=f"Session error: {exc}", session_id=session_id).to_dict())
