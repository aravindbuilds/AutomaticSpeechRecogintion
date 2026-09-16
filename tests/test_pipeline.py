from pathlib import Path
from fastapi.testclient import TestClient

from clinical_asr.main import app
from clinical_asr.resolver import TerminologyResolver


def test_resolver_resolves_known_medication():
    resolver = TerminologyResolver(Path(__file__).parent.parent / "vocabulary.json")
    text, terms = resolver.apply("Patient takes ram a pro five milligrams once daily")
    assert "ramipril" in text
    assert any(term["category"] == "medication" for term in terms)


def test_mock_websocket_emits_pipeline_events():
    with TestClient(app) as client:
        with client.websocket_connect("/v1/transcribe/stream") as websocket:
            startup_types = {websocket.receive_json()["type"] for _ in range(3)}
            assert startup_types == {"session_started", "model_loading", "model_ready"}
            websocket.send_json({"type": "mock_text", "text": "Patient takes ram a pro"})
            event_types = {websocket.receive_json()["type"] for _ in range(3)}
            assert "partial_transcript" in event_types
            assert "terminology_update" in event_types
            websocket.send_json({"type": "finalize"})
            final_types = {websocket.receive_json()["type"], websocket.receive_json()["type"], websocket.receive_json()["type"]}
            assert "final_transcript" in final_types
            assert "session_finished" in final_types
