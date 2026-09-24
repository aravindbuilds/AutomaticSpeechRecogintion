"""
Tests for the cache-aware streaming backends.

Both backends decode each audio frame exactly once (encoder caches /
single VAD-segment decode), so there is no rolling window to bound and no
overlap merge that could repeat or drop words. These tests verify the
model-free logic: silence gating, commit promotion, EOU handling, prefix
stripping, and single-decode segment semantics (with a stubbed model).
"""
import asyncio
import struct

import pytest

from pathlib import Path

from clinical_asr.backends import (
    MockStreamingASR,
    NemoCacheAwareStreamingASR,
    VadSegmentedOfflineASR,
    _extract_text,
    _pcm16_to_float32,
    _rms,
    _strip_committed_prefix,
    _strip_eou_markers,
    ensure_local_checkpoint,
)


def _pcm16(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


LOUD = _pcm16([8000] * 3200)  # 100 ms of loud speech
SILENT = bytes(3200)  # 100 ms of silence


class StubModel:
    """Stand-in for a NeMo model: records transcribe calls."""

    def __init__(self, texts: list[str] | None = None):
        self.texts = list(texts or ["hello world"])
        self.calls = 0

    def transcribe(self, waveforms):
        self.calls += 1
        if len(self.texts) > 1:
            return [self.texts.pop(0)]
        return list(self.texts)


@pytest.fixture
def stub_offline_model(monkeypatch):
    import clinical_asr.backends as backends

    stub = StubModel()
    monkeypatch.setattr(
        backends, "ensure_local_checkpoint", lambda *a, **k: Path("stub.nemo")
    )
    previous = VadSegmentedOfflineASR._model
    previous_key = VadSegmentedOfflineASR._model_key
    VadSegmentedOfflineASR._model = stub
    VadSegmentedOfflineASR._model_key = ("stub.nemo", "cpu")
    yield stub
    VadSegmentedOfflineASR._model = previous
    VadSegmentedOfflineASR._model_key = previous_key


class TestAudioHelpers:
    def test_pcm16_roundtrip(self):
        wave = _pcm16_to_float32(_pcm16([0, 16384, -16384]))
        assert wave.shape == (3,)
        assert abs(float(wave[1]) - 0.5) < 1e-4

    def test_rms_silence_vs_speech(self):
        assert _rms(SILENT) < NemoCacheAwareStreamingASR.SILENCE_RMS_THRESHOLD
        assert _rms(LOUD) > NemoCacheAwareStreamingASR.SILENCE_RMS_THRESHOLD

    def test_extract_text_handles_hypotheses_and_strings(self):
        class Hyp:
            text = "hello"

        assert _extract_text([Hyp(), " world "]) == "hello world"
        assert _extract_text([]) == ""


class TestCommitDisplay:
    def test_exact_prefix_stripped(self):
        tail = _strip_committed_prefix("patient takes", "patient takes ramipril")
        assert tail == "ramipril"

    def test_revision_falls_back_to_word_overlap(self):
        # Decoder re-emitted the boundary word: bounded overlap, no repeat.
        tail = _strip_committed_prefix("patient takes", "takes ramipril daily")
        assert tail == "ramipril daily"

    def test_no_overlap_returns_full_hypothesis(self):
        assert _strip_committed_prefix("abc def", "xyz") == "xyz"

    def test_eou_markers_stripped_and_detected(self):
        cleaned, found = _strip_eou_markers("thank you <EOU>")
        assert cleaned == "thank you"
        assert found is True
        _, not_found = _strip_eou_markers("thank you")
        assert not_found is False


class TestNemoCacheAwareCommitLogic:
    def test_silence_promotes_commit_without_redecode(self):
        backend = NemoCacheAwareStreamingASR("test-model")
        backend._hypothesis = "patient takes ramipril"
        needed = int(16000 * 2 * backend.SILENCE_COMMIT_SECONDS)
        backend._silent_audio_bytes = needed
        assert backend._silence_boundary_reached() is True
        backend._promote_commit()
        assert backend.cumulative_text == "patient takes ramipril"
        assert backend.pending_commit is True
        assert backend.consume_pending_commit() == "patient takes ramipril"
        assert backend.pending_commit is False

    def test_eou_boundary_promotes_and_cleans(self):
        backend = NemoCacheAwareStreamingASR("test-model")
        backend._hypothesis = "see you soon <eou>"
        assert backend._consume_eou_boundary() is True
        assert backend._hypothesis == "see you soon"
        backend._promote_commit()
        assert backend.cumulative_text == "see you soon"

    def test_push_auto_starts_session(self, monkeypatch):
        """push_audio auto-initializes the session instead of raising."""
        import clinical_asr.backends as bl
        from unittest.mock import MagicMock, Mock
        import nemo.collections.asr.parts.utils.streaming_utils as su

        mock_buf = MagicMock()
        mock_buf.buffer = None
        monkeypatch.setattr(su, "CacheAwareStreamingAudioBuffer", lambda *a, **k: mock_buf)
        monkeypatch.setattr(bl, "ensure_local_checkpoint", lambda *a, **k: Path("stub.nemo"))

        mock_model = MagicMock()
        mock_model.encoder.get_initial_cache_state.return_value = (None, None, None)
        mock_model.conformer_stream_step = True
        monkeypatch.setattr(bl, "_load_nemo_model", lambda *a, **k: mock_model)

        backend = NemoCacheAwareStreamingASR("test-model")
        result = asyncio.run(backend.push_audio(LOUD))
        assert result is None  # silent stub model → no words

    def test_requires_streaming_model(self, monkeypatch):
        import clinical_asr.backends as bl
        from unittest.mock import MagicMock
        import nemo.collections.asr.parts.utils.streaming_utils as su

        # Reset class-level cached model so _ensure_model actually checks this model
        bl.NemoCacheAwareStreamingASR._model = None
        bl.NemoCacheAwareStreamingASR._model_key = None

        mock_buf = MagicMock()
        mock_buf.buffer = None
        monkeypatch.setattr(su, "CacheAwareStreamingAudioBuffer", lambda *a, **k: mock_buf)
        monkeypatch.setattr(bl, "ensure_local_checkpoint", lambda *a, **k: Path("stub.nemo"))

        # Plain class — no auto-created attributes like Mock does
        class FakeModel:
            def __init__(self):
                import unittest.mock as um
                self.encoder = um.MagicMock()
                self.encoder.get_initial_cache_state.return_value = (None, None, None)

        monkeypatch.setattr(bl, "_load_nemo_model", lambda *a, **k: FakeModel())

        backend = NemoCacheAwareStreamingASR("nvidia/offline-only-checkpoint")
        with pytest.raises(RuntimeError, match="conformer_stream_step"):
            asyncio.run(backend.start_session())


class TestEnsureLocalCheckpoint:
    def test_explicit_existing_path_wins(self, tmp_path):
        weight = tmp_path / "custom.nemo"
        weight.write_bytes(b"fake")
        assert ensure_local_checkpoint("nvidia/anything", str(weight), tmp_path) == weight

    def test_reuses_slug_dir_checkpoint_without_download(self, tmp_path, monkeypatch):
        import clinical_asr.backends as backends

        slug_dir = tmp_path / "nemotron-speech-streaming-en-0.6b"
        slug_dir.mkdir()
        weight = slug_dir / "model.nemo"
        weight.write_bytes(b"fake")

        def _fail(*a, **k):
            raise AssertionError("must not download")

        monkeypatch.setattr(backends, "_download_checkpoint", _fail)
        resolved = ensure_local_checkpoint(
            "nvidia/nemotron-speech-streaming-en-0.6b", "", tmp_path
        )
        assert resolved == weight

    def test_finds_hand_placed_layout(self, tmp_path, monkeypatch):
        import clinical_asr.backends as backends

        placed = tmp_path / "parakeet"
        placed.mkdir()
        weight = placed / "parakeet-tdt-0.6b-v3.nemo"
        weight.write_bytes(b"fake")

        def _fail(*a, **k):
            raise AssertionError("must not download")

        monkeypatch.setattr(backends, "_download_checkpoint", _fail)
        resolved = ensure_local_checkpoint(
            "nvidia/parakeet-tdt-0.6b-v3", "", tmp_path
        )
        assert resolved == weight

    def test_downloads_missing_weight(self, tmp_path, monkeypatch):
        import clinical_asr.backends as backends

        def _fake_download(repo_id, local_dir):
            assert repo_id == "nvidia/parakeet_realtime_eou_120m-v1"
            local_dir.mkdir(parents=True, exist_ok=True)
            weight = local_dir / "parakeet_realtime_eou_120m-v1.nemo"
            weight.write_bytes(b"fake")
            return weight

        monkeypatch.setattr(backends, "_download_checkpoint", _fake_download)
        resolved = ensure_local_checkpoint(
            "nvidia/parakeet_realtime_eou_120m-v1", "", tmp_path
        )
        assert resolved.name == "parakeet_realtime_eou_120m-v1.nemo"

    def test_missing_explicit_path_without_repo_id_errors(self, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            ensure_local_checkpoint("", str(tmp_path / "absent.nemo"), tmp_path)


class TestVadSegmentedOffline:
    def test_speech_then_silence_decodes_once(self, stub_offline_model):
        backend = VadSegmentedOfflineASR("test", "", "cpu")
        asyncio.run(backend.start_session())
        for _ in range(6):  # 600 ms of speech
            assert asyncio.run(backend.push_audio(LOUD)) is None
        for _ in range(6):  # 600 ms of silence triggers the cut
            asyncio.run(backend.push_audio(SILENT))
        assert stub_offline_model.calls == 1
        assert backend.cumulative_text == "hello world"
        assert backend.pending_commit is True

    def test_second_segment_appends_without_repeating_first(
        self, stub_offline_model
    ):
        stub_offline_model.texts = ["first", "second"]
        backend = VadSegmentedOfflineASR("test", "", "cpu")
        asyncio.run(backend.start_session())
        for _ in range(6):
            asyncio.run(backend.push_audio(LOUD))
        for _ in range(6):
            asyncio.run(backend.push_audio(SILENT))
        for _ in range(6):
            asyncio.run(backend.push_audio(LOUD))
        for _ in range(6):
            asyncio.run(backend.push_audio(SILENT))
        assert stub_offline_model.calls == 2
        assert backend.cumulative_text == "first second"

    def test_finalize_decodes_remainder(self, stub_offline_model):
        backend = VadSegmentedOfflineASR("test", "", "cpu")
        asyncio.run(backend.start_session())
        for _ in range(6):
            asyncio.run(backend.push_audio(LOUD))
        result = asyncio.run(backend.finalize())
        assert stub_offline_model.calls == 1
        assert result == "hello world"

    def test_pure_silence_finalizes_empty(self, stub_offline_model):
        backend = VadSegmentedOfflineASR("test", "", "cpu")
        asyncio.run(backend.start_session())
        for _ in range(6):
            asyncio.run(backend.push_audio(SILENT))
        assert asyncio.run(backend.finalize()) == ""
        assert stub_offline_model.calls == 0

    def test_reset_clears_state(self, stub_offline_model):
        backend = VadSegmentedOfflineASR("test", "", "cpu")
        asyncio.run(backend.start_session())
        asyncio.run(backend.push_audio(LOUD))
        asyncio.run(backend.reset())
        assert backend.cumulative_text == ""
        assert len(backend._segment) == 0


class TestMockStreamingASR:
    """Verify mock backend still works for tests."""

    def test_mock_start_session(self):
        m = MockStreamingASR()
        asyncio.run(m.start_session())
        assert m.text == ""
        assert m.audio_bytes == 0

    def test_mock_push_audio_accumulates(self):
        m = MockStreamingASR()
        asyncio.run(m.push_audio(b"\x00\x01\x02"))
        assert m.audio_bytes == 3

    def test_mock_finalize_returns_text(self):
        m = MockStreamingASR()
        asyncio.run(m.push_demo_text("Patient takes ram a pro"))
        result = asyncio.run(m.finalize())
        assert result == "Patient takes ram a pro"
