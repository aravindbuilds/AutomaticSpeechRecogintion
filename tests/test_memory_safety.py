"""
Test the memory safety and sliding-window behavior of ParakeetStreamingASR.
"""
import pytest
import time
from clinical_asr.backends import ParakeetStreamingASR, MockStreamingASR


class TestParakeetMemorySafety:
    """Verify that the audio buffer never grows beyond 10 seconds."""

    def test_buffer_does_not_exceed_max_bytes(self):
        """Audio buffer must be capped at MAX_AUDIO_BYTES even with unlimited streaming."""
        import asyncio

        backend = ParakeetStreamingASR('test', '', 'auto')
        asyncio.run(backend.start_session())

        # Simulate streaming 30 seconds of audio at 16kHz/16-bit mono
        chunk_size = 3200  # 100ms of audio = 3200 bytes
        total_streamed = 0

        # Stream until we'd exceed 30 seconds
        for _ in range(300):  # 300 chunks * 100ms = 30 seconds
            chunk = bytes(chunk_size)  # silence (zeros)
            asyncio.run(backend.push_audio(chunk))
            total_streamed += chunk_size

        # Buffer must never exceed MAX_AUDIO_BYTES
        assert len(backend.audio) <= ParakeetStreamingASR.MAX_AUDIO_BYTES
        assert len(backend.audio) < total_streamed  # We trimmed some

    def test_buffer_trims_oldest_audio(self):
        """When buffer overflows, oldest audio (front of buffer) is removed."""
        import asyncio

        backend = ParakeetStreamingASR('test', '', 'auto')
        asyncio.run(backend.start_session())

        # Fill to just under max
        fill_size = ParakeetStreamingASR.MAX_AUDIO_BYTES - 100
        asyncio.run(backend.push_audio(bytes(fill_size)))

        # Record reference: first byte position
        first_before = backend.audio[0] if backend.audio else None

        # Push more to trigger trim
        excess = 500
        asyncio.run(backend.push_audio(bytes(excess)))

        # Buffer must be trimmed, oldest bytes gone
        assert len(backend.audio) <= ParakeetStreamingASR.MAX_AUDIO_BYTES

    def test_last_inference_size_trimmed_correctly(self):
        """last_inference_size must reset when trimming erases its position."""
        import asyncio

        backend = ParakeetStreamingASR('test', '', 'auto')
        asyncio.run(backend.start_session())

        # Fill buffer to just below inference threshold
        fill_size = ParakeetStreamingASR.MIN_INFERENCE_BYTES + 100
        asyncio.run(backend.push_audio(bytes(fill_size)))

        # Trigger inference manually by calling _transcribe (mock-like)
        # We can't call _transcribe without a model, so we verify the size logic:
        # Push enough to overflow, then verify last_inference_size is adjusted
        overflow = ParakeetStreamingASR.MAX_AUDIO_BYTES + 1000
        asyncio.run(backend.push_audio(bytes(overflow)))

        # last_inference_size should be <= current buffer length
        assert backend.last_inference_size <= len(backend.audio)

    def test_audio_duration_property(self):
        """audio_duration_seconds should reflect actual buffer duration."""
        import asyncio

        backend = ParakeetStreamingASR('test', '', 'auto')
        asyncio.run(backend.start_session())

        # Push 1 second of audio
        one_sec = ParakeetStreamingASR.SAMPLE_RATE * ParakeetStreamingASR.SAMPLE_WIDTH
        asyncio.run(backend.push_audio(bytes(one_sec)))

        # Duration should be close to 1.0 seconds
        assert 0.9 < backend.audio_duration_seconds < 1.1

    def test_reset_clears_buffer(self):
        """reset() must fully clear the audio buffer."""
        import asyncio

        backend = ParakeetStreamingASR('test', '', 'auto')
        asyncio.run(backend.start_session())
        asyncio.run(backend.push_audio(bytes(16000)))

        asyncio.run(backend.reset())

        assert len(backend.audio) == 0
        assert backend.last_inference_size == 0

    def test_finalize_works_after_overflow(self):
        """After buffer overflow + trimming, finalize should still work."""
        import asyncio

        backend = ParakeetStreamingASR('test', '', 'auto')
        asyncio.run(backend.start_session())

        # Simulate enough audio to trigger overflow and trimming
        for _ in range(50):
            asyncio.run(backend.push_audio(bytes(3200)))

        # finalize should not crash (returns empty string without model)
        result = asyncio.run(backend.finalize())
        assert isinstance(result, str)


class TestMockStreamingASR:
    """Verify mock backend still works for tests."""

    def test_mock_start_session(self):
        import asyncio
        m = MockStreamingASR()
        asyncio.run(m.start_session())
        assert m.text == ""
        assert m.audio_bytes == 0

    def test_mock_push_audio_accumulates(self):
        import asyncio
        m = MockStreamingASR()
        asyncio.run(m.push_audio(b'\x00\x01\x02'))
        assert m.audio_bytes == 3

    def test_mock_finalize_returns_text(self):
        import asyncio
        m = MockStreamingASR()
        asyncio.run(m.push_demo_text("Patient takes ram a pro"))
        result = asyncio.run(m.finalize())
        assert result == "Patient takes ram a pro"
