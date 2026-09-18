# Offline Clinical Real-Time ASR — Transfer Specification

## Runnable prototype

This repository includes a local FastAPI/WebSocket clinical ASR service. Python 3.12 is required.

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

Download the Parakeet model from [Hugging Face](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and place the model file in `models/parakeet/`. Create the folder first if it does not exist. Then open `http://127.0.0.1:8000/` and use **Load model**, followed by **Start recording**.

The model file is ignored by git and must remain outside source control. The checked-in vocabulary is a small demo fixture; replace it with a licensed, versioned formulary before clinical use.

## 1. Objective

Build the first **ready-to-ship, fully offline clinical speech-to-text model/service** for:

- Irish-accented English speakers
- Clinical conversations
- Medication names and medical terminology
- Very-low / near-real-time transcription
- Local inference with no cloud dependency
- Safe terminology correction without unrestricted LLM rewriting

The first production architecture is:

```text
                 ┌─────────────────────┐
                 │ Base ASR             │
                 │ Parakeet/Nemotron    │
                 └──────────┬──────────┘
                            │
                            ▼
                    Raw transcription
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
       Medical vocabulary         ASR confidence
               │                         │
               └────────────┬────────────┘
                            ▼
                    Terminology resolver
                            │
                            ▼
                   Clinical transcript
```

The system must operate **without sending audio or transcripts to external APIs**.

---

# 2. Primary Success Criteria

The system is not considered ready to ship until it satisfies all of the following:

### Functional

- [ ] Microphone/audio stream can be transcribed continuously.
- [ ] Partial transcripts appear while the speaker is talking.
- [ ] Final transcript is emitted after speech ends.
- [ ] Medical terminology is detected.
- [ ] Medication names can be resolved against a controlled vocabulary.
- [ ] ASR confidence is available or an equivalent confidence mechanism is implemented.
- [ ] Low-confidence clinical terms are flagged instead of silently invented.
- [ ] Entire pipeline works without internet access.

### Performance targets

Initial targets:

| Metric | Target |
|---|---:|
| First partial text | < 500 ms |
| Preferred first partial | < 300 ms |
| P95 partial latency | < 700 ms |
| Finalization after speech | < 1 s |
| Overall WER | < 10% |
| Target WER | < 7% |
| Medication-name accuracy | > 95% |
| Dosage accuracy | > 98% |
| No-network operation | Required |

These are engineering targets, not claims about current model performance.

---

# 3. Important Design Principle

Do **not** solve everything by fine-tuning the ASR model.

The system has multiple error sources:

```text
Audio quality
    ↓
Irish accent
    ↓
ASR decoding
    ↓
Medical terminology
    ↓
Medication names
    ↓
Dosage / frequency
```

Each should be addressed at the appropriate layer.

The first implementation should therefore use:

```text
Streaming ASR
+
Medical vocabulary
+
Confidence
+
Deterministic terminology resolution
+
Clinical validation
```

Fine-tuning comes after establishing the baseline.

---

# 4. Model Strategy

## Primary candidate

### NVIDIA Parakeet Unified 0.6B

Use as the first model to benchmark and deploy if its performance on the target hardware/audio is satisfactory.

Reasons:

- Native streaming architecture
- FastConformer/RNN-T based
- Small enough for local deployment
- Designed for low-latency streaming
- Strong published English accuracy
- Suitable foundation for domain adaptation

## Secondary candidate

### NVIDIA Nemotron ASR Streaming 0.6B

Keep as the main alternative, especially if ultra-low streaming latency is more important than maximum accuracy.

## Accuracy baseline

### Whisper Large-v3 Turbo

Use as an offline/general-purpose accuracy baseline.

Do not assume that Whisper's excellent throughput means equivalent interactive streaming latency.

---

# 5. Hardware Requirement

Before implementation, record the target production machine:

```text
CPU:
GPU:
VRAM:
RAM:
Operating system:
CUDA:
Python:
```

The model must be benchmarked on the **actual deployment hardware**.

Do not optimize against a development workstation and assume production performance.

---

# 6. Audio Pipeline

Normalize incoming audio to:

```text
Sample rate: 16 kHz
Channels: mono
PCM: 16-bit
```

Pipeline:

```text
Microphone
    ↓
Audio capture
    ↓
16 kHz mono PCM
    ↓
Voice Activity Detection
    ↓
Streaming chunks
    ↓
ASR
```

Avoid repeatedly encoding/decoding audio during the real-time path.

---

# 7. Voice Activity Detection

Use VAD to identify:

```text
SPEECH
SILENCE
SPEECH START
SPEECH END
```

Responsibilities:

- Avoid unnecessary inference during silence.
- Detect speech start.
- Detect speech end.
- Trigger final transcript generation.
- Prevent excessive finalization delays.

Candidate:

- Silero VAD
- A VAD already supported by the chosen ASR runtime

The VAD must also run locally.

---

# 8. Streaming ASR Contract

The ASR layer should expose a simple internal interface.

Conceptually:

```python
class StreamingASR:
    def start_session(self):
        ...

    def push_audio(self, audio_chunk):
        ...

    def get_partial(self):
        ...

    def finalize(self):
        ...

    def reset(self):
        ...
```

Do not allow the rest of the application to depend directly on the underlying model framework.

This allows:

```text
Parakeet
   ↓
Nemotron
   ↓
future model
```

without rewriting the terminology layer.

---

# 9. Transcript Event Model

The ASR service should emit events such as:

```json
{
  "type": "partial",
  "text": "patient is currently taking",
  "start_ms": 1200,
  "end_ms": 3100,
  "confidence": 0.94
}
```

Final:

```json
{
  "type": "final",
  "text": "Patient is currently taking ramipril five milligrams once daily.",
  "start_ms": 1200,
  "end_ms": 6200,
  "confidence": 0.96
}
```

Terminology resolver output should preserve the original ASR text.

Example:

```json
{
  "term": "ram a pro",
  "resolved_term": "ramipril",
  "category": "medication",
  "confidence": 0.97,
  "source": "medical_vocabulary",
  "requires_review": false
}
```

---

# 10. Medical Vocabulary

Create a local, versioned medical terminology database.

Minimum categories:

```text
Medication
Brand medication
Generic medication
Medical condition
Procedure
Anatomy
Clinical abbreviation
Dosage unit
Frequency
Route
Laboratory terminology
```

Medication records should contain:

```json
{
  "canonical_name": "ramipril",
  "category": "medication",
  "aliases": [],
  "brand_names": [],
  "dosage_units": ["mg"],
  "common_strengths": [],
  "phonetic_variants": [],
  "asr_confusions": []
}
```

The database must be locally available at runtime.

---

# 11. Medication Vocabulary Source

Do not manually maintain thousands of medications if an authoritative clinical formulary/database is available.

Import an approved source into the local terminology database.

Important:

- Check licensing.
- Version the imported dataset.
- Record the source/version/date.
- Do not dynamically fetch terminology from the internet in production.
- Never depend on a live external medication API.

The runtime artifact should be:

```text
medical_vocabulary.db
```

or an equivalent embedded local index.

---

# 12. Terminology Resolver

The terminology resolver is responsible for mapping potentially incorrect ASR text to known clinical terms.

Example:

```text
ASR:
"patient is taking ram a pro"

             ↓

Candidate generation

             ↓

ramipril
ramelteon
...

             ↓

Context + phonetic similarity
+ ASR confidence
+ vocabulary frequency

             ↓

ramipril
```

The resolver should NOT freely rewrite sentences.

---

# 13. Candidate Generation

Candidate generation can combine:

```text
1. Exact string matching
2. Normalized string matching
3. Fuzzy matching
4. Phonetic matching
5. Known ASR confusion pairs
6. Contextual medication list
7. ASR confidence
```

Example:

```text
"ram a pro"
       ↓
phonetic candidates
       ↓
ramipril
       ↓
patient medication list
       ↓
high confidence
```

---

# 14. Confidence Policy

Never silently correct low-confidence clinical terms.

Use three levels:

```text
HIGH
    ↓
Automatically resolve

MEDIUM
    ↓
Resolve + visually indicate / require confirmation

LOW
    ↓
Do not automatically replace
Mark as uncertain
```

Example:

```text
HIGH:
"Patient takes ramipril."

MEDIUM:
"Patient takes [ramipril?]."

LOW:
"Patient takes [unclear medication]."
```

The exact thresholds must be established empirically during benchmarking.

---

# 15. Contextual Biasing

If the application knows the patient's current medication list, use it as context.

Example:

```text
Patient medications:

Ramipril
Amlodipine
Metformin
Atorvastatin
```

ASR:

```text
"patient takes ram a pro"
```

Resolver:

```text
Candidate: ramipril
Patient context: present
Phonetic similarity: high
ASR confidence: medium/high

→ ramipril
```

This should improve medication recognition without modifying the ASR model.

---

# 16. Do Not Use an LLM as the Primary Correction Mechanism

Avoid:

```text
ASR
 ↓
LLM
 ↓
"corrected medical transcript"
```

This introduces unnecessary hallucination risk.

Prefer:

```text
ASR
 ↓
Deterministic terminology resolver
 ↓
Clinical validation
 ↓
Transcript
```

An LLM can be evaluated later as a separate optional layer for non-authoritative summarization or explanation, but it must not silently rewrite clinical facts.

---

# 17. Irish Accent Adaptation

Irish accent adaptation is a separate problem from terminology.

Collect paired data:

```text
Irish clinical audio
        +
accurate human transcript
```

Target:

```text
10+ hours → useful prototype
20–50 hours → serious adaptation experiment
50+ hours → potentially strong domain adaptation dataset
```

These are practical dataset targets, not guaranteed performance thresholds.

Prioritize:

- Irish speakers
- Actual clinical staff
- Actual clinical environments
- Different ages/voices
- Different speaking speeds
- Different microphones
- Medication-heavy speech
- Numbers and dosages

All data must be appropriately consented, anonymized, and handled according to the applicable clinical/data-protection requirements.

---

# 18. Training Data Format

Use a simple manifest:

```json
{
  "audio_filepath": "audio/001.wav",
  "text": "The patient is taking ramipril five milligrams once daily."
}
```

Recommended dataset split:

```text
Train       80%
Validation  10%
Test        10%
```

The test set must remain untouched during training.

Do not tune the model repeatedly against the same test recordings.

---

# 19. Fine-Tuning Strategy

Do not fine-tune immediately.

First establish:

```text
Base model
    ↓
benchmark
    ↓
medical vocabulary
    ↓
benchmark
    ↓
Irish clinical fine-tuning
    ↓
benchmark
```

Compare every stage.

The purpose is to determine whether errors are primarily caused by:

```text
accent
or
terminology
or
streaming latency
```

---

# 20. Benchmark Dataset

Create a representative local evaluation dataset.

Suggested composition:

```text
20% Irish conversational English
20% Irish clinical speech
20% medication-heavy speech
15% dosage/numbers
10% noisy clinical environment
10% different microphones
5% difficult/rapid speech
```

Prefer real representative data over public clean datasets.

---

# 21. Metrics

Measure:

### ASR accuracy

```text
WER
CER
```

### Clinical accuracy

```text
Medication-name accuracy
Clinical-term accuracy
Dosage accuracy
Frequency accuracy
Unit accuracy
Number accuracy
```

### Streaming performance

```text
Time to first partial
P50 latency
P90 latency
P95 latency
P99 latency
Finalization latency
```

### System performance

```text
Real-Time Factor
GPU utilization
CPU utilization
VRAM
RAM
Throughput
Concurrent sessions
```

---

# 22. Clinical Error Rate

Create a dedicated metric:

```text
Clinical Term Error Rate (CTER)
```

Example:

Reference:

```text
"Patient is taking ramipril 5 mg once daily."
```

Prediction:

```text
"Patient is taking ramelteon 5 mg once daily."
```

Normal WER may not look catastrophic.

Clinically:

```text
Medication error = CRITICAL
```

Therefore the benchmark must report clinical errors independently from WER.

---

# 23. Benchmark Matrix

Every model should be tested at multiple streaming configurations.

Example:

```text
Parakeet
    160 ms
    240 ms
    320 ms
    400 ms
    560 ms

Nemotron
    80 ms
    160 ms
    240 ms
    320 ms
    560 ms

Whisper Turbo
    streaming/window configuration
```

Record:

```text
Latency
WER
Clinical accuracy
VRAM
GPU utilization
```

Do not choose a model based only on WER.

---

# 24. Proprietary Baseline

For comparison, run the same evaluation audio through at least one commercial streaming ASR provider.

Recommended:

```text
Deepgram
+
one existing commercial provider
```

Record:

```text
WER
Medication accuracy
Clinical accuracy
Latency
```

The goal is not necessarily to beat commercial APIs.

The goal is to determine:

```text
How much accuracy are we sacrificing
for offline operation?
```

---

# 25. Model Selection Rule

The final model should be selected using a weighted score.

Initial proposal:

```text
Medication accuracy          30%
Clinical terminology         20%
Dosage/number accuracy       15%
Overall WER                  15%
P95 first-partial latency    10%
P95 finalization latency     5%
Resource efficiency           5%
```

The weights can be changed after collecting real application requirements.

---

# 26. Offline Deployment

Production must not require:

```text
Internet
Cloud API
External database
Remote authentication
External model download
```

Package all runtime artifacts:

```text
clinical-asr/
├── models/
│   └── parakeet/
├── vocabulary/
│   └── medical_vocabulary.db
├── vad/
├── config/
├── runtime/
├── benchmark/
└── service/
```

The production machine should be capable of being disconnected from the network and still performing transcription.

---

# 27. Service Architecture

Recommended:

```text
                 ┌───────────────────┐
                 │ Client/UI         │
                 └─────────┬─────────┘
                           │
                       WebSocket
                           │
                           ▼
                 ┌───────────────────┐
                 │ ASR Service       │
                 ├───────────────────┤
                 │ Audio buffer      │
                 │ VAD               │
                 │ Streaming ASR     │
                 │ Confidence        │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Terminology       │
                 │ Resolver          │
                 ├───────────────────┤
                 │ Vocabulary        │
                 │ Fuzzy matching    │
                 │ Phonetic matching │
                 │ Context ranking   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Clinical          │
                 │ validation        │
                 └─────────┬─────────┘
                           │
                           ▼
                    Final transcript
```

---

# 28. API

The service should expose a streaming endpoint conceptually:

```text
WebSocket:
    /v1/transcribe/stream
```

Events:

```text
session_started
partial_transcript
terminology_update
final_transcript
speech_started
speech_ended
error
session_finished
```

Example:

```json
{
  "type": "partial_transcript",
  "text": "patient is currently taking",
  "confidence": 0.94
}
```

Final:

```json
{
  "type": "final_transcript",
  "text": "Patient is currently taking ramipril five milligrams once daily.",
  "terms": [
    {
      "text": "ramipril",
      "type": "medication",
      "confidence": 0.98,
      "verified": true
    }
  ]
}
```

---

# 29. Logging

For every inference session, record enough information to diagnose failures without unnecessarily storing sensitive audio.

Recommended telemetry:

```text
model_version
vocabulary_version
runtime_version
hardware
audio_duration
first_partial_latency
finalization_latency
RTF
ASR confidence
terminology resolutions
low-confidence terms
```

Avoid storing raw clinical audio by default.

If audio retention is required for model improvement, implement explicit retention/access controls.

---

# 30. Model Versioning

Every production transcript must be traceable to:

```text
ASR model version
Vocabulary version
Resolver version
Configuration version
```

Example:

```text
ASR: parakeet-clinical-0.1.0
Vocabulary: medical-vocab-2026.08
Resolver: resolver-0.1.0
```

This is essential for reproducing clinical transcription behavior.

---

# 31. Safety Rules

The resolver must never:

- Invent a medication.
- Invent a dosage.
- Invent a frequency.
- Replace uncertain text with a plausible clinical term without evidence.
- Use an LLM to silently alter the transcript.
- Treat an ASR guess as verified clinical fact.

When uncertain:

```text
UNCERTAIN
```

is preferable to:

```text
WRONG BUT PLAUSIBLE
```

---

# 32. Development Phases

## Phase 1 — Baseline

Build:

```text
Audio
 ↓
VAD
 ↓
Parakeet
 ↓
Transcript
```

Benchmark latency and WER.

---

## Phase 2 — Streaming

Implement:

```text
WebSocket
+
streaming chunks
+
partial transcripts
+
final transcripts
```

Measure P50/P95/P99 latency.

---

## Phase 3 — Terminology

Implement:

```text
Medical vocabulary
+
fuzzy matching
+
phonetic matching
+
confidence
```

Measure medication accuracy improvement.

---

## Phase 4 — Context

Add:

```text
Patient medication list
+
clinical context
```

Measure improvement in medication recognition.

---

## Phase 5 — Irish adaptation

Collect and clean Irish clinical speech.

Fine-tune/adapt the ASR model.

Benchmark against the untouched base model.

---

## Phase 6 — Production hardening

Add:

```text
Offline packaging
Model versioning
Vocabulary versioning
Logging
Monitoring
Crash recovery
Resource limits
Security
```

---

# 33. Final Production Flow

The target system is:

```text
                  LIVE MICROPHONE
                         │
                         ▼
                  Audio Capture
                         │
                         ▼
                     VAD
                         │
                         ▼
              Streaming Parakeet
                         │
                         ▼
                  Raw Transcript
                         │
               ┌─────────┴─────────┐
               │                   │
               ▼                   ▼
      Medical Vocabulary       ASR Confidence
               │                   │
               └─────────┬─────────┘
                         ▼
                Terminology Resolver
                         │
                         ▼
                 Clinical Validation
                         │
                         ▼
               Clinical Transcript
                         │
               ┌─────────┴─────────┐
               │                   │
               ▼                   ▼
          High confidence     Low confidence
               │                   │
               ▼                   ▼
          Auto accept          Flag/confirm
```

---

# 34. Definition of Done

The first model/service is **ready to ship** only when:

- [ ] Runs fully offline.
- [ ] No runtime internet dependency.
- [ ] Streaming transcription works.
- [ ] P95 first-partial latency meets target.
- [ ] P95 finalization latency meets target.
- [ ] Irish clinical test set evaluated.
- [ ] Medication accuracy measured separately from WER.
- [ ] Dosage/number accuracy measured.
- [ ] Medical vocabulary is versioned.
- [ ] Terminology resolver is deterministic and testable.
- [ ] Low-confidence clinical terms are flagged.
- [ ] No unrestricted LLM correction exists in the authoritative transcript path.
- [ ] Model version is recorded.
- [ ] Vocabulary version is recorded.
- [ ] Regression benchmark exists.
- [ ] Commercial baseline comparison completed.
- [ ] Failure cases are documented.
- [ ] Security/privacy review completed for the deployment environment.

---

# 35. Immediate Next Actions

Start in this order:

```text
1. Record target hardware
2. Download/cache Parakeet Unified 0.6B
3. Verify completely offline inference
4. Implement 16 kHz audio pipeline
5. Implement VAD
6. Implement streaming ASR
7. Measure first-token/P95 latency
8. Build Irish clinical evaluation set
9. Build medication vocabulary
10. Implement terminology resolver
11. Benchmark terminology improvement
12. Compare Nemotron
13. Compare Whisper Turbo
14. Compare commercial baseline
15. Decide whether fine-tuning is justified
16. Fine-tune using Irish clinical speech
17. Regression test
18. Package offline production artifact
```

---

# 36. Key Principle

The goal is **not**:

> "Fine-tune an ASR model until it knows medicine."

The goal is:

> **Build a low-latency offline ASR system whose acoustic model handles Irish clinical speech, while a controlled terminology layer protects medical vocabulary accuracy.**

This separation makes the system easier to benchmark, safer to update, and much easier to ship.
