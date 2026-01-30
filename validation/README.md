# Pocket TTS Validation Harness

Validates the Rust/Candle implementation of Kyutai Pocket TTS against the official Python reference implementation.

## Purpose

We're not evaluating if Pocket TTS is good (Kyutai proved that). We're verifying our Rust port produces **the same output** as the Python reference. The reference implementation IS our ground truth.

## Model Setup

Before running validation, download the model files:

```bash
# From project root
python3 scripts/download-model.py
```

This downloads from HuggingFace to `models/kyutai-pocket-ios/`:
- `model.safetensors` (~225MB) - Main model weights
- `tokenizer.model` - SentencePiece tokenizer
- `voices/alba.safetensors` - Voice embedding

## Three-Layer Validation

### Layer 1: Reference Comparison (Ground Truth)
- Latent tensor cosine similarity (threshold: >0.99)
- Audio waveform correlation (threshold: >0.95)
- Sample count match (within tolerance)

### Layer 2: ASR Round-Trip (Intelligibility)
- Whisper transcription of generated audio
- WER comparison to reference
- Catches gross implementation errors

### Layer 3: Signal Health (Sanity)
- No NaN/Inf values
- Amplitude in reasonable range (0.01 - 1.0)
- No significant DC offset (<0.05)

## Quick Start

**Option 1: Automated (Recommended)**

```bash
# From project root - runs everything
./validation/run_tests.sh

# Quick mode (skip ASR round-trip)
./validation/run_tests.sh --quick

# Force rebuild
./validation/run_tests.sh --rebuild
```

**Option 2: Manual Steps**

```bash
# 1. Download model (if not already done)
python3 scripts/download-model.py

# 2. Install Python dependencies
pip install -r validation/requirements.txt

# 3. Generate reference outputs (run once)
python validation/reference_harness.py --with-whisper

# 4. Build Rust harness
cargo build --release --bin test-tts

# 5. Run validation
python validation/validate.py --model-dir models/kyutai-pocket-ios
```

## Files

| File | Purpose |
|------|---------|
| `run_tests.sh` | Master test runner (automated) |
| `validate.py` | Main validation orchestrator |
| `compare_intermediates.py` | Debug tool: compare .npy tensors |
| `reference_harness.py` | Generate Python reference outputs |
| `dump_intermediates.py` | Export Python intermediate tensors |
| `requirements.txt` | Python dependencies |
| `reference_outputs/` | Python-generated ground truth |
| `rust_outputs/` | Rust-generated outputs for comparison |
| `debug_outputs/` | Intermediate tensors for debugging |

## Usage

### Generate Reference Outputs

```bash
# Basic (audio only)
python validation/reference_harness.py

# With Whisper transcription (for WER baseline)
python validation/reference_harness.py --with-whisper

# Force regeneration
python validation/reference_harness.py --force --with-whisper
```

### Run Validation

```bash
# Full validation
python validation/validate.py --model-dir /path/to/model

# Skip ASR (faster)
python validation/validate.py --model-dir /path/to/model --skip-asr

# Use existing Rust outputs
python validation/validate.py --model-dir /path/to/model --skip-rust

# Save JSON report
python validation/validate.py --model-dir /path/to/model --json-report results.json
```

## Pass/Fail Criteria

| Test | Threshold | Rationale |
|------|-----------|-----------|
| Latent cosine similarity | >0.99 | Near-identical representations |
| Audio correlation | >0.95 | Accounts for FP differences |
| WER delta | <5% | Match Python within noise |
| NaN/Inf count | 0 | Implementation correctness |
| Max amplitude | 0.01-1.0 | Audible but not clipped |
| DC offset | <0.05 | No significant bias |

## Example Output

```
Validating phrase: 'Hello, this is a test of the Pocket TTS system.'
Reference: validation/reference_outputs/phrase_00.wav
Rust output: validation/rust_outputs/phrase_00_rust.wav

Layer 1: Reference Comparison...
Layer 2: ASR Round-Trip...
Layer 3: Signal Health...

============================================================
VALIDATION RESULTS
============================================================

✓ Layer 1: Reference Match: PASS
  ✓ Sample count: Rust: 78720, Ref: 78720, Diff: 0
  ✓ Audio correlation: Correlation: 0.9823
  ✓ Latent cosine similarity: Similarity: 0.9967

✓ Layer 2: ASR Round-Trip: PASS
  ✓ Rust WER: WER: 3.2%, Transcription: 'Hello, this is a test...'
  ✓ WER delta vs reference: Rust WER: 3.2%, Ref WER: 2.8%, Delta: 0.4%

✓ Layer 3: Signal Health: PASS
  ✓ No NaN/Inf: NaN: 0, Inf: 0
  ✓ Amplitude range: Max amplitude: 0.7234 (expected 0.01-1.0)
  ✓ DC offset: DC offset: 0.001234
  ✓ RMS level: RMS: 0.1523

============================================================
OVERALL RESULT: PASS
============================================================
```

## CI Integration

For GitHub Actions or similar:

```yaml
- name: Validate Pocket TTS
  run: |
    pip install -r rust/pocket-tts-ios/validation/requirements.txt
    cargo build --release --bin test-tts
    python rust/pocket-tts-ios/validation/validate.py \
      --model-dir models/kyutai-pocket-ios \
      --json-report validation-results.json
```

## Debugging with Intermediate Comparisons

When validation fails, use intermediate tensor comparison to locate the divergence:

```bash
# Export Rust latents
./target/release/test-tts \
  --model-dir models/kyutai-pocket-ios \
  --text "Hello world" \
  --output test.wav \
  --export-latents rust_latents.npy

# Export Python intermediates
python validation/dump_intermediates.py

# Compare specific tensors
python validation/compare_intermediates.py \
  --rust rust_latents.npy \
  --python validation/debug_outputs/flownet_output.npy

# Compare entire directories
python validation/compare_intermediates.py \
  validation/rust_outputs/ \
  validation/reference_outputs/
```

Output shows cosine similarity, RMSE, and pinpoints where the largest differences occur.

## Quality Metrics & Regression Detection

### Why This Exists

**The Problem**: When optimizing ML pipelines, it's easy to regress quality without noticing:
- Small changes accumulate into degraded output
- "Improvements" might actually hurt intelligibility
- Without measurements, you only notice after shipping to users

**The Goal**: Get the last few percentage points of quality by:
- Catching regressions before they merge
- Quantifying improvements objectively
- Ensuring every release maintains or exceeds baseline quality

### Overview

The validation suite includes comprehensive audio quality metrics with automated baseline tracking. This is critical because **waveform correlation is no longer meaningful** with random noise enabled (production mode).

**Key insight**: Different random number generators produce different latent trajectories and different waveforms - both equally valid. We measure **what the audio sounds like**, not **what random numbers were used**.

### Quality Metrics Suite

**Tier 1: Intelligibility** (Most Important)
- **WER (Word Error Rate)** via Whisper ASR
- Target: <5% excellent, <10% acceptable
- RNG-independent, validates actual user experience
- **This is the primary quality metric for TTS**

**Tier 2: Acoustic Quality**
- **MCD (Mel-Cepstral Distortion)** - Spectral similarity via MFCC distance
- Target: <4 dB excellent, <6 dB good (when comparing to reference)
- Validates decoder correctness with deterministic latents

**Tier 3: Signal Health**
- **SNR (Signal-to-Noise Ratio)** - Detects background noise and artifacts
  - Target: >25 dB excellent, >15 dB good (speech-specific thresholds)
  - Note: Speech naturally has high-frequency consonants, so SNR is lower than pure tones
- **THD (Total Harmonic Distortion)** - Detects distortion and clipping
  - Target: <10% excellent, <40% acceptable (speech-specific thresholds)
  - Note: Natural speech has harmonics - this is a feature, not a bug

**Tier 4: Spectral Features**
- Spectral centroid, rolloff, flatness, zero-crossing rate
- Tracks timbre and frequency characteristics over time
- No absolute thresholds - used for relative comparison

### Meta-Validation: Testing the Tests

Before establishing a baseline, we validate that the quality metrics themselves work correctly through a 4-run iterative process:

**Run 0: Meta-Validation** (Test metrics on synthetic audio)
```bash
cd validation
python validate_metrics.py
```
- Tests MCD, SNR, THD on synthetic audio with known properties
- Validates implementation correctness (10 tests must pass)
- **Status**: ✅ Complete (10/10 tests passing)

**Run 1: Sanity Check** (Test metrics on real TTS)
```bash
python quality_metrics.py \
  --audio run1_rust.wav \
  --text "Hello, this is a test." \
  --whisper-model base
```
- Establishes what "good" speech measures as
- Discovers speech-appropriate thresholds (different from pure tones)
- **Status**: ✅ Complete (all metrics in healthy ranges)

**Run 2: Cross-Validation** (Compare Rust vs Python)
- Validates MCD by comparing outputs on same text
- **Status**: 🔄 Next step

**Run 3: Stability Check** (Multiple runs)
- Runs TTS 3 times, verifies metric stability
- **Status**: 🔄 Pending

**Only after all 4 runs pass**: Establish baseline with confidence

See [docs/ITERATIVE_VALIDATION.md](docs/ITERATIVE_VALIDATION.md) for the complete validation process.

### Quick Start: Quality Checks

**Run single quality analysis:**
```bash
cd validation
python quality_metrics.py \
  --audio output.wav \
  --text "Hello, this is a test." \
  --whisper-model base \
  --output-json quality_results.json
```

**Compare two audio files (MCD):**
```bash
python quality_metrics.py \
  --audio rust_output.wav \
  --reference python_output.wav \
  --text "Hello, this is a test." \
  --output-json comparison_results.json
```

**Check for regressions (against baseline):**
```bash
python baseline_tracker.py \
  --check-regression \
  --baseline baselines/baseline_v0.4.1.json \
  --metrics quality_results.json
```

**Establish new baseline:**
```bash
./establish_baseline.sh
```

### Documentation

- **[docs/ITERATIVE_VALIDATION.md](docs/ITERATIVE_VALIDATION.md)** - 4-run validation process (start here!)
- **[docs/QUALITY_METRICS.md](docs/QUALITY_METRICS.md)** - Complete metric definitions and targets
- **[docs/REGRESSION_DETECTION.md](docs/REGRESSION_DETECTION.md)** - Usage guide for baseline tracking
- **[docs/SPEECH_VS_TONE_METRICS.md](docs/SPEECH_VS_TONE_METRICS.md)** - Why speech has different thresholds
- **[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)** - Step-by-step validation guide

### Files (Quality Metrics)

| File | Purpose |
|------|---------|
| `quality_metrics.py` | Core metrics: WER, MCD, SNR, THD, spectral features |
| `baseline_tracker.py` | Baseline storage and regression detection |
| `validate_metrics.py` | Meta-validation: test the metrics on synthetic audio |
| `compare_runs.py` | Stability analysis: compare metrics across multiple runs |
| `compare_waveforms.py` | Waveform analysis with amplitude/RMS comparison |
| `establish_baseline.sh` | Script to establish quality baseline for current version |
| `baselines/` | Stored baseline metrics per version |
| `quality_reports/` | Generated quality reports (JSON) |
| `docs/ITERATIVE_VALIDATION.md` | 4-run validation process guide |
| `docs/QUALITY_METRICS.md` | Metric definitions, formulas, targets |
| `docs/REGRESSION_DETECTION.md` | Baseline tracking usage guide |
| `docs/SPEECH_VS_TONE_METRICS.md` | Speech vs pure tone threshold analysis |
| `docs/NEXT_STEPS.md` | Step-by-step validation instructions |

### CI Integration

Quality checks run automatically in GitHub Actions (see `.github/workflows/validation.yml`):

**On Pull Requests** (BLOCKING):
```yaml
- name: Generate test audio
  run: ./target/release/test-tts --text "Hello, this is a test." --output test.wav

- name: Run quality metrics
  run: python quality_metrics.py --audio test.wav --text "..." --output-json results.json

- name: Check for regressions
  run: |
    python baseline_tracker.py \
      --check-regression \
      --baseline baselines/baseline_v0.4.1.json \
      --metrics results.json
  # Fails if WER increases >10%, SNR decreases >10dB, etc.
```

**On Main Branch** (AUTO-UPDATE):
```yaml
- name: Update baseline
  run: python baseline_tracker.py --update-baseline --metrics results.json

- name: Commit updated baseline
  run: |
    git add validation/baselines/
    git commit -m "chore: update quality baselines [skip ci]"
    git push
```

**Quality reports** are uploaded as artifacts for every run, even if model is not available.

## STT Models for Round-Trip Testing

The validation uses two speech-to-text models for cross-validation:

1. **openai-whisper** - OpenAI's Whisper (reference)
2. **faster-whisper** - CTranslate2 optimized version (faster, same accuracy)

Both are installed via `requirements.txt`. The validation uses the "base" model size for speed.

## Troubleshooting

**"Model directory not found"**
Run `python3 scripts/download-model.py` first.

**"Reference outputs not found"**
Run `python reference_harness.py --with-whisper` first, or use `./run_tests.sh` which handles this.

**"Rust binary not found"**
Run `cargo build --release --bin test-tts` first.

**Layer 1 fails with low correlation (~0)**
Major implementation bug. Use `compare_intermediates.py` to locate where outputs diverge.

**Layer 2 fails with high WER delta**
Audio is being generated but is distorted. Check Mimi decoder and SEANet blocks.

**Layer 3 fails with NaN/Inf**
Numerical instability in model. Check for overflow/underflow in convolutions.
