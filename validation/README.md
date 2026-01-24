# Pocket TTS Validation Harness

Validates the Rust/Candle implementation of Kyutai Pocket TTS against the official Python reference implementation.

## Purpose

We're not evaluating if Pocket TTS is good (Kyutai proved that). We're verifying our Rust port produces **the same output** as the Python reference. The reference implementation IS our ground truth.

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

```bash
cd rust/pocket-tts-ios

# Install Python dependencies
pip install -r validation/requirements.txt

# Generate reference outputs (run once)
python validation/reference_harness.py --with-whisper

# Build Rust harness
cargo build --release --bin test-tts

# Run validation
python validation/validate.py --model-dir /path/to/model
```

## Files

| File | Purpose |
|------|---------|
| `reference_harness.py` | Generate Python reference outputs |
| `validate.py` | Main validation orchestrator |
| `requirements.txt` | Python dependencies |
| `reference_outputs/` | Python-generated ground truth |
| `rust_outputs/` | Rust-generated outputs for comparison |

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

## Troubleshooting

**"Reference outputs not found"**
Run `python reference_harness.py --with-whisper` first.

**"Rust binary not found"**
Run `cargo build --release --bin test-tts` first.

**Layer 1 fails with low correlation**
Implementation bug. Check latent shapes and values at each stage.

**Layer 2 fails with high WER delta**
Audio is being generated but is distorted. Check signal processing stages.

**Layer 3 fails with NaN/Inf**
Numerical instability in model. Check for overflow/underflow in convolutions.
