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
