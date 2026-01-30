# Audio Quality Metrics for TTS Validation

## Overview

This document explains the comprehensive audio quality metrics used for Pocket TTS validation. These metrics provide objective, RNG-independent measurements that complement traditional waveform correlation analysis.

## Why Quality Metrics?

### The Random Noise Problem

With random noise enabled (production mode), Rust and Python use different random number generators, producing different latent trajectories and different final waveforms - even when both are producing correct, high-quality speech.

**Traditional waveform correlation is no longer meaningful** because:
- Different RNG → Different latents → Different waveforms
- Correlation ≈0 is expected and correct
- Low correlation does NOT indicate a problem

### The Solution: Objective Quality Metrics

We measure **what the audio sounds like**, not **what random numbers were used to generate it**:

1. **Intelligibility** - Can ASR understand it?
2. **Acoustic Quality** - Does it sound similar to reference?
3. **Signal Health** - Are there artifacts or distortions?
4. **Spectral Characteristics** - Is the timbre appropriate?

---

## Tier 1: Intelligibility Metrics

### Word Error Rate (WER)

**What it measures**: Speech recognition accuracy via Whisper ASR

**How it works**:
1. Transcribe TTS output with OpenAI Whisper (large-v3)
2. Compare transcription to reference text
3. Compute edit distance (insertions, deletions, substitutions)
4. WER = (errors / total words) × 100%

**Target ranges**:
- **Excellent**: WER <5% - Production quality
- **Acceptable**: WER 5-10% - Minor errors, still usable
- **Investigate**: WER >10% - Significant intelligibility issues

**Example**:
```
Reference:  "Hello this is a test"
Hypothesis: "Hello this is the test"
WER = 1/5 = 20% (one substitution: "a" → "the")
```

**Why it matters**:
- Industry-standard TTS metric
- Independent of RNG differences
- Validates actual user experience

**Limitations**:
- Requires reference text
- ASR model must support target language
- May miss subtle pronunciation differences

---

## Tier 2: Acoustic Quality Metrics

### Mel-Cepstral Distortion (MCD)

**What it measures**: Spectral similarity between two audio signals

**How it works**:
1. Extract MFCC (Mel-Frequency Cepstral Coefficients) from both audios
2. Compute Euclidean distance per frame
3. Average across all frames

**Formula**:
```
MCD = (10 / ln(10)) × sqrt(2 × Σ(mfcc₁ - mfcc₂)²)
```

**Target ranges**:
- **Excellent**: MCD <4 dB - Perceptually indistinguishable
- **Good**: MCD 4-6 dB - Similar, minor differences
- **Investigate**: MCD >6 dB - Noticeable spectral differences

**Why it matters**:
- Measures acoustic similarity
- Sensitive to timbre and frequency content
- Widely used in speech synthesis research

**Limitations**:
- Only meaningful when comparing same content (same text)
- Requires reference audio with deterministic generation
- Doesn't directly measure perceptual quality

**Use case**:
Compare Rust vs Python when using **deterministic latents** (via `decode_latents` API):
- Same latents → Same waveform expected → MCD should be very low (<2 dB)
- Different latents → MCD not meaningful (expect high MCD with random noise)

---

## Tier 3: Signal Health Metrics

### Signal-to-Noise Ratio (SNR)

**What it measures**: Ratio of signal power to noise power

**How it works**:
1. Estimate noise floor from quietest 10% of samples
2. Compute signal power from RMS
3. SNR (dB) = 10 × log₁₀(signal_power / noise_power)

**Target ranges**:
- **Excellent**: SNR >40 dB - Very clean signal
- **Good**: SNR 30-40 dB - Clean speech, minimal noise
- **Investigate**: SNR <30 dB - Noticeable background noise

**Why it matters**:
- Detects noise or artifacts
- Validates audio quality
- Helps catch decoder issues

**Limitations**:
- Estimation-based (no true silence in TTS)
- Sensitive to estimation parameters
- May not detect musical artifacts

### Total Harmonic Distortion (THD)

**What it measures**: Ratio of harmonic power to fundamental frequency power

**How it works**:
1. Compute FFT (Fast Fourier Transform)
2. Identify fundamental frequency (typically 80-300 Hz for speech)
3. Measure power at harmonics (2f, 3f, 4f, 5f)
4. THD (%) = 100 × sqrt(harmonic_power / fundamental_power)

**Target ranges**:
- **Excellent**: THD <1% - Very low distortion
- **Acceptable**: THD 1-3% - Minor harmonic content
- **Investigate**: THD >3% - Significant distortion

**Why it matters**:
- Detects non-linear distortion
- Validates decoder quality
- Catches clipping or quantization issues

**Limitations**:
- Speech is naturally harmonic
- May vary by speaker and phoneme
- Requires fundamental frequency detection

---

## Tier 4: Spectral Features

### Spectral Centroid

**What it measures**: "Center of mass" of spectrum (brightness)

- Higher centroid → Brighter, more high-frequency content
- Lower centroid → Darker, more low-frequency content

**Typical range**: 1000-4000 Hz for speech

### Spectral Rolloff

**What it measures**: Frequency below which 85% of spectral energy lies

- Indicates bandwidth and frequency content
- Typical range: 3000-8000 Hz for speech

### Spectral Flatness

**What it measures**: How noise-like vs tone-like the signal is

- Flatness ≈1.0 → Noise-like (white noise)
- Flatness ≈0.0 → Tone-like (pure tone)
- Speech typically: 0.1-0.3

**Use**: Detect overly noisy or overly tonal artifacts

### Zero-Crossing Rate (ZCR)

**What it measures**: How often signal crosses zero amplitude

- High ZCR → High-frequency content
- Low ZCR → Low-frequency content
- Typical range: 0.05-0.15 for speech

**Use**: Coarse frequency content indicator

---

## Baseline Tracking

### Purpose

Track metrics over time to detect regressions:
- Compare current metrics against baseline
- Detect when quality degrades
- Validate improvements

### Thresholds

| Metric | Warning Threshold | Error Threshold |
|--------|-------------------|-----------------|
| WER | +10% | +20% |
| MCD | +10% | +20% |
| SNR | -10% | -20% |
| THD | +50% | +100% |
| Amplitude | -20% | -40% |
| RMS | -20% | -40% |

### Baseline Format

```json
{
  "version": "v0.4.1",
  "git_commit": "47a1baf",
  "date": "2026-01-30T12:00:00",
  "metrics": {
    "wer": { "wer": 0.023 },
    "mcd": { "mcd": 4.5 },
    "snr": { "snr_db": 42.3 },
    "thd": { "thd_percent": 0.8 },
    "amplitude_max": 0.91,
    "rms": 0.105
  }
}
```

---

## Usage Examples

### Basic Quality Check

```bash
python quality_metrics.py \
  --audio rust_output.wav \
  --text "Hello, this is a test." \
  --reference python_ref.wav
```

### Full Quality Check with Baseline

```bash
./run_quality_check.sh \
  --reference python_ref.wav \
  --rust rust_output.wav \
  --text "Hello, this is a test." \
  --baseline validation/baselines/baseline_v0.4.1.json \
  --check-regression
```

### Save New Baseline

```bash
./run_quality_check.sh \
  --reference python_ref.wav \
  --rust rust_output.wav \
  --text "Hello, this is a test." \
  --save-baseline validation/baselines/baseline_v0.4.2.json
```

---

## Interpretation Guide

### Good Quality Indicators

✅ WER <5%
✅ MCD <6 dB (with deterministic latents)
✅ SNR >40 dB
✅ THD <1%
✅ Amplitude ratio 80-120%
✅ RMS ratio 80-120%

### Warning Signs

⚠️ WER 5-10%
⚠️ MCD 6-8 dB
⚠️ SNR 30-40 dB
⚠️ THD 1-3%

### Failure Indicators

❌ WER >10%
❌ MCD >8 dB (with deterministic latents)
❌ SNR <30 dB
❌ THD >3%
❌ Amplitude ratio <50% or >150%

---

## References

- [OpenAI Whisper ASR](https://openai.com/index/whisper/)
- [Mel-Cepstral Distortion (MCD)](https://arxiv.org/abs/2006.01463)
- [Speech Quality Assessment Methods](https://www.lxt.ai/services/speech-data-evaluation/)
- Research Advisor Report-1 (2026-01-30)

---

*Created: 2026-01-30*
*Last Updated: 2026-01-30*
