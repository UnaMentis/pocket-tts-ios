# Verification Report

**Date:** 2026-03-13
**Test Phrase:** "Hello, this is a test of the Pocket TTS system."
**Git State:** v0.4.1-12-gd1a60e9-dirty

## Build Status

| Check | Status | Notes |
|-------|--------|-------|
| Compilation | PASS | Clean release build |
| Clippy | PASS | No warnings with -D warnings |

## Primary Metrics (Correlation-First)

| Metric | Current | Previous | Status |
|--------|---------|----------|--------|
| Audio Correlation (noise-matched) | -0.0129 | 0.012 | Expected low (different RNG) |
| Frame 0 Latent Correlation | 0.7150 | N/A | Moderate transformer fidelity |
| Mean Latent Correlation | 0.2340 | N/A | Autoregressive divergence |
| Frame count (Py/Rs) | 47/46 | 23/23 | Close match (-1 frame) |
| Amplitude ratio | 0.92 | 0.91 | Target: 0.9-1.1 PASS |

### Note on Correlation

This run used `--noise-dir` to load pre-generated noise files for deterministic comparison. However, audio correlation remains low (-0.013), indicating that waveform-level matching is not achievable with the current noise-matching approach. The latent frame 0 correlation of 0.715 shows the transformer produces reasonably similar initial outputs, but autoregressive divergence causes the mean to drop to 0.234 over subsequent frames.

## Diagnostic Metrics

| Metric | Current | Previous | Status |
|--------|---------|----------|--------|
| WER (%) | 0.0 | N/A | PASS - perfect transcription |
| SNR (dB) | -2.7 | N/A | Low (expected with RNG diff) |
| THD (%) | 12.7 | N/A | Moderate harmonic distortion |
| Amplitude ratio | 0.92 | 0.91 | PASS |
| RMS (Rust) | 0.0893 | 0.1047 | Healthy |
| RMS (Python) | 0.0966 | 0.1066 | Healthy |
| Max amp (Rust) | 0.6487 | 0.5732 | Healthy, no clipping |
| Max amp (Python) | 0.5766 | 0.6322 | Healthy |

## Latency Metrics

| Metric | Target | Current | Previous | Status |
|--------|--------|---------|----------|--------|
| TTFA (avg) | <=200ms | 1361.9ms | N/A | FAIL - sync mode only |
| TTFA (short) | <=200ms | ~367ms | N/A | FAIL |
| RTF (avg) | >=3.0x | 3.31x | ~3.7x | PASS |

Note: TTFA is measured in sync mode (full synthesis before first audio). Streaming mode was not tested in this run. The short-phrase TTFA of ~367ms is closer to target but still exceeds the 200ms goal. RTF of 3.31x meets the target range of 3-4x.

## Transformer Divergence Analysis

- **Frame 0 correlation: 0.715** - The first transformer output shows moderate agreement between Python and Rust. This suggests the text encoding and initial conditioning are largely aligned but not identical.
- **Mean correlation: 0.234** - Rapid decay from frame 0 indicates autoregressive divergence. Each subsequent frame compounds small differences, causing the latent trajectory to diverge.
- **Frame count difference: 47 vs 46** - Only 1 frame difference, indicating EOS detection timing is very close between implementations.

The divergence pattern (0.715 -> 0.234 mean) is consistent with the known issue of different random noise between Python and Rust implementations. Even with noise-matched files, small numerical differences in floating-point operations accumulate through the autoregressive loop.

## Signal Health

| Check | Status |
|-------|--------|
| NaN samples | 0 - PASS |
| Inf samples | 0 - PASS |
| Clipped samples (>0.99) | 0 - PASS |
| DC offset | -0.000017 - PASS |

## Audio Quality Assessment

- **Intelligibility:** Perfect - Whisper transcribes exactly "Hello, this is a test of the Pocket TTS system." (0% WER)
- **Amplitude:** Healthy (0.65 max, no clipping)
- **RMS level:** 0.0893 (comparable to reference 0.0966)
- **Duration:** 3.68s Rust vs 3.60s Python (close match)

## Changes Since Previous Report (2026-01-25)

| Aspect | Previous | Current |
|--------|----------|---------|
| Test phrase | "Hello, this is a test." | "Hello, this is a test of the Pocket TTS system." |
| Noise matching | Not used | Used (--noise-dir) |
| Latent export | Not available | Enabled (--export-latents) |
| Frame 0 correlation | N/A | 0.715 (new metric) |
| WER | Not measured | 0.0% (new metric) |
| Latency benchmarks | ~0.5s synthesis | 1361.9ms avg TTFA |
| Sample count (Rust) | 44160 | 88320 |
| Git state | v0.4.0-dirty | v0.4.1-12-gd1a60e9-dirty |

## Recommendations

1. **Noise matching needs improvement** - Despite using `--noise-dir`, audio correlation is still near zero. The noise injection points or scaling may not fully match between implementations.
2. **Frame 0 correlation at 0.715 is a concern** - For identical noise, frame 0 should be closer to 1.0. Investigate potential differences in text tokenization, embedding lookup, or attention computation.
3. **TTFA needs optimization** - 1361.9ms average is far from the 200ms target. Focus on streaming mode TTFA rather than sync mode.
4. **THD at 12.7% warrants investigation** - This is moderately high and may indicate decoder differences.
5. **WER of 0% is excellent** - The speech is perfectly intelligible, which is the most important quality metric.

## Current State Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Build | PASS | Clean compilation, no warnings |
| Clippy | PASS | No linting issues |
| Audio output | PASS | Healthy signal, no artifacts |
| Intelligibility | PASS | 0% WER - perfect transcription |
| Amplitude match | PASS | 0.92 ratio (target 0.9-1.1) |
| Frame count match | PASS | 46 vs 47 frames (-1) |
| Latent correlation | PARTIAL | 0.715 frame 0, 0.234 mean |
| Waveform correlation | FAIL | -0.013 (noise mismatch) |
| Latency (RTF) | PASS | 3.31x (target >=3.0x) |
| Latency (TTFA) | FAIL | 1361.9ms (target <=200ms) |

---

*Previous report archived as verification-report-2.md*
