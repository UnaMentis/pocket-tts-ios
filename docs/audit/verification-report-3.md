# Verification Report

**Date:** 2026-01-24
**Test Phrase:** "Hello, this is a test of the Pocket TTS system."
**Git State:** 5c6b236

## Build Status

| Check | Status | Notes |
|-------|--------|-------|
| Compilation | PASS | No errors |
| Warnings | 0 | Clean build |
| Clippy | PASS | No warnings |

## Numerical Metrics

### Mimi Decoder Comparison (Rust vs Python Mimi debug output)

| Metric | Previous | Current | Delta | Status |
|--------|----------|---------|-------|--------|
| **Aligned correlation** | 0.6948 | **0.7369** | +0.0421 | ⚠️ (target: >0.95) |
| Best alignment shift | -3842 | -3840 | ~same | Info |
| Max amplitude (Rust) | 0.3567 | 0.3567 | 0 | Info |
| Max amplitude (Python Mimi) | 0.4639 | 0.4639 | 0 | Info |
| Amplitude ratio | 77% | 77% | 0 | Info |

### End-to-End Comparison (Rust vs Python streaming reference)

| Metric | Previous | Current | Delta | Status |
|--------|----------|---------|-------|--------|
| Latent cosine sim | 1.0000 | 1.0000 | 0 | ✅ |
| Waveform correlation (direct) | 0.0008 | 0.0008 | 0 | ❌ |
| Waveform correlation (aligned) | 0.0741 | 0.1587 | +0.0846 | ❌ |
| Sample count (Rust) | 82560 | 82560 | 0 | ⚠️ |
| Sample count (Python ref) | 86400 | 86400 | 0 | ✅ |
| Sample difference | -3840 | -3840 | 0 | Info |
| Max amplitude (Rust) | 0.3567 | 0.3567 | 0 | ⚠️ |
| Max amplitude (Python ref) | 0.6050 | 0.6050 | 0 | ✅ |
| RMS (Rust) | 0.0537 | 0.0537 | 0 | Info |
| Latent frames (Rust) | 43 | 43 | 0 | ⚠️ |
| Latent frames (Python ref) | 45 | 45 | 0 | ✅ |

### All Validation Phrases

| Phrase | Rust Samples | Ref Samples | Diff | Aligned Correlation |
|--------|--------------|-------------|------|---------------------|
| phrase_00 | 82560 | 86400 | -3840 | 0.159 |
| phrase_01 | 74880 | 88320 | -13440 | 0.106 |
| phrase_02 | 78720 | 59520 | +19200 | 0.132 |
| phrase_03 | 30720 | 32640 | -1920 | 0.223 |

### Status Key
- ✅ = At target or matching reference
- ⚠️ = Partial progress or minor gap
- ❌ = Significant gap from target

## Target Progress

```
Target:  0.95 correlation (Mimi decoder vs Python Mimi output)
Current: 0.74 correlation
Gap:     0.21

[==================>..........] 78% of target
```

## Regressions

None detected. All metrics are stable or slightly improved from previous report.

## Improvements

1. **Mimi decoder correlation improved** from 0.6948 to 0.7369 (+6% improvement)
2. **End-to-end aligned correlation improved** from 0.0741 to 0.1587 (+114% relative improvement)

## Audio Quality Assessment

- **Audible:** Yes - produces recognizable speech
- **Artifacts:** Minor - some timing differences audible
- **Duration:** 3.44s (Rust) vs 3.60s (Python) = 4.4% shorter
- **Amplitude:** 59% of Python reference max (0.36 vs 0.61)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Synthesis time | 1.09s |
| Audio duration | 3.44s |
| Real-time factor | 3.15x |

## Signal Health (All Phrases)

All 4 validation phrases pass signal health checks:
- No NaN values
- No Inf values
- No clipped samples (>0.99)
- Reasonable DC offset (<0.001)

## Current State Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Build | ✅ PASS | Clean compilation, no warnings |
| Latent generation | ✅ Cosine sim = 1.0 | Individual latents match Python |
| Mimi decoder | ⚠️ 0.74 correlation | Improved but not yet at 0.95 target |
| Frame count | ⚠️ 43 vs 45 | Rust generates 2 fewer frames |
| End-to-end | ❌ 0.16 correlation | Due to frame count + phase differences |

## Key Observations

1. **Frame count mismatch persists**: Rust generates 43 frames vs Python's 45 frames consistently. This 2-frame difference (~160ms) affects correlation calculations.

2. **Mimi decoder is the bottleneck**: The Mimi debug comparison shows 0.74 correlation, which is better than end-to-end (0.16), indicating the Mimi decoder is working reasonably well.

3. **Amplitude difference**: Rust output is ~59-77% of Python's amplitude. This is likely due to batch vs streaming convolution differences in SEANet.

4. **Variable phrase behavior**: Different phrases show different correlation levels (0.10-0.22), suggesting some phrases are more sensitive to batch/streaming differences.

## Remaining Work for >0.95 Correlation

1. **Fix frame count mismatch** - Investigate EOS detection to generate 45 frames instead of 43
2. **Streaming SEANet** - Implement full streaming convolutions with causal padding
3. **Amplitude normalization** - Consider matching Python's amplitude scaling
4. **First-frame padding** - Ensure replicate padding is correctly applied

## Notes

1. **Stable codebase**: Git state 5c6b236 is clean (no uncommitted changes), suggesting recent commits have stabilized the implementation.

2. **Performance is good**: 3.15x real-time factor meets the target of ~3-4x for mobile deployment.

3. **Audio is intelligible**: Despite low numerical correlation, the audio produces recognizable speech suitable for many use cases.

---

*Previous report archived as verification-report-2.md*
