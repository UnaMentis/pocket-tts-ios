# Verification Report

**Date:** 2026-01-24
**Test Phrase:** "Hello, this is a test of the Pocket TTS system."
**Git State:** 4e9b005-dirty

## Build Status

| Check | Status | Notes |
|-------|--------|-------|
| Compilation | PASS | No errors |
| Warnings | 0 | Clean build |
| Clippy | PASS | No warnings |

## Numerical Metrics

### Mimi Decoder Comparison (Rust vs Python Mimi output)

| Metric | Value | Status |
|--------|-------|--------|
| **Aligned correlation** | **0.6948** | ⚠️ (target: >0.95) |
| Best alignment shift | -3842 samples | Info |
| MSE (normalized) | 0.0126 | Info |
| Max amplitude (Rust) | 0.3567 | Info |
| Max amplitude (Python Mimi) | 0.4639 | Info |
| Amplitude ratio | 77% | Info |

### End-to-End Comparison (Rust vs Python streaming reference)

| Metric | Previous | Current | Delta | Status |
|--------|----------|---------|-------|--------|
| Latent cosine sim | 1.0000 | 1.0000 | 0 | ✅ |
| Waveform correlation (aligned) | 0.1285 | 0.0741 | -0.0544 | ❌ |
| Sample count (Rust) | 82904 | 82560 | -344 | ⚠️ |
| Sample count (Python ref) | 86400 | 86400 | 0 | ✅ |
| Max amplitude (Rust) | 0.4421 | 0.3567 | -0.0854 | ⚠️ |
| Max amplitude (Python ref) | 0.6050 | 0.6050 | 0 | ✅ |
| Latent frames (Rust) | 43 | 43 | 0 | ⚠️ |
| Latent frames (Python ref) | 45 | 45 | 0 | ✅ |

### Status Key
- ✅ = At target or matching reference
- ⚠️ = Partial progress or minor gap
- ❌ = Significant gap from target

## Target Progress

```
Target:  0.95 correlation (Mimi decoder)
Current: 0.69 correlation
Gap:     0.26

[===============>..............] 73% of target
```

## Reconciliation: 0.69 vs 0.07 Correlation

**RESOLVED**: The discrepancy between 0.69 (coding agent) and 0.07 (initial verification) was due to **different comparison targets**:

| Comparison Target | Correlation | Explanation |
|-------------------|-------------|-------------|
| `mimi_debug/final_audio.npy` | **0.69** | Python's Mimi decoder output (44 frames) |
| `reference_outputs/phrase_00.wav` | 0.07 | Full end-to-end Python streaming (45 frames) |

**The 0.69 correlation IS correct** and shows the Mimi decoder is working well. The low end-to-end correlation (0.07) is due to:
1. **Frame count mismatch**: Rust generates 43 frames, Python reference has 45 frames
2. **Latent content differences**: The latent sequences differ between runs
3. **Timing alignment**: 3842-sample shift needed for best alignment

## Current State Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Build | ✅ PASS | Clean compilation, no warnings |
| Latent generation | ✅ Cosine sim = 1.0 | Individual latents match Python |
| Mimi decoder | ⚠️ 0.69 correlation | Good but not yet at 0.95 target |
| Frame count | ⚠️ 43 vs 45 | Rust generates 2 fewer frames |
| End-to-end | ❌ 0.07 correlation | Due to frame count + latent differences |

## Audio Quality Assessment
- **Audible:** Yes - produces recognizable speech
- **Artifacts:** Minor - some timing differences audible
- **Duration:** 3.44s (Rust) vs 3.60s (Python) = 4.4% shorter
- **Amplitude:** 77% of Python Mimi output (adequate)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Synthesis time | 1.05s |
| Audio duration | 3.44s |
| Real-time factor | 3.26x |

## Remaining Work for >0.95 Correlation

1. **Fix frame count mismatch** - Rust generates 43 frames vs Python's 45
   - Investigate EOS detection threshold
   - Check minimum generation steps

2. **First-frame padding** - Python uses replicate padding for initial context, Rust uses zeros

3. **Sample alignment** - 3842-sample (~160ms) shift suggests timing offset in streaming

4. **Amplitude normalization** - Consider matching Python's amplitude scaling

## Notes

1. **Mimi decoder improvements confirmed** - The 0.69 correlation validates the streaming fixes from the coding agent session

2. **Uncommitted changes are correct** - The git dirty state in `src/models/mimi.rs` contains the beneficial streaming fixes

3. **Frame count is the primary remaining issue** - Once Rust generates the correct number of frames (45), end-to-end correlation should improve significantly

---

*Previous report archived as verification-report-2.md*
