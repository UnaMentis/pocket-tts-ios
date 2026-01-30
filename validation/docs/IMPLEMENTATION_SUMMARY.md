# Quality Metrics Implementation Summary

**Date**: 2026-01-30
**Implementation Status**: ✅ Complete

## Overview

Implemented a comprehensive audio quality measurement and regression detection system for Pocket TTS. This addresses the critical need for objective, RNG-independent quality metrics now that random noise is enabled (production mode).

## Problem Statement

With random noise enabled, Rust and Python use different random number generators, producing different latent trajectories and different waveforms. **Traditional waveform correlation is no longer meaningful** (correlation ≈0 is expected and correct).

**The risk**: Without objective quality metrics, we could introduce regressions without knowing it.

## Solution

Implemented three-tier quality measurement suite with baseline tracking:

### Tier 1: Intelligibility (WER via Whisper ASR)
- Industry-standard metric for TTS quality
- RNG-independent (transcribes actual content)
- Target: WER <5% = production quality

### Tier 2: Acoustic Quality (MCD)
- Mel-Cepstral Distortion - measures spectral similarity
- Useful with deterministic latents
- Target: MCD <6 dB = perceptually similar

### Tier 3: Signal Health (SNR, THD, Spectral Features)
- SNR (Signal-to-Noise Ratio) - detects artifacts
- THD (Total Harmonic Distortion) - detects non-linear distortion
- Spectral features (centroid, rolloff, flatness, ZCR) - validates timbre

### Regression Detection
- Baseline storage in JSON format
- Automated comparison with configurable thresholds
- CI integration with automatic baseline updates

## Implementation Details

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `validation/quality_metrics.py` | ~500 | Core quality metrics (WER, MCD, SNR, THD, spectral) |
| `validation/baseline_tracker.py` | ~400 | Baseline storage and regression detection |
| `validation/run_quality_check.sh` | ~150 | Orchestration script |
| `validation/establish_baseline.sh` | ~80 | Helper to establish initial baseline |
| `validation/docs/QUALITY_METRICS.md` | ~600 | Comprehensive metric documentation |
| `validation/docs/REGRESSION_DETECTION.md` | ~500 | Usage guide for baseline tracking |
| `validation/docs/IMPLEMENTATION_SUMMARY.md` | ~300 | This document |

### Files Modified

| File | Changes |
|------|---------|
| `validation/requirements.txt` | Added `librosa` dependency |
| `validation/compare_waveforms.py` | Added `--quality-metrics` flag |
| `validation/README.md` | Added quality metrics section |
| `.github/workflows/validation.yml` | Added quality-metrics job |

### Directories Created

- `validation/baselines/` - Baseline metrics storage
- `validation/quality_reports/` - Generated quality reports
- `validation/docs/` - Documentation

## Metrics Implementation

### WER (Word Error Rate)
- Uses OpenAI Whisper (configurable model size)
- Computes edit distance between transcription and reference text
- Formula: `WER = (S + D + I) / N` where S=substitutions, D=deletions, I=insertions, N=total words

### MCD (Mel-Cepstral Distortion)
- Extracts 13 MFCC coefficients using librosa
- Computes Euclidean distance per frame
- Formula: `MCD = (10/ln(10)) × sqrt(2 × Σ(mfcc₁ - mfcc₂)²)`
- Averaged across all frames

### SNR (Signal-to-Noise Ratio)
- Estimates noise floor from quietest 10% of samples
- Computes signal power from RMS
- Formula: `SNR (dB) = 10 × log₁₀(signal_power / noise_power)`

### THD (Total Harmonic Distortion)
- FFT analysis to find fundamental frequency
- Measures power at harmonics (2f, 3f, 4f, 5f)
- Formula: `THD (%) = 100 × sqrt(harmonic_power / fundamental_power)`

### Spectral Features
- Spectral centroid - "brightness" (center of mass of spectrum)
- Spectral rolloff - frequency below which 85% of energy lies
- Spectral flatness - how noise-like vs tone-like (0=tone, 1=noise)
- Zero-crossing rate - coarse frequency content indicator

## Baseline Tracking

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
    "rms": 0.105,
    "spectral": { ... }
  }
}
```

### Regression Thresholds

| Metric | Warning | Error | Direction |
|--------|---------|-------|-----------|
| WER | +10% | +20% | Higher is worse |
| MCD | +10% | +20% | Higher is worse |
| SNR | -10% | -20% | Lower is worse |
| THD | +50% | +100% | Higher is worse |
| Amplitude | -20% | -40% | Lower is worse |
| RMS | -20% | -40% | Lower is worse |

## CI Integration

### Pull Requests
1. Build Rust binary
2. Generate test audio
3. Run quality metrics
4. Compare against baseline
5. **Block merge** if regressions detected (errors)

### Main Branch
1. All PR steps above
2. **Automatically update baseline** if metrics improved or changed
3. Commit baseline update with `[skip ci]`

### Workflow
```yaml
quality-metrics:
  - Download binary (artifact)
  - Install Python deps
  - Generate test audio
  - Run quality metrics
  - Check regression (PR only) → BLOCKS if errors
  - Update baseline (main only) → AUTO-COMMIT
  - Upload quality report (artifact)
```

## Usage Examples

### Basic Quality Check
```bash
cd validation
python quality_metrics.py \
  --audio rust_output.wav \
  --text "Hello, this is a test." \
  --reference python_ref.wav
```

### Full Quality Check with Regression Detection
```bash
./run_quality_check.sh \
  --reference python_ref.wav \
  --rust rust_output.wav \
  --text "Hello, this is a test." \
  --baseline baselines/baseline_v0.4.1.json \
  --check-regression
```

### Establish Initial Baseline
```bash
./establish_baseline.sh
```

## Testing Plan

### Manual Testing
1. ✅ Run `quality_metrics.py` standalone
2. ✅ Run `baseline_tracker.py` with sample data
3. ✅ Run `run_quality_check.sh` end-to-end
4. ⏳ Establish initial baseline for v0.4.1
5. ⏳ Test regression detection with artificial modification

### CI Testing
1. ⏳ Create PR with model changes
2. ⏳ Verify quality metrics run
3. ⏳ Verify regression detection works
4. ⏳ Verify baseline update on main

## Benefits

### Immediate
- **Objective quality measurement** - No longer relying on correlation
- **Regression prevention** - Catch quality degradation in CI
- **Production confidence** - WER validates real-world intelligibility

### Long-term
- **Quality trends** - Track improvements over time via baselines
- **Release validation** - Verify quality before each release
- **Research insights** - Understand how changes affect audio quality

## Next Steps

### Phase 1: Initial Deployment ✅ Complete
- [x] Implement core metrics
- [x] Implement baseline tracker
- [x] Create orchestration scripts
- [x] Write comprehensive documentation
- [x] Integrate with CI

### Phase 2: Baseline Establishment ⏳ In Progress
- [ ] Generate reference audio with Python
- [ ] Run quality metrics on v0.4.1
- [ ] Establish initial baseline
- [ ] Test regression detection

### Phase 3: Validation & Refinement
- [ ] Create PR with intentional change
- [ ] Verify CI blocks on regression
- [ ] Verify baseline update on main
- [ ] Tune thresholds based on real data

### Phase 4: iOS Device Testing (Future)
- [ ] Run AB test on iPhone 12+
- [ ] Verify >0.95 correlation with deterministic latents
- [ ] Document minimum device requirements
- [ ] Add iOS testing to pre-release checklist

## Technical Decisions

### Why Whisper "base" model?
- Balance between accuracy and speed
- "large-v3" is more accurate but 10x slower
- Can be configured per use case

### Why librosa for audio processing?
- Industry standard for audio analysis
- Comprehensive feature extraction
- Well-documented and maintained

### Why JSON for baselines?
- Human-readable for review
- Git-friendly (easy to diff)
- Easily parsed by CI and tools

### Why automatic baseline updates on main?
- Reduces manual overhead
- Ensures baseline stays current
- Preserves historical baselines in git

## References

- Research Advisor Report-1 (2026-01-30) - Recommended WER, MCD approach
- Verification Report-1 (2026-01-25) - Identified need for RNG-independent metrics
- Quality Plan (docs/quality/QUALITY_PLAN.md) - Overall quality strategy
- [OpenAI Whisper](https://openai.com/index/whisper/)
- [Mel-Cepstral Distortion](https://arxiv.org/abs/2006.01463)

## Maintenance Notes

### Updating Thresholds
Edit `baseline_tracker.py::THRESHOLDS` dictionary:
```python
THRESHOLDS = {
    "wer.wer": (10.0, 20.0),  # Adjust as needed
    ...
}
```

### Adding New Metrics
1. Implement in `quality_metrics.py::analyze_audio()`
2. Add extraction in `baseline_tracker.py::_extract_comparable_metrics()`
3. Add threshold in `baseline_tracker.py::THRESHOLDS`
4. Document in `QUALITY_METRICS.md`

### Baseline Rotation
- Keep at least 3 most recent baselines
- Archive old baselines to `baselines/archive/`
- Update `baseline_latest.json` symlink

---

**Status**: ✅ Implementation Complete - Ready for Baseline Establishment
**Next Action**: Run `./validation/establish_baseline.sh` to create v0.4.1 baseline
