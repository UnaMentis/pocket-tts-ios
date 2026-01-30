# Run 1 Analysis: Sanity Check Results

**Date**: 2026-01-30
**Audio**: `run1_rust.wav` (2.0s, "Hello, this is a test.")
**Status**: ⚠️ NEEDS REVIEW - Metrics show unexpected values

## Results Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **WER** | 0.0% | <20% | ✅ EXCELLENT |
| **SNR** | 18.8 dB | >30 dB | ❌ INVESTIGATE |
| **THD** | 27.59% | <5% | ❌ INVESTIGATE |
| **Max Amplitude** | 0.553 | 0.1-1.0 | ✅ OK |
| **RMS** | 0.075 | - | ✅ OK |
| **Spectral Centroid** | 3073 Hz | - | ✅ OK |
| **Spectral Rolloff** | 5906 Hz | - | ✅ OK |
| **Spectral Flatness** | 0.070 | - | ✅ OK |

## Critical Question: Are These Targets Appropriate for Speech?

### Issue 1: THD = 27.59% (Expected <5%)

**Context**: THD (Total Harmonic Distortion) was validated in Run 0 using **pure sine waves**.

**For Pure Tones**:
- THD <5% is correct - harmonics indicate distortion
- Our test generated 440Hz pure sine → THD = 0.44% ✅

**For Speech Audio**:
- Speech NATURALLY contains harmonics - that's what makes vowels sound different
- A voice at 135Hz fundamental SHOULD have strong 270Hz, 405Hz, 540Hz harmonics
- This is a **feature** of natural speech, not distortion
- 27.59% harmonic content is **expected and healthy** for speech

**Conclusion**: THD metric as designed is **inappropriate for speech audio**. It measures intentional harmonic structure, not distortion.

**Recommendation**:
1. For speech, we should track THD but **not flag it as regression** unless it changes dramatically
2. Or implement a speech-specific distortion metric (e.g., detecting clipping, artifacts)

### Issue 2: SNR = 18.8 dB (Expected >30 dB)

**Context**: SNR was validated in Run 0 using:
- Clean pure tone → SNR = 62.7 dB ✅
- Noisy pure tone → SNR = 12.4 dB ✅

**For Pure Tones**:
- Most energy concentrated at fundamental frequency
- High-pass filter (>8kHz) captures mostly noise
- SNR >50 dB is achievable

**For Speech Audio**:
- Energy distributed across many frequencies (100Hz - 8kHz)
- Consonants like "s", "t", "h" have significant high-frequency content
- High-pass filter >8kHz captures **legitimate speech content**, not just noise
- This inflates the "noise floor" estimate

**Analysis of Run 1 Values**:
- Signal RMS: 0.0747
- Noise floor (>8kHz): 0.0086
- Ratio: 0.0086 / 0.0747 = 11.5%

The high-frequency content is ~11.5% of total signal - this is **reasonable for speech** which naturally has fricatives, sibilants, and breath sounds.

**Conclusion**: SNR calculation using high-pass filtering is **misleading for speech**. The >8kHz content is legitimate speech, not noise.

**Recommendation**:
1. For speech, accept SNR 15-25 dB as normal (high-frequency consonants are expected)
2. Or implement speech-specific SNR using voice activity detection and silent regions
3. Or track SNR but adjust thresholds for speech (>15 dB acceptable, <10 dB investigate)

## What IS Working Well

### 1. WER = 0% - Perfect Intelligibility ✅
- Whisper transcribed perfectly: "Hello, this is a test."
- This is the MOST IMPORTANT metric for TTS quality
- Indicates excellent articulation and clarity

### 2. Spectral Features Look Healthy ✅
- **Centroid = 3073 Hz**: Typical for adult speech (2000-4000 Hz)
- **Rolloff = 5906 Hz**: Good bandwidth, not muffled
- **Flatness = 0.070**: Low value indicates tonal (not noisy) - good for speech
- **ZCR = 0.148**: Reasonable for speech with consonants

### 3. Amplitude Levels Appropriate ✅
- Max = 0.553: Good headroom, not clipping
- RMS = 0.075: Reasonable dynamic range
- Peak-to-RMS ratio = 7.4:1 (typical for speech)

## Comparison to Meta-Validation (Run 0)

| Test | Run 0 (Synthetic) | Run 1 (Speech) | Expected? |
|------|------------------|---------------|-----------|
| MCD(identical) | 0.04 dB ✅ | N/A | - |
| MCD(different) | 31.48 dB ✅ | N/A | - |
| SNR(clean) | 62.7 dB ✅ | 18.8 dB ❓ | Different signal type |
| SNR(noisy) | 12.4 dB ✅ | N/A | - |
| THD(pure tone) | 0.44% ✅ | 27.59% ❓ | Speech has harmonics |
| WER(identical) | 0% ✅ | 0% ✅ | Perfect match! |

**Key insight**: Run 0 validated the **implementation** of metrics using synthetic signals. Run 1 tests **real speech**, which has fundamentally different characteristics.

## Decision Point: Does Run 1 Pass?

### Arguments for PASS ✅:

1. **WER = 0%** - Core quality metric is perfect
2. **Spectral features** - All look healthy for speech
3. **Amplitude levels** - Appropriate and not clipping
4. **SNR/THD** - May be inappropriate metrics for speech, or need different thresholds

### Arguments for INVESTIGATE ❌:

1. **SNR = 18.8 dB** - Could indicate actual background noise or processing artifacts
2. **THD = 27.59%** - Could indicate actual distortion, not just natural harmonics
3. **Haven't listened to audio** - Need subjective verification
4. **No reference comparison** - Can't compare to known-good TTS

## Recommended Next Steps

### Option A: Proceed to Run 2 (if audio sounds good)
If the audio sounds clear, natural, and intelligible:
1. Accept that SNR/THD need speech-specific thresholds
2. Proceed to Run 2 (cross-validation with deterministic latents)
3. Adjust metric thresholds based on what "good" speech measures as

### Option B: Investigate Further (if audio has issues)
If the audio sounds distorted, noisy, or unnatural:
1. Analyze waveform for clipping, artifacts
2. Compare spectrogram to Python reference
3. Debug TTS or metric implementation
4. Retry Run 1 after fixes

### Option C: Revise Metric Targets
Update `quality_metrics.py` status thresholds for speech:

```python
# In compute_snr():
"status": "excellent" if snr_db > 25 else "good" if snr_db > 15 else "investigate"

# In compute_thd():
# For speech, THD is not a distortion metric - remove or reinterpret
"status": "tracked"  # Don't flag as regression unless >50% change
```

## User Action Required

**Listen to the audio file**:
```bash
afplay validation/run1_rust.wav
```

**Question**: Does it sound clear, natural, and intelligible?

- **YES** → Proceed to Run 2, adjust SNR/THD thresholds
- **NO** → Investigate waveform/spectrogram for actual issues

---

**Status**: ⏸️ PENDING USER VERIFICATION
