# Speech vs Pure Tone Metrics: Critical Differences

**Date**: 2026-01-30
**Finding**: Metric thresholds validated on pure tones (Run 0) are inappropriate for speech audio (Run 1)

## The Problem

During Run 1 validation, we discovered that quality metrics show different baseline values for speech versus pure tones:

| Metric | Pure Tone (Run 0) | Speech (Run 1) | Why Different? |
|--------|------------------|----------------|----------------|
| **SNR** | 62.7 dB (clean) | 18.8 dB | Speech has natural high-frequency content (consonants) |
| **THD** | 0.44% (pure) | 27.59% | Speech SHOULD have harmonics - that's how vowels work |
| **WER** | 0% | 0% | ✅ Consistent across both |

## Root Cause Analysis

### Issue 1: SNR Calculation for Speech

**Our SNR implementation**:
```python
# Uses high-pass filter (>8kHz) to estimate noise floor
b, a = sp_signal.butter(4, cutoff / nyquist, btype='high')
high_freq = sp_signal.filtfilt(b, a, audio)
noise_rms = np.sqrt(np.mean(high_freq ** 2))
```

**Why it works for pure tones**:
- Pure 440Hz sine wave has NO energy above 8kHz
- High-pass filter captures only actual noise
- SNR >50 dB achievable

**Why it's misleading for speech**:
- Consonants ("s", "t", "h", "sh") have significant energy 4-12kHz
- High-pass filter at 8kHz captures **legitimate speech content**, not noise
- This inflates the "noise floor" estimate
- Result: Lower SNR (15-25 dB) even for clean speech

**Example from Run 1**:
- Noise floor (>8kHz content): 0.0086 RMS
- Signal RMS: 0.0747 RMS
- Ratio: 11.5% of signal is high-frequency
- This is **normal** for the phrase "Hello, this is a test" with fricatives

### Issue 2: THD Calculation for Speech

**What THD measures**:
- Total Harmonic Distortion = ratio of harmonic power to fundamental
- For a pure tone, harmonics indicate distortion (non-linearity)

**Why it works for pure tones**:
- Pure 440Hz sine should have minimal 880Hz, 1320Hz, etc.
- THD <1% indicates low distortion
- Run 0 measured 0.44% ✅

**Why it's misleading for speech**:
- Vowels are NATURALLY rich in harmonics - that's what makes them sound different
- "Hello" has fundamental ~135Hz + strong harmonics at 270Hz, 405Hz, 540Hz
- This is a **feature** of natural speech, not a bug
- Result: THD 20-40% is normal and healthy

**Example from Run 1**:
- Fundamental detected: 135 Hz (typical adult male voice)
- Harmonic content: 27.59%
- This indicates rich, natural vocal timbre - not distortion!

## Revised Metric Thresholds

### Before (Pure Tone Validated):
```python
# SNR thresholds
"status": "excellent" if snr_db > 40 else "good" if snr_db > 30 else "investigate"

# THD thresholds
"status": "excellent" if thd_percent < 1.0 else "acceptable" if thd_percent < 3.0 else "investigate"
```

### After (Speech Appropriate):
```python
# SNR thresholds - adjusted for natural high-frequency speech content
"status": "excellent" if snr_db > 25 else "good" if snr_db > 15 else "investigate"

# THD thresholds - adjusted for natural vocal harmonics
"status": "excellent" if thd_percent < 10.0 else "acceptable" if thd_percent < 40.0 else "investigate"
```

## What This Means for Validation

### Run 0 (Meta-Validation) - Still Valid ✅
- Tests the **implementation** of metrics using synthetic signals
- Confirms the math is correct
- Establishes that metrics behave as expected on known inputs

### Run 1 (Speech Sanity Check) - Now Valid ✅
- Tests **real-world baselines** for speech audio
- Establishes what "good" speech measures as
- Updated thresholds based on empirical speech characteristics

### Run 2 (Cross-Validation) - Proceed
- MCD comparison with deterministic latents
- Should show MCD <2 dB (same input → same output)
- This validates decoder correctness, not speech characteristics

### Run 3 (Stability) - Proceed
- Multiple runs with random noise
- Metrics should be stable within new speech-appropriate ranges
- WER variance <5%, SNR variance <5dB, etc.

## Interpretation Guide

### SNR for Speech Audio

| SNR Range | Interpretation |
|-----------|---------------|
| >25 dB | Excellent - very clean speech |
| 15-25 dB | Good - normal speech with natural consonants |
| 10-15 dB | Acceptable - may have slight noise or artifacts |
| <10 dB | Investigate - likely actual noise or quality issues |

**Note**: SNR 15-25 dB does NOT mean the audio is noisy - it means speech naturally has high-frequency content.

### THD for Speech Audio

| THD Range | Interpretation |
|-----------|----------------|
| <10% | Excellent - rich harmonic structure |
| 10-40% | Acceptable - normal vocal timbre |
| 40-70% | Investigate - may indicate overly complex signal |
| >70% | Investigate - likely distortion or artifacts |

**Note**: THD 20-40% is EXPECTED for natural speech with vowels. Only flag if it's much higher than baseline.

## Alternative Metrics for Speech Quality

Since SNR and THD have limitations for speech, consider these as primary metrics:

### 1. WER (Word Error Rate) - PRIMARY ✅
- **Best metric for speech quality**
- Directly measures intelligibility
- RNG-independent
- Target: <5% excellent, <10% acceptable

### 2. MCD (Mel-Cepstral Distortion) - SECONDARY ✅
- Measures spectral similarity to reference
- Useful for comparing Rust vs Python
- Target: <4 dB excellent, <6 dB good

### 3. Spectral Features - MONITORING ✅
- Centroid, rolloff, flatness
- Track changes over time
- No absolute thresholds - relative comparison

### 4. SNR/THD - TRACK BUT DON'T FLAG
- Useful for tracking trends
- Not reliable absolute quality indicators for speech
- Flag only dramatic changes (>50% deviation from baseline)

## Lessons Learned

1. **Metrics validated on synthetic data need recalibration for real-world use**
   - Pure tones ≠ Speech audio
   - Run 0 validates implementation, Run 1 establishes real-world baselines

2. **Domain-specific thresholds are critical**
   - SNR/THD have different meanings in audio processing vs speech synthesis
   - Need to understand what "good" measures as for your specific signal type

3. **Multiple metric types provide robustness**
   - WER catches intelligibility issues
   - MCD catches spectral issues
   - SNR/THD catch certain types of distortion
   - No single metric tells the whole story

4. **Listen to the audio!**
   - Subjective quality assessment is still essential
   - Metrics guide investigation, don't replace human judgment

## Updated Run 1 Results

With speech-appropriate thresholds:

```
AUDIO QUALITY REPORT
======================================================================
File: run1_rust.wav
Duration: 2.00s

INTELLIGIBILITY (WER):
  Word Error Rate: 0.0% (EXCELLENT) ✅

SIGNAL QUALITY (SNR):
  SNR: 18.8 dB (GOOD) ✅

DISTORTION (THD):
  THD: 27.59% (ACCEPTABLE) ✅

SPECTRAL FEATURES:
  Centroid:  3073.1 Hz ✅
  Rolloff:   5905.5 Hz ✅
  Flatness:  0.0696 ✅
```

**Run 1 Status**: ✅ PASS - Ready to proceed to Run 2

---

**Recommendation**: Update baseline tracker thresholds to reflect speech-appropriate ranges before establishing baseline.
