# Iterative Validation Guide: Testing the Tests

**Critical**: Do NOT establish a baseline until completing all validation runs!

## The Problem: Unknown Unknowns

We've implemented quality metrics (WER, MCD, SNR, THD), but **we don't know if our implementation is correct**. Before trusting these metrics for regression detection, we must validate them against known cases.

### "Who Watches the Watchmen?"

- Is our WER calculation correct?
- Is our MCD formula implemented properly?
- Are our spectral features computed correctly?
- Do the metrics behave as expected on real TTS audio?

**The risk**: Establishing a baseline with incorrect metrics means we'll be comparing future runs against a broken reference.

## The Solution: Three-Run Validation Process

### Run 0: Meta-Validation (Test the Metrics Themselves)

**Goal**: Verify quality metrics implementation on synthetic audio with known properties

```bash
cd validation
python validate_metrics.py
```

**What it tests**:
- ✅ MCD(identical audio) ≈ 0 dB
- ✅ MCD(very different audio) >> 10 dB
- ✅ SNR(clean signal) > 50 dB
- ✅ SNR(noisy signal) < 15 dB
- ✅ THD(pure tone) < 5%
- ✅ WER calculation correct (using jiwer directly)
- ✅ Spectral features in reasonable ranges
- ✅ Metrics stable across multiple runs
- ✅ Amplitude/RMS calculations correct

**Exit criteria**: ALL tests must pass before proceeding to Run 1

**If tests fail**: Fix quality_metrics.py implementation, then re-run

---

### Run 1: Initial Sanity Check

**Goal**: Verify metrics produce reasonable values on real TTS audio

```bash
# Generate Rust TTS audio
./target/release/test-tts \
  --model-dir models/kyutai-pocket-ios \
  --text "Hello, this is a test." \
  --output validation/run1_rust.wav

# Run quality metrics
python validation/quality_metrics.py \
  --audio validation/run1_rust.wav \
  --text "Hello, this is a test." \
  --whisper-model base \
  --output-json validation/run1_results.json
```

**What to check**:

1. **No errors or NaN values**
   - All metrics should compute successfully
   - No NaN, Inf, or obviously wrong values

2. **Values in reasonable ranges**:
   - WER: Should be low (<20%) - Whisper should understand the speech
   - SNR: Should be decent (>30 dB) - TTS shouldn't be noisy
   - THD: Should be low (<5%) - TTS shouldn't be distorted
   - Amplitude: Should be audible (0.1-1.0)
   - RMS: Should match amplitude roughly

3. **Listen to the audio**:
   - Is it intelligible?
   - Does WER match what you hear?
   - Are there artifacts that SNR/THD should catch?

**Red flags**:
- ❌ WER >50% - Whisper can't understand it (metric or TTS broken)
- ❌ SNR <20 dB - Implies very noisy audio (metric or TTS broken)
- ❌ THD >10% - Implies heavy distortion (metric or TTS broken)
- ❌ Amplitude <0.01 or >1.0 - Unusual output level

**If red flags appear**: Debug quality_metrics.py or TTS implementation

---

### Run 2: Cross-Validation (Rust vs Python with Deterministic Latents)

**Goal**: Validate MCD works correctly by comparing Rust vs Python with same latents

**Background**: With deterministic latents, Rust and Python should produce very similar audio (MCD <2 dB). This validates both:
- MCD implementation is correct
- Rust decoder matches Python decoder

```bash
# Generate deterministic latents with Python
cd validation
python generate_reference_audio.py \
  --text "Hello, this is a test." \
  --output run2_python.wav \
  --export-latents run2_latents.npy

# Decode same latents with Rust (using decode_latents API)
cd ..
./target/release/test-tts \
  --model-dir models/kyutai-pocket-ios \
  --decode-latents validation/run2_latents.npy \
  --output validation/run2_rust.wav

# Compare with MCD
cd validation
python quality_metrics.py \
  --audio run2_rust.wav \
  --reference run2_python.wav \
  --output-json run2_results.json
```

**What to check**:

1. **MCD should be very low** (<2 dB)
   - Same latents → same waveform → low MCD
   - If MCD >5 dB, either MCD implementation or decoder has issues

2. **Waveform correlation should be high** (>0.95)
   - With deterministic latents, correlation is meaningful again
   - If correlation low but MCD low, MCD is working but correlation calc is off

3. **Listen to both audios**:
   - Should sound nearly identical
   - Any perceptible difference suggests decoder divergence

**Red flags**:
- ❌ MCD >5 dB with deterministic latents - Decoder divergence or MCD broken
- ❌ Correlation <0.90 with deterministic latents - Decoder issue
- ❌ Audible differences - Decoder issue

**If red flags appear**: Debug Mimi decoder or MCD implementation

---

### Run 3: Stability Check (Multiple Runs with Random Noise)

**Goal**: Verify metrics are stable and production mode (random noise) works

**Background**: With random noise, each run produces different audio but similar quality. Metrics should be stable within reasonable variance.

```bash
# Run 3 times with random noise
for i in 1 2 3; do
  ./target/release/test-tts \
    --model-dir models/kyutai-pocket-ios \
    --text "Hello, this is a test." \
    --output validation/run3_${i}.wav

  python validation/quality_metrics.py \
    --audio validation/run3_${i}.wav \
    --text "Hello, this is a test." \
    --whisper-model base \
    --output-json validation/run3_${i}.json
done

# Compare results
python validation/compare_runs.py \
  validation/run3_1.json \
  validation/run3_2.json \
  validation/run3_3.json
```

**What to check**:

1. **WER variance should be low**:
   - WER might vary slightly (e.g., 2%, 3%, 4%)
   - Variance should be <5 percentage points
   - If WER varies wildly, Whisper might be unstable or TTS broken

2. **SNR/THD/Spectral should be similar**:
   - These should be very stable (±5%)
   - Large variance suggests implementation issues

3. **All runs should sound good**:
   - Listen to all 3 outputs
   - All should be intelligible and similar quality
   - If one sounds bad, check for edge cases

**Red flags**:
- ❌ WER varies by >10 percentage points - Unstable TTS or metric
- ❌ SNR varies by >10 dB - Unstable TTS or metric
- ❌ One run sounds much worse - Edge case in TTS

**If red flags appear**: Debug TTS stability or metric implementation

---

### Only After All Runs Pass: Establish Baseline

**Decision point**: Are you confident in the quality metrics?

✅ **All checks passed**:
- Run 0: All meta-validation tests pass
- Run 1: Reasonable values, no red flags
- Run 2: MCD <2 dB with deterministic latents
- Run 3: Stable metrics across 3 runs

**Then and only then**:
```bash
./validation/establish_baseline.sh
```

❌ **Any checks failed**:
- DO NOT establish baseline
- Debug the failing component
- Restart from Run 0

---

## Helper Scripts

### compare_runs.py

Create this to compare stability across runs:

```python
#!/usr/bin/env python3
"""Compare quality metrics across multiple runs."""

import json
import sys
import numpy as np

def compare_runs(json_files):
    results = []
    for f in json_files:
        with open(f) as fp:
            results.append(json.load(fp))

    # Extract WER
    wers = [r.get("wer", {}).get("wer", 0) * 100 for r in results]
    snrs = [r.get("snr", {}).get("snr_db", 0) for r in results]
    thds = [r.get("thd", {}).get("thd_percent", 0) for r in results]

    print(f"WER: {wers} (mean={np.mean(wers):.1f}%, std={np.std(wers):.1f}%)")
    print(f"SNR: {snrs} (mean={np.mean(snrs):.1f} dB, std={np.std(snrs):.1f} dB)")
    print(f"THD: {thds} (mean={np.mean(thds):.2f}%, std={np.std(thds):.2f}%)")

    # Check stability
    wer_stable = np.std(wers) < 5.0
    snr_stable = np.std(snrs) < 5.0
    thd_stable = np.std(thds) < 1.0

    if wer_stable and snr_stable and thd_stable:
        print("\n✅ Metrics are stable across runs")
        return 0
    else:
        print("\n❌ High variance detected - investigate")
        return 1

if __name__ == '__main__':
    sys.exit(compare_runs(sys.argv[1:]))
```

---

## Reference Datasets (Future Enhancement)

To further validate metrics against published values:

### LibriSpeech (Whisper validation)
- Public dataset with ground truth transcriptions
- Run Whisper on samples and compare WER to published benchmarks
- Validates our WER calculation matches Whisper's paper

### VCTK Corpus (MCD validation)
- Multi-speaker speech corpus
- Published MCD values for various TTS systems
- Can validate our MCD implementation

### Published TTS Benchmarks
- Kyutai Pocket TTS paper may include quality metrics
- Compare our measurements to theirs on same test phrases
- Ultimate validation of correctness

---

## Decision Tree

```
Start
  ↓
Run 0: Meta-validation
  ├─ PASS → Run 1
  └─ FAIL → Fix quality_metrics.py, retry Run 0
  ↓
Run 1: Sanity check
  ├─ PASS → Run 2
  └─ FAIL → Debug metrics or TTS, retry Run 1
  ↓
Run 2: Cross-validation
  ├─ PASS → Run 3
  └─ FAIL → Debug decoder or MCD, retry Run 2
  ↓
Run 3: Stability check
  ├─ PASS → Establish baseline ✅
  └─ FAIL → Debug TTS stability, retry Run 3
```

---

## FAQ

**Q: Can I skip Run 0 (meta-validation)?**
A: NO. This is the foundation. If metrics are broken, everything else is meaningless.

**Q: Can I establish baseline after just Run 1?**
A: NO. You need cross-validation (Run 2) and stability (Run 3) to have confidence.

**Q: What if Run 2 fails but Run 1 passed?**
A: Run 1 only tests reasonableness. Run 2 validates correctness via comparison.

**Q: How long does this take?**
A: ~30-60 minutes total if everything passes. Worth it to avoid bad baseline.

**Q: Can I automate this?**
A: Yes, create a script that runs all 3 and checks exit codes. But review output manually first time.

---

## Success Criteria Summary

| Run | What | Pass Criteria |
|-----|------|---------------|
| 0 | Meta-validation | All 10 synthetic tests pass |
| 1 | Sanity check | WER <20%, SNR >30dB, audio sounds good |
| 2 | Cross-validation | MCD <2dB with deterministic latents |
| 3 | Stability | WER variance <5%, SNR variance <5dB across 3 runs |

**Only after ALL pass**: Establish baseline

---

*Created: 2026-01-30*
*Last Updated: 2026-01-30*
