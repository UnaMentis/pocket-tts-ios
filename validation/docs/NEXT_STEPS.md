# Next Steps: Validating the Quality Metrics

**Status**: Implementation complete - Ready for validation runs

## What We Built

You now have a comprehensive quality measurement system with:

✅ **Quality Metrics** (WER, MCD, SNR, THD, spectral features)
✅ **Baseline Tracking** (regression detection)
✅ **Meta-Validation** (test the tests themselves)
✅ **Iterative Validation Guide** (3-run process)
✅ **CI Integration** (automatic regression blocking)
✅ **Updated Verification Agent** (uses new metrics)

## Critical: Do NOT Establish Baseline Yet!

As you correctly noted, we need to **validate the validators** first. Here's the process:

## Run 0: Meta-Validation (Testing the Tests)

**Purpose**: Verify quality metrics behave correctly on synthetic audio

```bash
cd validation
python validate_metrics.py
```

**Expected output**: 10/10 tests pass

**What it tests**:
- MCD(identical) ≈ 0 dB ✓
- MCD(different) >> 10 dB ✓
- SNR(clean) > 50 dB ✓
- SNR(noisy) < 15 dB ✓
- THD(pure tone) < 5% ✓
- WER calculation correct ✓
- Spectral features reasonable ✓
- Metrics stable across runs ✓
- Amplitude/RMS calculations ✓

**If ANY tests fail**: Fix `quality_metrics.py` before proceeding

## Run 1: Sanity Check (Real TTS Audio)

**Purpose**: Verify metrics produce reasonable values on actual TTS output

```bash
# Generate audio
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

**Check results**:
- WER <20% ✓ (Whisper understands it)
- SNR >30 dB ✓ (Not noisy)
- THD <5% ✓ (Not distorted)
- Listen to audio - sounds good? ✓

**If red flags**: Debug metrics or TTS

## Run 2: Cross-Validation (Deterministic Latents)

**Purpose**: Validate MCD works by comparing Rust vs Python with same latents

**NOTE**: This requires implementing the `--decode-latents` API if not already available.

```bash
# Generate reference with Python (deterministic)
cd validation
python generate_reference_audio.py \
  --text "Hello, this is a test." \
  --output run2_python.wav \
  --export-latents run2_latents.npy

# Decode same latents with Rust
cd ..
./target/release/test-tts \
  --model-dir models/kyutai-pocket-ios \
  --decode-latents validation/run2_latents.npy \
  --output validation/run2_rust.wav

# Compare
python validation/quality_metrics.py \
  --audio validation/run2_rust.wav \
  --reference validation/run2_python.wav \
  --output-json validation/run2_results.json
```

**Check MCD**:
- MCD <2 dB ✓ (Same latents → same waveform)
- Correlation >0.95 ✓ (Confirms decoder match)

**If MCD >5 dB**: Either MCD implementation or decoder has issues

## Run 3: Stability Check (Multiple Runs)

**Purpose**: Verify metrics are stable with random noise

```bash
# Run 3 times
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

# Compare stability
python validation/compare_runs.py \
  validation/run3_1.json \
  validation/run3_2.json \
  validation/run3_3.json
```

**Check stability**:
- WER variance <5% ✓
- SNR variance <5 dB ✓
- All 3 sound good ✓

**If high variance**: Debug TTS stability

## Only After ALL Pass: Establish Baseline

```bash
./validation/establish_baseline.sh
```

This creates `validation/baselines/baseline_v0.4.1.json`

## Then: Test Regression Detection

Make a small intentional change and verify detection:

```bash
# Modify src/models/mimi.rs slightly (e.g., adjust a parameter)
cargo build --release --bin test-tts

# Generate new audio
./target/release/test-tts \
  --text "Hello, this is a test." \
  --output test_modified.wav

# Check for regression
cd validation
./run_quality_check.sh \
  --reference reference_outputs/phrase_00.wav \
  --rust ../test_modified.wav \
  --text "Hello, this is a test." \
  --baseline baselines/baseline_v0.4.1.json \
  --check-regression
```

**Expected**: Script should detect the change and report regression

## Commit Strategy

After successful validation (all runs pass):

```bash
git add .
git commit -m "feat: add comprehensive quality metrics with meta-validation

- Implement WER, MCD, SNR, THD, and spectral features
- Add baseline tracking with regression detection
- Create meta-validation suite to test metrics themselves
- Implement 3-run iterative validation process
- Update verification agent to use quality metrics
- Integrate with CI for automatic blocking on regressions

IMPORTANT: Baseline NOT yet established - requires validation runs first.
See validation/docs/ITERATIVE_VALIDATION.md for process.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

## Files Ready for Use

All files are staged and ready:

```
M  .github/workflows/validation.yml          # CI integration
M  docs/prompts/verification-agent.md        # Updated agent
M  validation/README.md                      # Updated docs
M  validation/requirements.txt               # Added librosa
A  validation/baseline_tracker.py            # Baseline tracking
A  validation/compare_runs.py                # Stability checker
A  validation/compare_waveforms.py           # Updated comparison
A  validation/docs/IMPLEMENTATION_SUMMARY.md # Tech overview
A  validation/docs/ITERATIVE_VALIDATION.md   # Validation guide
A  validation/docs/QUALITY_METRICS.md        # Metric definitions
A  validation/docs/REGRESSION_DETECTION.md   # Usage guide
A  validation/establish_baseline.sh          # Baseline script
A  validation/quality_metrics.py             # Core metrics
A  validation/run_quality_check.sh           # Orchestration
A  validation/validate_metrics.py            # Meta-validation
```

## Expected Timeline

- **Run 0** (Meta-validation): ~5 minutes
- **Run 1** (Sanity check): ~2 minutes
- **Run 2** (Cross-validation): ~5 minutes (if decode_latents exists)
- **Run 3** (Stability check): ~6 minutes (3 runs × 2 min)
- **Total**: ~20-30 minutes for full validation

## Success Criteria

✅ Run 0: All 10 meta-validation tests pass
✅ Run 1: WER <20%, SNR >30dB, sounds good
✅ Run 2: MCD <2dB with deterministic latents
✅ Run 3: Stable metrics (WER ±5%, SNR ±5dB)

**Then**: Establish baseline with confidence!

## Alternative: Known Reference Values

If you have access to published Kyutai Pocket TTS quality metrics:
- Compare our measurements to theirs on same test phrases
- Validates our implementation matches their measurements
- Ultimate confidence check

## References

- [ITERATIVE_VALIDATION.md](ITERATIVE_VALIDATION.md) - Detailed 3-run process
- [QUALITY_METRICS.md](QUALITY_METRICS.md) - Metric definitions
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- [REGRESSION_DETECTION.md](REGRESSION_DETECTION.md) - Baseline usage

---

**Remember**: The goal is to build confidence in the metrics BEFORE trusting them for regression detection. Take the time to validate properly - it's worth it!
