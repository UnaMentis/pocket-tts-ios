# Regression Detection Guide

## Overview

This guide explains how to use the baseline tracking system to detect quality regressions in the Pocket TTS implementation.

## Key Concepts

### Baselines

A **baseline** is a snapshot of quality metrics from a known-good version:
- Stored as JSON files in `validation/baselines/`
- Contains metrics: WER, MCD, SNR, THD, spectral features
- Includes metadata: version, git commit, timestamp

### Regression Detection

**Regression** = Quality metrics getting worse compared to baseline

The system automatically:
- Compares current metrics against baseline
- Applies configurable thresholds per metric
- Reports warnings and errors
- Fails CI builds on regression (if configured)

---

## Quick Start

### 1. Establish Initial Baseline

First time setup - create a baseline for the current version:

```bash
# Generate reference audio
cd validation
python generate_reference_audio.py --text "Hello, this is a test." --output python_ref.wav

# Build and run Rust TTS
cd ..
cargo build --release --bin test-tts
./target/release/test-tts --text "Hello, this is a test." --output rust_output.wav

# Run quality check and save baseline
cd validation
./run_quality_check.sh \
  --reference python_ref.wav \
  --rust ../rust_output.wav \
  --text "Hello, this is a test." \
  --save-baseline baselines/baseline_v0.4.1.json
```

### 2. Check for Regressions

After making changes, compare against baseline:

```bash
./run_quality_check.sh \
  --reference python_ref.wav \
  --rust ../rust_output.wav \
  --text "Hello, this is a test." \
  --baseline baselines/baseline_v0.4.1.json \
  --check-regression
```

**Exit codes**:
- `0` = No regressions detected
- `1` = Regressions found (fails in CI)

### 3. Update Baseline

When you've validated improvements:

```bash
./run_quality_check.sh \
  --reference python_ref.wav \
  --rust ../rust_output.wav \
  --text "Hello, this is a test." \
  --save-baseline baselines/baseline_v0.4.2.json
```

---

## Workflow Scenarios

### Scenario 1: Local Development

**You're working on a feature and want to ensure no regressions:**

```bash
# Before starting work - establish current baseline
./run_quality_check.sh \
  --reference ref.wav \
  --rust output.wav \
  --text "Test phrase" \
  --save-baseline baselines/baseline_local.json

# After making changes - check for regressions
./run_quality_check.sh \
  --reference ref.wav \
  --rust output.wav \
  --text "Test phrase" \
  --baseline baselines/baseline_local.json \
  --check-regression
```

**Interpretation**:
- ✅ No regressions → Safe to commit
- ⚠️ Warnings → Review changes, consider if acceptable
- ❌ Errors → Fix before committing

### Scenario 2: Pull Request Review

**CI automatically runs quality checks on PRs:**

```yaml
# .github/workflows/validation.yml
- name: Run quality metrics
  run: cd validation && ./run_quality_check.sh ...

- name: Check regressions (BLOCKING)
  run: |
    cd validation
    python baseline_tracker.py \
      --check-regression \
      --baseline baselines/baseline_v0.4.1.json \
      --metrics quality_reports/latest.json
```

**What happens**:
- PR builds and tests code
- Generates audio with new code
- Runs quality metrics
- Compares against `main` branch baseline
- **Blocks merge** if regressions detected

### Scenario 3: Release Validation

**Before releasing a new version:**

```bash
# 1. Run comprehensive quality check
./run_quality_check.sh \
  --reference python_ref.wav \
  --rust rust_output.wav \
  --text "Hello world" \
  --baseline baselines/baseline_v0.4.1.json

# 2. Review results
# - Are there any regressions?
# - Are improvements validated?
# - Is WER still <5%?

# 3. If all good, save as new baseline
./run_quality_check.sh \
  --reference python_ref.wav \
  --rust rust_output.wav \
  --text "Hello world" \
  --save-baseline baselines/baseline_v0.4.2.json

# 4. Commit new baseline
git add validation/baselines/baseline_v0.4.2.json
git commit -m "chore: add quality baseline for v0.4.2"
```

### Scenario 4: Investigating a Regression

**CI reports a regression - what do you do?**

```bash
# 1. Get the full quality report
cat validation/quality_reports/quality_report_TIMESTAMP.json

# 2. Compare specific metrics
python baseline_tracker.py \
  --compare \
  --baseline baselines/baseline_v0.4.1.json \
  --metrics quality_reports/quality_report_TIMESTAMP.json

# 3. Listen to the audio
# Compare Rust output against Python reference

# 4. Check which change caused it
git log --oneline src/models/ src/modules/
git diff baseline_commit..HEAD src/models/mimi.rs

# 5. Fix the issue or accept intentional change
```

---

## Threshold Configuration

Thresholds are defined in `baseline_tracker.py`:

```python
THRESHOLDS = {
    "wer.wer": (10.0, 20.0),           # Warning at +10%, Error at +20%
    "mcd.mcd": (10.0, 20.0),           # Warning at +10%, Error at +20%
    "snr.snr_db": (-10.0, -20.0),      # Warning at -10%, Error at -20%
    "thd.thd_percent": (50.0, 100.0),  # Warning at +50%, Error at +100%
    "amplitude_max": (-20.0, -40.0),   # Warning at -20%, Error at -40%
    "rms": (-20.0, -40.0),             # Warning at -20%, Error at -40%
}
```

**Positive thresholds** = Higher is worse (WER, MCD, THD)
**Negative thresholds** = Lower is worse (SNR, amplitude, RMS)

### Customizing Thresholds

Edit `baseline_tracker.py` to adjust sensitivity:

```python
# More strict (catch smaller changes)
"wer.wer": (5.0, 10.0),  # Warning at +5%, Error at +10%

# More lenient (allow larger changes)
"wer.wer": (20.0, 50.0),  # Warning at +20%, Error at +50%
```

---

## Baseline File Format

```json
{
  "version": "v0.4.1",
  "git_commit": "47a1baf...",
  "date": "2026-01-30T12:34:56",
  "metrics": {
    "wer": {
      "wer": 0.023
    },
    "mcd": {
      "mcd": 4.5
    },
    "snr": {
      "snr_db": 42.3,
      "noise_floor": 0.001234
    },
    "thd": {
      "thd_percent": 0.8
    },
    "amplitude_max": 0.91,
    "rms": 0.105,
    "spectral": {
      "spectral_centroid_hz": 2345.6,
      "spectral_rolloff_hz": 6543.2,
      "spectral_flatness": 0.234,
      "zero_crossing_rate": 0.089
    }
  }
}
```

---

## Interpreting Regression Reports

### Example: No Regressions

```
========================================
BASELINE COMPARISON REPORT
========================================

Baseline Version: v0.4.1
Baseline Date:    2026-01-30T12:00:00
Baseline Commit:  47a1baf

✅ IMPROVEMENTS:
  ✅ wer.wer: 0.0230 → 0.0210 (-8.7%)
  ✅ mcd.mcd: 4.5000 → 4.2000 (-6.7%)

✅ No regressions detected - all metrics within acceptable ranges

SUMMARY:
  Improvements: 2
  Warnings:     0
  Errors:       0
```

**Interpretation**: Safe to merge!

### Example: Warnings

```
========================================
BASELINE COMPARISON REPORT
========================================

⚠️  WARNINGS (Minor Regressions):
  ⚠️  snr.snr_db: 42.3000 → 38.1000 (-9.9%)
  ⚠️  thd.thd_percent: 0.8000 → 1.2000 (+50.0%)

SUMMARY:
  Improvements: 0
  Warnings:     2
  Errors:       0
```

**Interpretation**:
- SNR dropped by 10% (close to warning threshold)
- THD increased by 50% (at warning threshold)
- Review changes - is this expected?
- Listen to audio - is quality still acceptable?

### Example: Errors (Blocking)

```
========================================
BASELINE COMPARISON REPORT
========================================

🚨 ERRORS (Significant Regressions):
  ❌ wer.wer: 0.0230 → 0.0530 (+130.4%)
  ❌ amplitude_max: 0.9100 → 0.4500 (-50.5%)

SUMMARY:
  Improvements: 0
  Warnings:     0
  Errors:       2
```

**Interpretation**:
- WER more than doubled (130% increase) - intelligibility problem!
- Amplitude dropped 50% - major output level issue
- **DO NOT MERGE** - investigate and fix

---

## CI Integration

### Automatic Baseline Updates (Main Branch)

When changes merge to `main`, CI automatically updates the baseline:

```yaml
- name: Update baseline
  if: github.ref == 'refs/heads/main'
  run: |
    cd validation
    python baseline_tracker.py --update-baseline --metrics latest.json

- name: Commit updated baseline
  if: github.ref == 'refs/heads/main'
  run: |
    git config user.name "github-actions[bot]"
    git add validation/baselines/
    git commit -m "chore: update quality baselines [skip ci]"
    git push
```

### Pull Request Checks

On PRs, CI compares against the baseline but doesn't update it:

```yaml
- name: Check for regressions (BLOCKING)
  run: |
    cd validation
    python baseline_tracker.py \
      --check-regression \
      --baseline baselines/baseline_v0.4.1.json \
      --metrics latest.json
```

---

## Best Practices

### 1. Use Consistent Test Phrases

Always use the same test phrases for baseline comparisons:
- "Hello, this is a test."
- "The quick brown fox jumps over the lazy dog."
- Longer phrases for stress testing

### 2. Baseline Multiple Phrases

Create baselines for various scenarios:
- Short phrases (1-2 seconds)
- Medium phrases (3-5 seconds)
- Long phrases (10+ seconds)

### 3. Document Intentional Changes

If you intentionally change something that affects metrics:

```bash
# Save new baseline with descriptive name
./run_quality_check.sh ... \
  --save-baseline baselines/baseline_v0.4.2_improved_snr.json

# Commit with explanation
git commit -m "feat: improve SNR by 5dB through noise gate

This intentionally changes the baseline SNR from 42dB to 47dB.
See docs/improvements/noise_gate.md for details."
```

### 4. Review Before Updating Baselines

Never blindly update baselines - always:
1. Listen to the audio
2. Review the metrics
3. Understand why they changed
4. Validate changes are improvements

### 5. Keep Historical Baselines

Don't delete old baselines - they're useful for:
- Tracking quality trends over time
- Bisecting when regressions were introduced
- Comparing different approaches

```
validation/baselines/
  baseline_v0.4.0.json
  baseline_v0.4.1.json
  baseline_v0.4.2.json
  baseline_latest.json  # Symlink to current
```

---

## Troubleshooting

### "No baseline found"

```bash
# Create initial baseline
./run_quality_check.sh ... \
  --save-baseline baselines/baseline_v0.4.1.json
```

### "Metrics file not found"

```bash
# Ensure quality check ran successfully
./run_quality_check.sh ... \
  --output-dir quality_reports
```

### "Import Error: whisper not found"

```bash
# Install dependencies
cd validation
pip install -r requirements.txt
```

### False Positives (Metric varies naturally)

Some metrics may vary between runs:
- THD can vary by ±20% due to speech content
- Spectral features vary by phoneme

**Solution**: Adjust thresholds to be more lenient for these metrics

---

## References

- [QUALITY_METRICS.md](QUALITY_METRICS.md) - Metric definitions
- Research Advisor Report-1 (2026-01-30) - Validation methodology
- [Quality Plan](../../docs/quality/QUALITY_PLAN.md) - Overall quality strategy

---

*Created: 2026-01-30*
*Last Updated: 2026-01-30*
