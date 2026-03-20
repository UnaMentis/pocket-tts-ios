#!/bin/bash
# Reproducible baseline measurement script for Pocket TTS
# Used by /verify skill and the optimize loop
#
# Outputs structured metrics to stdout as KEY=VALUE pairs.
# All runs use identical methodology for direct comparability.
#
# Usage:
#   .claude/skills/verify/run_baseline.sh [--all-phrases] [--output-dir DIR]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration — NEVER change these without updating all docs
NOISE_DIR="validation/reference_outputs/noise"
REFERENCE_DIR="validation/reference_outputs"
SEED=42
CONSISTENCY_STEPS=1
MODEL_DIR="./kyutai-pocket-ios"
PYTHON="${PROJECT_ROOT}/validation/.venv/bin/python3"

OUTPUT_DIR="${2:-/tmp/tts-baseline}"
ALL_PHRASES=false
if [[ "${1:-}" == "--all-phrases" ]]; then
    ALL_PHRASES=true
    OUTPUT_DIR="${3:-/tmp/tts-baseline}"
fi

mkdir -p "$OUTPUT_DIR"

# Phrases and their IDs (must match reference_harness.py TEST_PHRASES)
PHRASES=(
    "Hello, this is a test of the Pocket TTS system."
    "The quick brown fox jumps over the lazy dog."
    "I can speak with different voices and expressions."
    "How are you doing today?"
)
PHRASE_IDS=(phrase_00 phrase_01 phrase_02 phrase_03)

# Build if needed
if [[ ! -f target/release/test-tts ]] || [[ $(find src -newer target/release/test-tts -name '*.rs' 2>/dev/null | head -1) ]]; then
    echo "BUILD: Compiling release binary..." >&2
    cargo build --release --bin test-tts 2>&1 | tail -3 >&2
fi

# Determine which phrases to test
if $ALL_PHRASES; then
    INDICES=(0 1 2 3)
else
    INDICES=(0)  # Just phrase_00 by default
fi

echo "# Pocket TTS Baseline Measurement"
echo "# Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "# Git: $(git describe --always --dirty 2>/dev/null || echo unknown)"
echo "# Noise dir: $NOISE_DIR"
echo "# Seed: $SEED"
echo "# Consistency steps: $CONSISTENCY_STEPS"
echo "#"

OVERALL_CORRS=()

for idx in "${INDICES[@]}"; do
    pid="${PHRASE_IDS[$idx]}"
    phrase="${PHRASES[$idx]}"
    wav_out="$OUTPUT_DIR/${pid}_rust.wav"
    seed=$((SEED + idx))

    echo "# --- $pid ---"

    # Generate audio with noise override (suppress all output)
    ./target/release/test-tts \
        -m "$MODEL_DIR" \
        -t "$phrase" \
        -o "$wav_out" \
        --noise-dir "$NOISE_DIR" \
        --noise-phrase-id "$pid" \
        --consistency-steps "$CONSISTENCY_STEPS" \
        --seed "$seed" \
        >/dev/null 2>&1

    ref_wav="$REFERENCE_DIR/${pid}.wav"
    if [[ ! -f "$ref_wav" ]]; then
        echo "${pid}_CORRELATION=N/A"
        echo "${pid}_ERROR=reference_not_found"
        continue
    fi

    # Run correlation analysis
    $PYTHON -c "
import numpy as np
from scipy.io import wavfile

_, ref = wavfile.read('$ref_wav')
_, rust = wavfile.read('$wav_out')
if ref.dtype != np.float32: ref = ref.astype(np.float32) / 32768.0
if rust.dtype != np.float32: rust = rust.astype(np.float32) / 32768.0

ml = min(len(ref), len(rust))
ref, rust = ref[:ml], rust[:ml]

corr = np.corrcoef(ref, rust)[0, 1]

# Per-frame analysis
frame_size = 1920
n_frames = ml // frame_size
frame_corrs = []
for i in range(n_frames):
    s, e = i * frame_size, (i+1) * frame_size
    fc = np.corrcoef(ref[s:e], rust[s:e])[0, 1]
    frame_corrs.append(fc)

fc = np.array(frame_corrs)
rms_ref = np.sqrt(np.mean(ref**2))
rms_rust = np.sqrt(np.mean(rust**2))

print(f'${pid}_CORRELATION={corr:.6f}')
print(f'${pid}_SAMPLES_REF={len(ref)}')
print(f'${pid}_SAMPLES_RUST={len(rust)}')
print(f'${pid}_FRAMES={n_frames}')
print(f'${pid}_FRAME_MEDIAN_CORR={np.median(fc):.6f}')
print(f'${pid}_FRAME_MEAN_CORR={np.mean(fc):.6f}')
print(f'${pid}_FRAME_MIN_CORR={np.min(fc):.6f}')
print(f'${pid}_FRAME_MAX_CORR={np.max(fc):.6f}')
print(f'${pid}_FRAMES_ABOVE_0_8={int(np.sum(fc > 0.8))}')
print(f'${pid}_FRAMES_ABOVE_0_9={int(np.sum(fc > 0.9))}')
print(f'${pid}_RMS_RATIO={rms_rust / rms_ref:.4f}')
"
    # Capture correlation for overall summary
    corr_val=$($PYTHON -c "
import numpy as np; from scipy.io import wavfile
_, r = wavfile.read('$ref_wav'); _, s = wavfile.read('$wav_out')
if r.dtype != np.float32: r = r.astype(np.float32) / 32768.0
if s.dtype != np.float32: s = s.astype(np.float32) / 32768.0
ml = min(len(r), len(s))
print(f'{np.corrcoef(r[:ml], s[:ml])[0, 1]:.6f}')
")
    OVERALL_CORRS+=("$corr_val")
done

# Summary
echo "#"
echo "# --- Summary ---"
if [[ ${#OVERALL_CORRS[@]} -gt 0 ]]; then
    $PYTHON -c "
corrs = [float(c) for c in '${OVERALL_CORRS[*]}'.split()]
import numpy as np
print(f'MEAN_CORRELATION={np.mean(corrs):.6f}')
print(f'MIN_CORRELATION={np.min(corrs):.6f}')
print(f'MAX_CORRELATION={np.max(corrs):.6f}')
print(f'NUM_PHRASES={len(corrs)}')
"
fi
echo "OUTPUT_DIR=$OUTPUT_DIR"
