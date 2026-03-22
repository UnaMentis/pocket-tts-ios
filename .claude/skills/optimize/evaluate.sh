#!/bin/bash
# Standardized evaluation script for /optimize skill
#
# Runs the full evaluate pipeline:
#   1. Build test-tts if source changed
#   2. Synthesize with noise matching (phrase_00)
#   3. Run correlation analysis
#   4. Run quality metrics (WER, MCD, SNR, THD)
#   5. Run composite scorer
#   6. Output JSON to /tmp/optimize-metrics.json
#   7. Print human-readable summary
#
# Usage:
#   bash evaluate.sh
#   bash evaluate.sh --temperature 0.65 --consistency-steps 2
#
# Accepts the same CLI flags as test-tts for config overrides.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Fixed parameters for reproducibility
NOISE_DIR="validation/reference_outputs/noise"
REFERENCE_DIR="validation/reference_outputs"
SEED=42
MODEL_DIR="./kyutai-pocket-ios"
PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
PHRASE_ID="phrase_00"
PHRASE_TEXT="Hello, this is a test of the Pocket TTS system."

OUTPUT_DIR="/tmp/optimize-eval"
WAV_OUT="${OUTPUT_DIR}/${PHRASE_ID}_rust.wav"
METRICS_JSON="/tmp/optimize-metrics.json"
QUALITY_JSON="${OUTPUT_DIR}/quality_metrics.json"

# Parse config overrides (pass-through to test-tts)
TEMPERATURE="0.7"
TOP_P="0.9"
CONSISTENCY_STEPS="1"
SPEED="1.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top-p) TOP_P="$2"; shift 2 ;;
        --consistency-steps) CONSISTENCY_STEPS="$2"; shift 2 ;;
        --speed) SPEED="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ─── Step 1: Build if needed ───────────────────────────────────────────
if [[ ! -f target/release/test-tts ]] || [[ $(find src -newer target/release/test-tts -name '*.rs' 2>/dev/null | head -1) ]]; then
    echo "BUILD: Compiling release binary..." >&2
    cargo build --release --bin test-tts 2>&1 | tail -5 >&2
fi

# ─── Step 2: Synthesize with noise matching ─────────────────────────────
echo "SYNTH: Generating ${PHRASE_ID}..." >&2
./target/release/test-tts \
    -m "$MODEL_DIR" \
    -t "$PHRASE_TEXT" \
    -o "$WAV_OUT" \
    --noise-dir "$NOISE_DIR" \
    --noise-phrase-id "$PHRASE_ID" \
    --consistency-steps "$CONSISTENCY_STEPS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --speed "$SPEED" \
    --seed "$SEED" \
    >/dev/null 2>&1

REF_WAV="${REFERENCE_DIR}/${PHRASE_ID}.wav"
if [[ ! -f "$REF_WAV" ]]; then
    echo "ERROR: Reference audio not found: $REF_WAV" >&2
    exit 1
fi

# ─── Step 3: Correlation analysis ───────────────────────────────────────
CORR_OUTPUT=$($PYTHON -c "
import numpy as np, json
from scipy.io import wavfile

_, ref = wavfile.read('$REF_WAV')
_, rust = wavfile.read('$WAV_OUT')
if ref.dtype != np.float32: ref = ref.astype(np.float32) / 32768.0
if rust.dtype != np.float32: rust = rust.astype(np.float32) / 32768.0

ml = min(len(ref), len(rust))
ref, rust = ref[:ml], rust[:ml]

corr = float(np.corrcoef(ref, rust)[0, 1])

# Per-frame analysis
frame_size = 1920
n_frames = ml // frame_size
frame_corrs = []
for i in range(n_frames):
    s, e = i * frame_size, (i+1) * frame_size
    fc = float(np.corrcoef(ref[s:e], rust[s:e])[0, 1])
    frame_corrs.append(fc)

fc = np.array(frame_corrs)
rms_ref = float(np.sqrt(np.mean(ref**2)))
rms_rust = float(np.sqrt(np.mean(rust**2)))

result = {
    'correlation': corr,
    'samples_ref': len(ref),
    'samples_rust': len(rust),
    'n_frames': n_frames,
    'frame_median_corr': float(np.median(fc)) if len(fc) > 0 else 0.0,
    'frame_mean_corr': float(np.mean(fc)) if len(fc) > 0 else 0.0,
    'frames_above_0_8': int(np.sum(fc > 0.8)) if len(fc) > 0 else 0,
    'frames_above_0_9': int(np.sum(fc > 0.9)) if len(fc) > 0 else 0,
    'rms_ratio': rms_rust / rms_ref if rms_ref > 0 else 0.0,
}
print(json.dumps(result))
")

# ─── Step 4: Quality metrics (WER, MCD, SNR, THD) ──────────────────────
echo "QUALITY: Running metrics..." >&2
$PYTHON validation/quality_metrics.py \
    --audio "$WAV_OUT" \
    --text "$PHRASE_TEXT" \
    --reference "$REF_WAV" \
    --whisper-model base \
    --output-json "$QUALITY_JSON" \
    2>/dev/null || echo '{}' > "$QUALITY_JSON"

# ─── Step 5: Composite score ────────────────────────────────────────────
CORRELATION=$(echo "$CORR_OUTPUT" | $PYTHON -c "import sys, json; print(json.load(sys.stdin)['correlation'])")

SCORE_OUTPUT=$($PYTHON autotuning/scorer.py \
    --metrics-json "$QUALITY_JSON" \
    --correlation "$CORRELATION" \
    --output-json "$METRICS_JSON" 2>/dev/null || echo "Scorer failed" >&2)

# ─── Step 6: Combine into final JSON ────────────────────────────────────
$PYTHON -c "
import json, sys

# Load correlation data
corr_data = json.loads('$CORR_OUTPUT')

# Load quality metrics
try:
    with open('$QUALITY_JSON') as f:
        quality = json.load(f)
except:
    quality = {}

# Load composite score
try:
    with open('$METRICS_JSON') as f:
        score_data = json.load(f)
except:
    score_data = {'composite_score': 0.0, 'components': {}, 'status': 'error'}

# Combine
result = {
    'composite_score': score_data.get('composite_score', 0.0),
    'status': score_data.get('status', 'unknown'),
    'components': score_data.get('components', {}),
    'weights_used': score_data.get('weights_used', {}),
    'correlation': corr_data,
    'quality_metrics': {
        'wer': quality.get('wer', {}).get('wer'),
        'mcd': quality.get('mcd', {}).get('mcd'),
        'snr_db': quality.get('snr', {}).get('snr_db'),
        'thd_percent': quality.get('thd', {}).get('thd_percent'),
    },
    'config': {
        'temperature': float('$TEMPERATURE'),
        'top_p': float('$TOP_P'),
        'consistency_steps': int('$CONSISTENCY_STEPS'),
        'speed': float('$SPEED'),
        'seed': $SEED,
        'phrase_id': '$PHRASE_ID',
    },
}

with open('$METRICS_JSON', 'w') as f:
    json.dump(result, f, indent=2)

# Print human-readable summary
print()
print('=' * 60)
print(f'COMPOSITE SCORE: {result[\"composite_score\"]:.4f} ({result[\"status\"].upper()})')
print('=' * 60)
print()
print('Components:')
for name, value in result['components'].items():
    weight = result['weights_used'].get(name, 0)
    print(f'  {name:25s}: {value:.4f} (weight: {weight:.2f})')
print()
print(f'Correlation: {corr_data[\"correlation\"]:.6f}')
print(f'  Frame median: {corr_data[\"frame_median_corr\"]:.4f}')
print(f'  Frames >0.9:  {corr_data[\"frames_above_0_9\"]}/{corr_data[\"n_frames\"]}')
print()
wer = result['quality_metrics']['wer']
mcd = result['quality_metrics']['mcd']
snr = result['quality_metrics']['snr_db']
thd = result['quality_metrics']['thd_percent']
print(f'WER:  {wer:.4f}' if wer is not None else 'WER:  N/A')
print(f'MCD:  {mcd:.1f} dB' if mcd is not None else 'MCD:  N/A')
print(f'SNR:  {snr:.1f} dB' if snr is not None else 'SNR:  N/A')
print(f'THD:  {thd:.1f}%' if thd is not None else 'THD:  N/A')
print()
print(f'Output: $METRICS_JSON')
"
