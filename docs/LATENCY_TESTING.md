# Latency Testing Guide

This document describes how to measure and validate latency for Pocket TTS.

## Reference Baselines

Based on the Kyutai Pocket TTS architecture documentation:

| Metric | Target Value | Description |
|--------|--------------|-------------|
| **TTFA** | ~200ms | Time To First Audio - from text input to first audio chunk |
| **RTF** | 3-4x | Real-Time Factor - audio duration / synthesis time |
| **Per-latent** | ~50ms | Time to generate one latent frame |
| **Model Load** | 0.3-0.5s | Time to load model from disk |

### Device-Specific Targets

| Device | Model Load | RTF | Notes |
|--------|------------|-----|-------|
| iPhone 15 Pro | ~0.3s | 2.5-3.0x | Target device |
| iPhone 14 | ~0.5s | 2.0-2.5x | Acceptable |
| iPad Pro M2 | ~0.2s | 4.0-5.0x | Best performance |

## Running Benchmarks

### Development Environment (Rust)

Use the latency benchmark binary:

```bash
# Quick test (3 iterations)
./scripts/run-latency-bench.sh --quick

# Full streaming benchmark (measures TTFA)
./scripts/run-latency-bench.sh --streaming

# Full sync benchmark
./scripts/run-latency-bench.sh --sync

# Both modes
./scripts/run-latency-bench.sh --all

# Custom configuration
./scripts/run-latency-bench.sh --streaming --iterations 10 --warmup 2
```

Or run directly with cargo:

```bash
# Build release (important for accurate benchmarks)
cargo build --release --bin latency-bench

# Run streaming benchmark
cargo run --release --bin latency-bench -- \
    --model-dir /path/to/kyutai-pocket-ios \
    --streaming \
    --iterations 5 \
    --json ./benchmark-results/latency.json
```

### iOS Demo Client

1. Build and run the PocketTTSDemo app in Xcode
2. Toggle **"Stream"** mode (instead of "Sync")
3. Tap **Synthesize**
4. View TTFA and RTF in the Performance section

The demo shows:
- **TTFA** in milliseconds (green ≤200ms, orange ≤300ms, red >300ms)
- **RTF** (Real-Time Factor)
- **Chunk count** for streaming mode
- Baseline comparison indicator

## Understanding the Metrics

### TTFA (Time To First Audio)

The most important latency metric for user experience. Measures the delay from when text is submitted until audio begins playing.

**Breakdown:**
1. Tokenization (~10ms)
2. Voice embedding lookup (~1ms)
3. First latent generation (~50ms)
4. Mimi decoder warmup (~100ms)
5. First chunk delivery (~40ms buffer)

**Total: ~200ms**

### RTF (Real-Time Factor)

Measures throughput: `audio_duration / synthesis_time`

- RTF of 3.0x means generating 3 seconds of audio per 1 second of compute
- Higher is better
- Below 1.0x means slower than real-time (unusable for streaming)

### Per-Latent Generation

Each latent frame represents 80ms of audio (1920 samples @ 24kHz).
Generation rate should be ~20 latents/second for 1.6x real-time latent throughput.

## Benchmark Output

JSON output includes:

```json
{
  "benchmark": "pocket_tts_latency",
  "mode": "streaming",
  "iterations": 5,
  "summary": {
    "avg_ttfa_ms": 195.5,
    "avg_rtf": 3.2,
    "target_ttfa_ms": 200,
    "target_rtf": 3.5
  },
  "results": [
    {
      "phrase_type": "short",
      "ttfa_ms": 180.5,
      "total_synthesis_ms": 450.2,
      "audio_duration_ms": 1520.0,
      "rtf": 3.38,
      "chunk_count": 2,
      "avg_chunk_latency_ms": 225.1
    }
  ]
}
```

## Troubleshooting

### High TTFA (>300ms)

1. **Check model loading**: Ensure model is pre-loaded before synthesis
2. **Cold vs warm**: First synthesis after model load may be slower
3. **Device thermals**: Check thermal state in iOS demo
4. **Memory pressure**: Low memory causes slowdowns

### Low RTF (<2.0x)

1. **Build mode**: Always benchmark release builds (`cargo build --release`)
2. **Background processes**: Close other apps
3. **Power mode**: Ensure device is not in Low Power Mode
4. **Thermal throttling**: Let device cool down

### Inconsistent Results

1. **Warmup runs**: Use `--warmup 2` to stabilize caches
2. **Multiple iterations**: Use `--iterations 10` for statistical significance
3. **Controlled environment**: Same device state between runs

## CI Integration

Add to CI pipeline:

```yaml
- name: Run latency benchmark
  run: |
    ./scripts/run-latency-bench.sh --streaming --iterations 5 \
      --json ./benchmark-results/ci-latency.json

- name: Check baseline compliance
  run: |
    TTFA=$(jq '.summary.avg_ttfa_ms' ./benchmark-results/ci-latency.json)
    RTF=$(jq '.summary.avg_rtf' ./benchmark-results/ci-latency.json)

    if (( $(echo "$TTFA > 300" | bc -l) )); then
      echo "FAIL: TTFA $TTFA ms exceeds 300ms threshold"
      exit 1
    fi

    if (( $(echo "$RTF < 2.5" | bc -l) )); then
      echo "FAIL: RTF $RTF below 2.5x threshold"
      exit 1
    fi

    echo "PASS: TTFA=${TTFA}ms, RTF=${RTF}x"
```

## Historical Results

Store benchmark results in `benchmark-results/` directory with timestamps for trend analysis.

Compare over time:
```bash
# View recent results
ls -la benchmark-results/

# Compare summaries
for f in benchmark-results/latency_streaming_*.json; do
  echo "$f:"
  jq '.summary' "$f"
done
```
