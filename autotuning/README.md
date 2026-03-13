# TTS Autotuning

Autoresearch-style iterative optimization loop for Pocket TTS quality.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — an autonomous AI research loop that iteratively improves a system by modifying parameters, evaluating, and keeping only improvements.

## Quick Start

```bash
# 1. Establish baseline with default config
python autotuning/autotune.py --phase baseline --model-dir kyutai-pocket-ios

# 2. Sweep individual parameters
python autotuning/autotune.py --phase sweep --param temperature --model-dir kyutai-pocket-ios
python autotuning/autotune.py --phase sweep --param consistency_steps --model-dir kyutai-pocket-ios

# 3. Joint optimization (random perturbation search)
python autotuning/autotune.py --phase optimize --iterations 100 --model-dir kyutai-pocket-ios

# 4. Check results
python autotuning/autotune.py --phase summary --model-dir kyutai-pocket-ios
```

## AI Agent Loop (Phase 4)

For autonomous code-level optimization, start a fresh Claude Code session and paste the contents of `program.md` as the initial prompt. The agent will:

1. Analyze current quality bottlenecks
2. Hypothesize a targeted change
3. Implement and evaluate it
4. Keep improvements, discard regressions
5. Loop indefinitely

## Composite Score

Single scalar metric combining:

| Component | Weight | Source |
|-----------|--------|--------|
| Intelligibility | 40% | WER via Whisper |
| Acoustic similarity | 25% | MCD (MFCC distance) |
| Signal quality | 15% | SNR |
| Waveform correlation | 10% | Pearson correlation |
| Low distortion | 10% | THD |

## Files

| File | Purpose |
|------|---------|
| `autotune.py` | Main loop orchestrator |
| `scorer.py` | Composite quality scoring |
| `program.md` | AI agent instructions |
| `results.tsv` | Experiment log (auto-generated) |
| `baselines/` | Saved baseline measurements |
| `configs/` | Best configuration snapshots |

## Design

See `docs/research/autoresearch-tts-adaptation.md` for the full design document.
