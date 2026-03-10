# Autoresearch-Style Iterative TTS Fine-Tuning

## Background

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) introduces an elegant pattern: an AI agent autonomously optimizes a training script by running a tight loop of **modify → evaluate → keep/discard**. The metric is `val_bpb` (validation bits-per-byte), the budget is 5 minutes per experiment, and git serves as memory — good ideas accumulate on the branch, bad ideas get reset.

A [macOS port](https://github.com/miolini/autoresearch-macos) adapts this for Apple Silicon with MPS, but the core loop is identical.

## The Insight

Our Pocket TTS quality optimization is structurally identical to autoresearch:

| Autoresearch (GPT) | Pocket TTS |
|---|---|
| Modify `train.py` | Modify tuning parameters / model code |
| Train for 5 minutes | Synthesize test phrases + evaluate |
| Measure `val_bpb` | Measure composite quality score |
| `val_bpb` improved → git commit | Score improved → save config |
| `val_bpb` regressed → git reset | Score regressed → discard changes |
| Loop forever | Loop until convergence |

The key properties that make this work:

1. **Clear scalar metric** — We can reduce WER, MCD, SNR, correlation into a single composite score
2. **Fast evaluation** — Synthesis + quality measurement takes seconds, not hours
3. **Well-defined search space** — Temperature, top_p, consistency_steps, speed, plus code-level changes
4. **Deterministic comparison** — Fixed seed enables reproducible A/B comparison
5. **Monotonic improvement** — Each iteration either improves or stays the same

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AUTOTUNING LOOP                    │
│                                                      │
│  ┌──────────────┐    ┌──────────────┐               │
│  │  Hypothesis   │───→│   Apply      │               │
│  │  Generator    │    │   Changes    │               │
│  └──────────────┘    └──────┬───────┘               │
│                             │                        │
│                      ┌──────▼───────┐               │
│                      │  Synthesize   │               │
│                      │  Test Phrases │               │
│                      └──────┬───────┘               │
│                             │                        │
│                      ┌──────▼───────┐               │
│                      │   Evaluate    │               │
│                      │   Quality     │               │
│                      └──────┬───────┘               │
│                             │                        │
│                      ┌──────▼───────┐               │
│                      │   Compare    │               │
│                      │ vs Baseline  │               │
│                      └──────┬───────┘               │
│                             │                        │
│                    ┌────────┼────────┐              │
│                    │        │        │              │
│               Improved   Same    Regressed          │
│                    │        │        │              │
│               ┌────▼──┐ ┌──▼──┐ ┌──▼────┐         │
│               │ Keep  │ │Skip │ │Discard│         │
│               │+Record│ │     │ │+Reset │         │
│               └───────┘ └─────┘ └───────┘         │
│                                                      │
│  Loop continues indefinitely until convergence       │
└─────────────────────────────────────────────────────┘
```

## Composite Quality Score

The single scalar metric combining all quality dimensions:

```python
score = (
    0.40 * (1.0 - wer)           # Intelligibility (most important)
  + 0.25 * normalize_mcd(mcd)     # Acoustic similarity to reference
  + 0.15 * normalize_snr(snr)     # Signal cleanliness
  + 0.10 * correlation            # Waveform match to reference
  + 0.10 * (1.0 - normalize_thd(thd))  # Low distortion
)
```

Where normalization maps each metric to [0.0, 1.0]:
- `normalize_mcd(mcd)`: MCD < 2dB → 1.0, MCD > 10dB → 0.0 (linear)
- `normalize_snr(snr)`: SNR > 40dB → 1.0, SNR < 5dB → 0.0 (linear)
- `normalize_thd(thd)`: THD < 1% → 1.0, THD > 50% → 0.0 (linear)

Score ranges:
- **0.90+** — Excellent (near-reference quality)
- **0.75-0.90** — Good (production-ready)
- **0.60-0.75** — Acceptable (needs improvement)
- **< 0.60** — Poor (significant issues)

## Search Space

### Tier 1: Configuration Parameters (fast, no rebuild needed)

| Parameter | Range | Step | Default |
|-----------|-------|------|---------|
| `temperature` | 0.0 - 1.0 | 0.05 | 0.7 |
| `top_p` | 0.1 - 1.0 | 0.05 | 0.9 |
| `consistency_steps` | 1 - 4 | 1 | 2 |
| `speed` | 0.8 - 1.2 | 0.05 | 1.0 |
| `seed` | various | - | 42 |

This gives ~20 × 18 × 4 × 8 = **11,520** configuration combinations for grid search, or can be explored via random/Bayesian search.

### Tier 2: Model-Level Tuning (requires rebuild)

- FlowNet step schedule modifications
- Attention temperature scaling
- Layer norm epsilon adjustments
- Sampling strategy changes (top-k, typical sampling, etc.)

### Tier 3: Code-Level Changes (requires rebuild + review)

- Numerical precision adjustments (f32 vs f64 for critical paths)
- Alternative activation functions
- Convolution padding strategies
- Resampling filter parameters

## Test Corpus

A fixed set of test phrases covering different speech patterns:

```python
TEST_PHRASES = [
    # Short, simple
    "Hello, how are you today?",
    # Medium with numbers
    "The temperature is 72 degrees Fahrenheit.",
    # Long, complex
    "In the beginning, there was nothing but darkness and silence, until a single spark of light appeared.",
    # Prosody challenge (question)
    "Did you really think that was going to work?",
    # Technical terms
    "The API returns a JSON response with the authentication token.",
]
```

Each phrase is synthesized with the current config and evaluated against Python-generated reference audio.

## Loop Protocol

### Phase 1: Baseline Establishment
1. Synthesize all test phrases with default config
2. Measure quality metrics for each
3. Compute composite score → this is the starting baseline
4. Save baseline to `autotuning/baselines/initial.json`

### Phase 2: Parameter Sweep
1. Pick a parameter to vary (start with highest-impact: temperature, consistency_steps)
2. Try N values across the parameter range
3. For each value:
   a. Synthesize all test phrases
   b. Compute composite score
   c. If score > best_score: save as new best, record in results.tsv
   d. If score ≤ best_score: discard, note in results.tsv
4. Move to next parameter with the best value locked in

### Phase 3: Combinatorial Refinement
1. With best individual parameters identified, try combinations
2. Bayesian optimization over the joint parameter space
3. Each experiment: modify config → synthesize → evaluate → keep/discard

### Phase 4: Code-Level Exploration (AI agent loop)
1. Analyze quality reports to identify specific failure modes
2. Hypothesize code changes that could address them
3. Apply change → rebuild → synthesize → evaluate → keep/discard
4. This is the "autoresearch" loop proper

## Results Logging

All experiments logged to `autotuning/results.tsv`:

```tsv
experiment_id	timestamp	temperature	top_p	consistency_steps	speed	seed	composite_score	wer	mcd	snr	thd	correlation	status	description
001	2026-03-10T12:00:00	0.7	0.9	2	1.0	42	0.72	0.05	4.5	42.3	0.8	0.95	baseline	Default configuration
002	2026-03-10T12:01:30	0.5	0.9	2	1.0	42	0.75	0.03	4.2	43.1	0.7	0.96	improved	Lower temperature
003	2026-03-10T12:03:00	0.3	0.9	2	1.0	42	0.71	0.08	4.0	44.0	0.6	0.94	regressed	Too low temperature
```

## Implementation Plan

### Files to Create

```
autotuning/
├── README.md              # This document (adapted)
├── program.md             # AI agent instructions (autoresearch-style)
├── autotune.py            # Main loop orchestrator
├── scorer.py              # Composite quality scoring
├── synthesizer.py         # Rust TTS invocation wrapper
├── results.tsv            # Experiment log (auto-generated)
├── baselines/             # Saved baselines
│   └── initial.json
└── configs/               # Saved best configurations
    └── best.json
```

### Dependencies on Existing Infrastructure

- `validation/quality_metrics.py` — WER, MCD, SNR, THD computation
- `validation/baseline_tracker.py` — Baseline comparison
- `src/config.rs` / `TTSConfig` — Parameter definitions
- `cargo build --release` — Rebuilding after code changes
- Reference audio from Python TTS pipeline

## Key Differences from Autoresearch

| Aspect | Autoresearch | TTS Autotuning |
|--------|-------------|----------------|
| What changes | `train.py` source code | Config params → model code |
| Evaluation time | 5 minutes (fixed) | ~10-30 seconds per phrase |
| Metric | Single (`val_bpb`) | Composite (weighted multi-metric) |
| Search space | Unbounded (code changes) | Bounded (param ranges) + unbounded (code) |
| Memory | Git commits | `results.tsv` + git |
| Agent | Modifies training code | Modifies config, then code |
| Convergence | Never (always exploring) | Approaches 1.0 asymptotically |

## Convergence Expectations

Based on the parameter space:
- **Tier 1 (config sweep)**: ~50-200 experiments, likely 5-15% improvement
- **Tier 2 (model tuning)**: ~20-50 experiments, incremental gains
- **Tier 3 (code changes)**: Unbounded, largest potential gains

The autoresearch pattern is most valuable at Tier 3 where the AI agent proposes code-level hypotheses, tests them, and learns from results. Tiers 1-2 are systematic sweeps that build the foundation.

## Running the Loop

```bash
# Phase 1: Establish baseline
python autotuning/autotune.py --phase baseline

# Phase 2: Parameter sweep
python autotuning/autotune.py --phase sweep --param temperature

# Phase 3: Joint optimization
python autotuning/autotune.py --phase optimize --iterations 100

# Phase 4: AI agent loop (run in Claude Code session)
# Paste autotuning/program.md as the prompt
```
