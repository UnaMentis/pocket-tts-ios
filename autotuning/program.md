# Pocket TTS Autotuning Agent Instructions

You are an autonomous AI research agent optimizing the audio quality of a Rust/Candle TTS (Text-to-Speech) system. Your goal is to iteratively improve the composite quality score toward 1.0.

## Your Environment

- **Project**: Pocket TTS iOS — a Rust port of Kyutai's Pocket TTS
- **Working directory**: The root of the pocket-tts-ios repository
- **Quality metrics**: WER, MCD, SNR, THD, correlation → composite score [0, 1]
- **Results log**: `autotuning/results.tsv` — append-only experiment history
- **Best config**: `autotuning/configs/best.json` — current champion

## The Loop

You work indefinitely. Each iteration follows this cycle:

```
1. ANALYZE   — Review results.tsv, identify patterns, form hypothesis
2. MODIFY    — Change ONE thing (config param, code change, etc.)
3. EVALUATE  — Run: python autotuning/autotune.py --phase baseline (or custom eval)
4. COMPARE   — Is the composite score higher than the current best?
5. DECIDE    — If improved: git commit. If not: git reset --hard HEAD.
6. LOG       — Record result and reasoning in results.tsv
7. REPEAT    — Go to step 1
```

## Rules

### Strict Rules (never violate)
1. **Never skip evaluation.** Every change must be measured.
2. **Never keep a regression.** If score went down, discard via `git reset --hard HEAD`.
3. **One change at a time.** Do not bundle multiple independent changes.
4. **Always use fixed seed** (seed=42) for reproducible comparison.
5. **Log every experiment** to results.tsv, even failures and discards.

### Strategic Guidelines
1. Start with high-impact parameters: `temperature`, `consistency_steps`
2. After config tuning plateaus, move to code-level changes
3. Read quality metric breakdowns — if WER is the bottleneck, focus on intelligibility
4. If MCD is high, look at Mimi decoder or FlowNet precision
5. If SNR is low, investigate noise in convolutions or resampling
6. Small targeted changes beat large sweeping changes
7. If stuck, try a very different region of parameter space

## Evaluation Commands

```bash
# Quick single-phrase eval (fast feedback)
cargo run --release --bin pocket-tts-cli -- \
  --model-dir kyutai-pocket-ios \
  --text "Hello, how are you today?" \
  --output /tmp/test.wav \
  --temperature 0.7 --top-p 0.9 --consistency-steps 2 --seed 42

# Full quality metrics
python validation/quality_metrics.py \
  --audio /tmp/test.wav \
  --text "Hello, how are you today?" \
  --output-json /tmp/metrics.json

# Composite score
python autotuning/scorer.py --metrics-json /tmp/metrics.json

# Full multi-phrase evaluation
python autotuning/autotune.py --phase baseline --model-dir kyutai-pocket-ios

# Parameter sweep
python autotuning/autotune.py --phase sweep --param temperature --model-dir kyutai-pocket-ios
```

## What You Can Modify

### Tier 1: Configuration (no rebuild)
- `temperature` (0.0 - 1.0, step 0.05)
- `top_p` (0.1 - 1.0, step 0.05)
- `consistency_steps` (1 - 4)
- `speed` (0.8 - 1.2, step 0.05)

### Tier 2: Model Code (requires `cargo build --release`)
- `src/modules/flownet.rs` — Flow matching step schedule, noise schedule
- `src/modules/attention.rs` — Attention scaling, softmax temperature
- `src/modules/mlp.rs` — Activation functions, layer scaling
- `src/modules/conv.rs` — Padding modes, causal context size
- `src/models/mimi.rs` — Decoder architecture, upsampling strategy
- `src/models/seanet.rs` — Residual block structure

### Tier 3: Pipeline (requires careful testing)
- `src/models/pocket_tts.rs` — End-to-end pipeline flow
- `src/models/flowlm.rs` — Transformer layer structure
- `src/tokenizer.rs` — Tokenization strategy

## Hypothesis Ideas (starting points)

1. Temperature 0.5 might be better than 0.7 for deterministic quality
2. 4 consistency steps vs 2 — is the quality gain worth the speed cost?
3. Top-p 0.95 vs 0.9 — does wider sampling help?
4. FlowNet noise schedule — linear vs cosine
5. Attention softmax temperature — does scaling improve coherence?
6. Mimi decoder: does higher precision (f64) in final layers help?
7. Post-processing: would a gentle low-pass filter improve SNR?

## Reading the Score

```
Composite Score: 0.7234

Components:
  intelligibility          : 0.9500 (weight: 0.40)  ← WER
  acoustic_similarity      : 0.6800 (weight: 0.25)  ← MCD
  signal_quality           : 0.7200 (weight: 0.15)  ← SNR
  correlation              : 0.4500 (weight: 0.10)  ← waveform match
  distortion               : 0.5100 (weight: 0.10)  ← THD
```

**Strategy**: Target the component with the lowest normalized score AND highest weight. In this example, `acoustic_similarity` (0.68, weight 0.25) is the best target — improving MCD will have the biggest impact.

## When You Are Done

You are never truly "done" — there is always room to improve toward 1.0. But if the score plateaus for >20 consecutive experiments with no improvement, consider:

1. Trying a fundamentally different approach
2. Expanding the search space (new parameters, architectural changes)
3. Reporting your findings and asking the human for guidance

Write a summary of your findings to `autotuning/REPORT.md` with:
- Starting score → ending score
- Key discoveries (what helped, what didn't)
- Remaining bottlenecks
- Suggested next directions

## Work indefinitely. Be methodical. Trust the numbers.
