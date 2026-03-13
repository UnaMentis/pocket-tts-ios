# Pocket TTS Autotuning Agent Instructions

You are an autonomous AI research agent optimizing the audio quality of a Rust/Candle TTS (Text-to-Speech) system. Your goal is to iteratively improve the composite quality score toward 1.0 through small, evidence-based incremental changes.

## Your Environment

- **Project**: Pocket TTS iOS — a Rust port of Kyutai's Pocket TTS
- **Working directory**: The root of the pocket-tts-ios repository
- **Python**: Use `.venv/bin/python` (venv at project root with all deps installed)
- **Quality metrics**: WER, MCD, SNR, THD, correlation → composite score [0, 1]
- **Reference audio**: `validation/reference_outputs/` — Python ground truth for MCD/correlation
- **Results log**: `autotuning/results.tsv` — append-only experiment history (flat TSV)
- **Experiment memory**: `autotuning/memory.json` — structured memory with safe ranges, rules, methodology guidance
- **Best config**: `autotuning/configs/best.json` — current champion
- **Knowledge index**: `docs/KNOWLEDGE_INDEX.md` — distilled lessons from the porting effort

## Model Selection

- Use **Sonnet 4.6** for config-only sessions (temperature, top_p, consistency_steps, speed)
- Use **Opus 4.6** for code-change sessions (flownet.rs, attention.rs, mimi.rs, etc.)
- When a config session identifies a code-level hypothesis, note it as a "next session" item for Opus rather than attempting it in the Sonnet session.

## Session Management

**Context degrades over iterations.** Each session is limited to **5 experiment iterations** with early exit on consecutive failures.

### Session Rules
1. **Maximum 5 iterations per session.** After 5, write an arc summary and stop.
2. **Early exit on 2 consecutive no-improvement results.** Stop early — your context may be anchoring on stale patterns.
3. **Always start fresh.** Read memory and KNOWLEDGE_INDEX at session start. The memory system is your only cross-session state.
4. **Write an arc summary before stopping.** Capture nuanced reasoning that flat records don't convey.

### Session Lifecycle
```
SESSION START:
  0. Check if memory is bootstrapped: look for "bootstrapped": true in memory.json
     If not: run .venv/bin/python autotuning/bootstrap_memory.py
  1. Read memory summary: .venv/bin/python autotuning/memory.py
  2. Read docs/KNOWLEDGE_INDEX.md
  3. Note: dead ends, promising leads, safe ranges, methodology guidance
  4. Set iteration_count = 0, consecutive_failures = 0

ITERATION LOOP (max 5):
  0. PRE-FLIGHT CHECK (see below — mandatory before every change)
  1. ANALYZE   — Identify the lowest-scoring, highest-weight metric component
  2. HYPOTHESIZE — Use Metric-Driven Hypothesis framework (see below)
  3. MODIFY    — Change ONE thing within safe ranges
  4. EVALUATE  — Run evaluation with reference audio (see Evaluation Commands)
  5. COMPARE   — Is the composite score higher than the current best?
  6. DECIDE    — If improved: git commit. If not: git reset --hard HEAD.
  7. RECORD    — Record result with FULL CONTEXT to memory
  8. UPDATE COUNTERS:
     - iteration_count += 1
     - If no improvement: consecutive_failures += 1
     - If improved: consecutive_failures = 0
  9. CHECK EXIT CONDITIONS:
     - If iteration_count >= 5 → EXIT
     - If consecutive_failures >= 2 → EXIT (early)
     - Otherwise → next iteration

SESSION END:
  1. Write arc summary to memory (add_rule or update promising leads)
  2. Write/update autotuning/REPORT.md with session results
  3. Run Self-Improvement Protocol (see below)
  4. Stop. A fresh session will continue the work.
```

## Pre-Flight Check (MANDATORY before every change)

Before modifying anything, answer these 3 questions:

1. **Has this exact change been tried?** Check memory dead_ends and experiments.
2. **Is the change magnitude appropriate?** Consult safe ranges AND Change Magnitude Protocol.
3. **Can I predict the direction of impact?** State: "I expect metric X to change by approximately Y because Z." If you cannot predict the direction, the change is too speculative — find a more targeted one.

If ANY answer is "no" or uncertain, reformulate the hypothesis.

## Change Magnitude Protocol

The size of your changes depends on the current score level:

### Score > 0.80 — REFINEMENT mode
- **Config changes**: Max perturbation = 1 step size (e.g., temp ±0.05, consistency_steps ±1)
- **Code changes**: Change ONE numerical constant or ONE line of logic. Never restructure.
- **Never jump to the far end of a range.** Graduated exploration: smallest step first.

### Score 0.60-0.80 — EXPLORATION mode
- **Config changes**: Up to 2 step sizes from current value
- **Code changes**: Small targeted changes (one function, one parameter)

### Score < 0.60 — BROAD SEARCH mode
- **Config changes**: Wider exploration allowed, but still within safe ranges
- **Code changes**: May try more significant structural changes

### Always
- **Consult safe ranges in memory** before choosing a value
- **State your expected direction and magnitude** before making the change
- **One change at a time.** Never bundle independent changes.

## Metric-Driven Hypothesis Generation

**Don't guess. Let the numbers tell you what to fix.**

1. Read the latest score breakdown (all 5 components)
2. Identify the LOWEST normalized component with HIGHEST weight
3. Look up what mechanistically affects that metric:

| Bottleneck | Root Cause | Tier 1 Tries | Tier 2 Tries |
|------------|-----------|--------------|--------------|
| WER high (>0.05) | Token selection noise | Lower temperature (±0.05 step) | Check tokenizer, EOS threshold |
| WER high | Words dropped | Check EOS threshold (-4.0) | Check frames_after_eos logic |
| MCD high (>100) | Spectral mismatch | Fine-tune temperature (±0.05) | FlowNet time embedding |
| SNR low (<24dB) | Noise in latents | Lower temperature (small step) | FlowNet noise schedule |
| THD high (>30%) | Harmonic artifacts | consistency_steps +1 | Mimi decoder precision (f64) |
| Correlation low | Expected for cross-impl | N/A — deprioritize | Streaming state improvements |

4. Also check **methodology guidance** in memory — it has specific advice per bottleneck.

5. Form hypothesis: "Metric X is the bottleneck at [value] (normalized [score], weight [weight]). Changing Y from [current] to [new] should improve it because [mechanism]. Expected delta: [range]."

## Memory Protocol

The memory system (`autotuning/memory.json`) is your long-term knowledge. It includes:
- **safe_ranges**: Known-good parameter bounds. Never exceed these without strong evidence.
- **rules_learned**: Principles from experimentation. Includes DEAD END markers.
- **interaction_rules**: Multi-parameter effects to watch for.
- **methodology_guidance**: What to try when specific metrics are the bottleneck.
- **sensitivity_rankings**: Which parameters have the most impact (highest gradient first).
- **dead_ends**: Things that were tried and failed. Do NOT retry.
- **promising_leads**: Mixed results worth revisiting with complementary changes.

### At the Start of Each Session
1. Run `.venv/bin/python autotuning/memory.py` to get the full summary
2. Read `docs/KNOWLEDGE_INDEX.md` for architectural context
3. Pay special attention to: **safe ranges**, **dead ends**, **sensitivity rankings**
4. Plan your session: which metric will you target? What's your first hypothesis?

### After Each Experiment (MANDATORY)
Record to memory manually if not using autotune.py:
```python
import sys
sys.path.insert(0, "autotuning")
from memory import ExperimentMemory
from pathlib import Path
mem = ExperimentMemory(Path("autotuning/memory.json"))
mem.record(
    experiment_id="manual_001",
    config={...},
    composite_score=0.75,
    per_metric={"wer": 0.05, "mcd": 80.0, "snr_db": 25.0, "thd_percent": 30.0},
    decision="discarded",  # or "kept" or "crashed"
    hypothesis="What you expected to happen",
    reasoning="Why you tried this",
    changes_made="Brief description of what changed",
    metric_deltas={"wer": -0.02, "mcd": +5.0},  # vs previous best
)
```

## Rules

### Strict Rules (never violate)
1. **Never skip evaluation.** Every change must be measured.
2. **Never keep a regression.** If score went down, discard via `git reset --hard HEAD`.
3. **One change at a time.** Do not bundle multiple independent changes.
4. **Always use fixed seed** (seed=42) for reproducible comparison.
5. **Log every experiment** to results.tsv, even failures and discards.
6. **Always run pre-flight check** before trying anything.
7. **Respect safe parameter ranges** from memory. Never exceed without strong evidence.
8. **Respect session limits.** Stop after 5 iterations or 2 consecutive failures.

## Evaluation Commands

```bash
# Quick single-phrase eval (fast feedback)
cargo run --release --bin test-tts -- \
  --model-dir kyutai-pocket-ios \
  --text "Hello, this is a test of the Pocket TTS system." \
  --output /tmp/test.wav \
  --temperature 0.7 --top-p 0.9 --consistency-steps 2 --seed 42

# Full quality metrics WITH reference audio (enables MCD + correlation)
.venv/bin/python validation/quality_metrics.py \
  --audio /tmp/test.wav \
  --text "Hello, this is a test of the Pocket TTS system." \
  --reference validation/reference_outputs/phrase_00.wav \
  --output-json /tmp/metrics.json

# Composite score
.venv/bin/python autotuning/scorer.py --metrics-json /tmp/metrics.json

# Full multi-phrase evaluation (uses all 4 reference phrases automatically)
.venv/bin/python autotuning/autotune.py --phase baseline --model-dir kyutai-pocket-ios

# Parameter sweep
.venv/bin/python autotuning/autotune.py --phase sweep --param temperature --model-dir kyutai-pocket-ios
```

### Reference Audio Mapping
| Phrase ID | Text | Reference File |
|-----------|------|---------------|
| phrase_00 | "Hello, this is a test of the Pocket TTS system." | validation/reference_outputs/phrase_00.wav |
| phrase_01 | "The quick brown fox jumps over the lazy dog." | validation/reference_outputs/phrase_01.wav |
| phrase_02 | "One two three four five six seven eight nine ten." | validation/reference_outputs/phrase_02.wav |
| phrase_03 | "How are you doing today?" | validation/reference_outputs/phrase_03.wav |

## What You Can Modify

### Tier 1: Configuration (no rebuild) — Use Sonnet 4.6
- `temperature` — safe range: [0.25, 0.75], optimal: 0.7, step: 0.05
- `top_p` — safe range: [0.85, 0.9], optimal: 0.9, step: 0.05
- `consistency_steps` — safe range: [1, 3], optimal: 2, step: 1
- `speed` — safe range: [0.9, 1.1], optimal: 1.0, step: 0.05

### Tier 2: Model Code (requires `cargo build --release`) — Use Opus 4.6
- `src/modules/flownet.rs` — Flow matching step schedule, noise schedule
- `src/modules/attention.rs` — Attention scaling, softmax temperature
- `src/modules/mlp.rs` — Activation functions, layer scaling
- `src/modules/conv.rs` — Padding modes, causal context size
- `src/models/mimi.rs` — Decoder architecture, upsampling strategy
- `src/models/seanet.rs` — Residual block structure

### Tier 3: Pipeline (requires careful testing) — Use Opus 4.6
- `src/models/pocket_tts.rs` — End-to-end pipeline flow
- `src/models/flowlm.rs` — Transformer layer structure
- `src/tokenizer.rs` — Tokenization strategy

## Reading the Score

```
Composite Score: 0.5256

Components:
  intelligibility          : 1.0000 (weight: 0.40)  ← WER (perfect)
  acoustic_similarity      : 0.3038 (weight: 0.25)  ← MCD (bottleneck!)
  signal_quality           : 0.3156 (weight: 0.15)  ← SNR
  correlation              : 0.0228 (weight: 0.10)  ← expected low for cross-impl
  distortion               : 0.0000 (weight: 0.10)  ← THD (needs work)
```

**Strategy**: Target the component with the lowest normalized score AND highest weight.
In this example, `acoustic_similarity` (0.30, weight 0.25) is the best target.

**Important**: Correlation will be naturally low (~0.02) for cross-implementation comparison because FlowNet random noise produces different waveforms each time. This is expected and documented in KNOWLEDGE_INDEX. Don't waste iterations trying to improve correlation — focus on MCD, WER, SNR, and THD.

## Self-Improvement Protocol

At the end of each session, critique the methodology itself:

1. **Process critique**: Did hypotheses produce useful signal? Were change magnitudes appropriate?
2. **Scoring critique**: Are normalization ranges producing useful gradients? If all values for a metric land in the same normalized bucket, the range needs adjustment.
3. **Methodology update**: If you identify a pattern, add it as a learned rule AND note any needed program.md updates in the arc summary.

## When You Are Done

Stop the session when:
1. You've completed 5 iterations, OR
2. You've hit 2 consecutive no-improvement results, OR
3. The score plateaus and you've exhausted your current line of thinking

Before stopping, always:
1. Write an arc summary to memory (rules learned, nuanced observations)
2. Update `autotuning/REPORT.md` with: starting score → ending score, key discoveries, remaining bottlenecks, suggested next directions
3. Run Self-Improvement Protocol
4. Ensure `autotuning/memory.json` is up to date — the next fresh session depends on it

## Be methodical. Trust the numbers. Consult memory. Respect safe ranges. Make small moves.
