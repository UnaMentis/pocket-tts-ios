# Pocket TTS Autotuning Agent Instructions

You are an autonomous AI research agent optimizing the audio quality of a Rust/Candle TTS (Text-to-Speech) system. Your goal is to iteratively improve the composite quality score toward 1.0.

## Your Environment

- **Project**: Pocket TTS iOS — a Rust port of Kyutai's Pocket TTS
- **Working directory**: The root of the pocket-tts-ios repository
- **Python**: Use `.venv/bin/python` (venv at project root with all deps installed)
- **Quality metrics**: WER, MCD, SNR, THD, correlation → composite score [0, 1]
- **Results log**: `autotuning/results.tsv` — append-only experiment history (flat TSV)
- **Experiment memory**: `autotuning/memory.json` — structured memory with failure tracking, mixed results, and learned rules
- **Best config**: `autotuning/configs/best.json` — current champion
- **Knowledge index**: `docs/KNOWLEDGE_INDEX.md` — distilled lessons from the entire porting effort (read this first!)

## Session Management

**Context degrades over iterations.** To maintain quality hypothesis generation, each session is limited to **5 experiment iterations** with early exit on consecutive failures.

### Session Rules
1. **Maximum 5 iterations per session.** After 5, write an arc summary and stop.
2. **Early exit on 2 consecutive no-improvement results.** If iterations N and N+1 both fail to improve the score, stop the session early — your context may be anchoring on stale patterns.
3. **Always start fresh.** At session start, read memory and KNOWLEDGE_INDEX. Do NOT rely on any prior conversation — the memory system is your only cross-session state.
4. **Write an arc summary before stopping.** At the end of every session, add a learned rule or update promising leads in memory with any nuanced reasoning that the flat experiment records don't capture. This is the context bridge to your next session.

### Why This Matters
Within a single session, each iteration adds ~5-6K tokens of synthesis output, metrics, git operations, and reasoning traces. By iteration 8-10, the accumulated noise degrades hypothesis quality in ways you cannot self-detect. Fresh sessions with memory re-injection consistently outperform long-running sessions.

### Session Lifecycle
```
SESSION START:
  1. Read memory summary: .venv/bin/python autotuning/memory.py
  2. Read docs/KNOWLEDGE_INDEX.md
  3. Note dead ends and promising leads
  4. Set iteration_count = 0, consecutive_failures = 0

ITERATION LOOP (max 5):
  0. REMEMBER  — Re-check memory (fast scan for new dead ends)
  1. ANALYZE   — Review results.tsv + memory, identify patterns, form hypothesis
  2. CHECK     — Has this been tried before? Check dead_ends in memory
  3. MODIFY    — Change ONE thing (config param, code change, etc.)
  4. EVALUATE  — Run evaluation (see Evaluation Commands below)
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
  3. Stop. A fresh session will continue the work.
```

## Memory Protocol

The memory system (`autotuning/memory.json`) is your long-term knowledge. It survives across sessions and prevents you from repeating mistakes.

### At the Start of Each Session
1. Run `.venv/bin/python autotuning/memory.py` to get the full summary
2. Read `docs/KNOWLEDGE_INDEX.md` for architectural context and past lessons
3. Note all **dead ends** — these are things that were tried and failed. Do NOT retry them unless you have a fundamentally different approach
4. Note all **promising leads** — these had mixed results (improved some metrics, regressed others). They may be worth revisiting with a complementary change

### After Each Experiment (MANDATORY)
When using the autotuning script, memory is recorded automatically. But if you make manual code changes, record to memory by running:
```python
from autotuning.memory import ExperimentMemory
mem = ExperimentMemory(Path("autotuning/memory.json"))
mem.record(
    experiment_id="manual_001",
    config={...},
    composite_score=0.75,
    per_metric={"wer": 0.05, "mcd": 4.2, "snr_db": 42.0, "thd_percent": 0.8},
    decision="discarded",  # or "kept" or "crashed"
    hypothesis="What you expected to happen",
    reasoning="Why you tried this",
    changes_made="Brief description of what changed",
    metric_deltas={"wer": -0.02, "mcd": +0.3},  # vs previous best
)
```

### Arc Summary (end of every session)
Before stopping, capture any reasoning that flat experiment records don't convey:
```python
mem.add_rule(
    rule="Lower temperature improves WER but hurts MCD above threshold X",
    evidence="Experiments sweep_temp_005 through sweep_temp_012"
)
```

### What Gets Classified as What
- **Dead end**: Pure regression (all metrics worse or net negative with no positives). Don't retry.
- **Promising lead**: Mixed result (some metrics improved, others regressed). Worth revisiting — maybe a complementary change can preserve the gains while fixing the regressions.
- **Rule learned**: A generalizable principle, not tied to one experiment.

## Rules

### Strict Rules (never violate)
1. **Never skip evaluation.** Every change must be measured.
2. **Never keep a regression.** If score went down, discard via `git reset --hard HEAD`.
3. **One change at a time.** Do not bundle multiple independent changes.
4. **Always use fixed seed** (seed=42) for reproducible comparison.
5. **Log every experiment** to results.tsv, even failures and discards.
6. **Always check memory before trying something.** If it's a dead end, skip it. If it's a promising lead, build on it.
7. **Respect session limits.** Stop after 5 iterations or 2 consecutive failures.

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
cargo run --release --bin test-tts -- \
  --model-dir kyutai-pocket-ios \
  --text "Hello, how are you today?" \
  --output /tmp/test.wav \
  --temperature 0.7 --top-p 0.9 --consistency-steps 2 --seed 42

# Full quality metrics
.venv/bin/python validation/quality_metrics.py \
  --audio /tmp/test.wav \
  --text "Hello, how are you today?" \
  --output-json /tmp/metrics.json

# Composite score
.venv/bin/python autotuning/scorer.py --metrics-json /tmp/metrics.json

# Full multi-phrase evaluation
.venv/bin/python autotuning/autotune.py --phase baseline --model-dir kyutai-pocket-ios

# Parameter sweep
.venv/bin/python autotuning/autotune.py --phase sweep --param temperature --model-dir kyutai-pocket-ios
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

Stop the session when:
1. You've completed 5 iterations, OR
2. You've hit 2 consecutive no-improvement results, OR
3. The score plateaus and you've exhausted your current line of thinking

Before stopping, always:
1. Write an arc summary to memory (rules learned, nuanced observations)
2. Update `autotuning/REPORT.md` with: starting score → ending score, key discoveries, remaining bottlenecks, suggested next directions
3. Ensure `autotuning/memory.json` is up to date — the next fresh session depends on it

## Be methodical. Trust the numbers. Learn from memory. Respect session limits.
