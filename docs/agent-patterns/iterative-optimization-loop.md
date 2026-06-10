# Pattern: Autoresearch-Style Iterative Optimization Loop

> **Where this fits**: this is one of the agentic collaboration patterns in use
> across my projects. It is intended to be readable on its own *and* to drop
> straight into a cross-project presentation covering multiple patterns. Where
> this project's specifics appear, they are clearly labelled so they can be
> generalized or stripped out.

---

## TL;DR

An autonomous worker agent performs one bounded-scope improvement attempt per
invocation, measures the result with a hard quantitative gate, and **keeps the
change iff the gate says it improved**. Regressions are automatically reverted.
A timer-driven outer loop keeps firing the agent on a cadence. Around the core
loop sits a small cast of specialized auditing/monitoring agents that handle
concerns the worker shouldn't: measurement stability, research breakouts, dead
code, and progress reporting.

The pattern is inspired by [Karpathy's
autoresearch](https://github.com/karpathy/autoresearch), where an LLM
iteratively modifies a training script and commits only when `val_bpb`
improves. In this project it is adapted for quality optimization of a Rust port
of a neural TTS model, where the gated metric is a weighted composite of
waveform correlation, WER, MCD, SNR and THD against a Python reference.

**Outcome here:** end-to-end waveform correlation went from **0.0016** (Jan
2026) → **0.839** (noise-alignment fix, 2026-03-18) → **1.000** (causal + context
attention mask fix, 2026-03-21), with the last two jumps produced inside this
loop.

---

## 1. The Core Loop

### 1.1 Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                 /loop <interval>  /optimize                      │
│  (outer driver — fires the worker on a cadence, e.g. every 5m)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      /optimize  (one iteration)                  │
│                                                                  │
│   1. READ STATE from disk (memory, approaches-tried, reports)   │
│   2. PRE-FLIGHT CHECK  (3 mandatory questions)                  │
│   3. HYPOTHESIZE       (bottleneck → root cause → change)       │
│   4. MODIFY            (exactly ONE change, built)              │
│   5. EVALUATE          (standardized eval script, JSON out)     │
│   6. COMPARE & DECIDE                                           │
│         improved?  → KEEP (git add, update best)                │
│         regressed? → DISCARD (git checkout -- src/)             │
│   7. RECORD            (memory.json + approaches-tried.md +     │
│                         results.tsv + REPORT.md — ALWAYS)       │
│   8. EXIT              (next invocation picks up from disk)     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 The two properties that make this safe

1. **Monotonicity by construction.** The worker is *not trusted* to judge
   whether its change helped. It must execute a standardized evaluation and
   read back a numeric composite score. If the score didn't go up, the
   change is reverted in the same turn. The codebase can only move forward or
   stay flat; it cannot drift downward.

2. **Fresh context every iteration.** The worker is invoked with
   `context: fork`, so each run starts with an empty conversation and pulls
   all prior state from disk. This is the opposite of a long-running
   conversation that accumulates bias — every hypothesis has to justify itself
   against a distilled summary rather than "what we've been thinking about".

### 1.3 The hard-gate primitive

The keep/discard decision is a *data contract*, not a judgement:

```
bash .claude/skills/optimize/evaluate.sh
# writes /tmp/optimize-metrics.json with composite_score, components, config
```

The evaluator is locked to fixed inputs:

- Seed 42
- A captured noise-tensor set (`--noise-dir`) so the reference and the
  implementation use bit-identical random draws
- `consistency_steps=1` so FlowNet sampling is deterministic
- Same test phrase every run

Without that reproducibility, the gate would be noisy and you'd keep "winners"
that were really just RNG. **This is the single biggest thing that separates
"actually works" from "theatre" — when I got lazy about noise matching, the
gate became meaningless and the loop rewarded random drift.**

### 1.4 Composite score

```
correlation       × 0.50   ← primary, dominates decisions
intelligibility   × 0.20   ← 1 − WER (Whisper)
acoustic_similarity × 0.15 ← normalized MCD
signal_quality    × 0.08   ← normalized SNR
distortion        × 0.07   ← normalized (1 − THD)
```

A single scalar is essential — without it, "did it improve?" becomes a
subjective argument. The weights are project-specific but the principle is
general: **pick a primary signal, weight it heavily enough that it can't be
overwhelmed by diagnostics, and normalize everything into [0, 1].**

---

## 2. What I Added On Top of the Base Pattern

The bare autoresearch pattern is "change → evaluate → commit or reset, loop
forever". That's the skeleton. The rest of this section is the meaningful
customization for this project.

### 2.1 Structured experiment memory (`autotuning/memory.py`)

A JSON store that's richer than a flat log. On every iteration the worker
appends an entry and the memory automatically classifies it:

- `pure_improvement` — all metric deltas non-negative
- `pure_regression` — all metric deltas non-positive → added to `dead_ends`
- `mixed` — some up, some down → added to `promising_leads`
- `neutral` — no meaningful change

It also maintains:

- `safe_ranges` per parameter (min/optimal/max) — the worker must respect
  these unless it has strong evidence
- `rules_learned` — human-readable principles earned from experiments
- `interaction_rules` — known multi-parameter effects
- `methodology_guidance` — keyed by *bottleneck*, not by *parameter*, so the
  worker can look up "when correlation is the bottleneck, try X first"
- `sensitivity_rankings` — which knobs have the highest gradient
- `bootstrapped` flag — first-run seeding from project history

**Why this matters for the pattern, not just this project:** flat experiment
logs don't prevent repeated mistakes; structured memory does. The worker is
told "don't repeat anything in `dead_ends`" and the store tracks them
automatically. This is the closest thing to *learning* the loop does, since
each invocation has no memory of the previous one.

### 2.2 Dual memory surfaces

- `autotuning/memory.json` — machine-queryable, the agent reads it through a
  `memory.py` summary command
- `docs/audit/approaches-tried.md` — human-readable markdown log with the
  narrative ("What / Why / Result / Status / Files") for each non-trivial
  attempt

Keeping both surfaces is deliberate. The JSON is for the agent's bottleneck
lookups and dead-end filtering. The markdown is what *I* actually read when I
open the project after a week, and it's what the Research Advisor agent reads
to avoid re-suggesting a known-dead approach in human prose.

### 2.3 Pre-flight check (mandatory)

Before any change, the worker must answer three questions:

1. Has this exact change already been tried?
2. Is the change magnitude appropriate for the current score level?
3. Can I predict the direction of impact? "I expect metric X to change by ~Y
   because Z."

If any answer is "no" or "unsure", the hypothesis has to be reformulated. In
practice this single guard rail eliminated most of the "random thrashing"
failure mode where the agent would try wildly speculative things just because
it had a fresh context.

### 2.4 Change Magnitude Protocol (adaptive exploration)

The allowed change size scales with the current score:

- **Score > 0.80 — REFINEMENT**: one-step perturbations only, one constant or
  one line at a time, never restructure
- **Score 0.60–0.80 — EXPLORATION**: up to two-step perturbations, small
  targeted code changes
- **Score < 0.60 — BROAD SEARCH**: wider parameter exploration, structural
  changes permitted

This protects late-game progress (the worst possible outcome near a
converged optimum is a big speculative change that breaks something subtle)
while still letting the loop do real work early on.

### 2.5 Metric-driven hypothesis table

A lookup table from `bottleneck → root cause → tiered fixes`:

| Bottleneck | Likely cause | Tier 1 tries | Tier 2 tries |
|---|---|---|---|
| correlation < 0.95 | Mimi decoder streaming divergence | Compare per-block Mimi outputs | SEANet streaming vs batch |
| MCD high (>100) | Spectral mismatch | Mimi `output_proj` precision | SEANet residual precision |
| SNR low (<24 dB) | Noise in decoded audio | Mimi `ConvTranspose1d` | Overlap-add state |
| THD high (>30%) | Harmonic artifacts | Decoder transformer | SEANet activations |
| WER high (>0.05) | Token selection noise | Lower temperature ±0.05 | EOS threshold |

This forces the worker to pick the lowest-scoring × highest-weight component
as its target, and then draws from a vetted list rather than improvising. It
is the dumbest, most effective optimization: *stop letting the agent pick
what to work on.*

### 2.6 Git as the discard button, never as the keep button

The discard path is mechanical: `git checkout -- src/` reverts Rust source on
a regression. Critically:

- **Staging on improvements, never committing.** Improvements are `git
  add`-ed but not committed — I commit manually. This is a project policy
  that keeps human review in the loop and prevents the agent from landing an
  unexamined change into history.
- The worker may not run destructive git operations beyond `checkout --
  src/` against its own changes in this iteration.

### 2.7 Dynamic context injection via skill frontmatter

Each skill's markdown file has `!`command`` blocks in its front-matter that
execute at invocation time and inject their output into the prompt. The
worker sees:

- Current memory summary (from `memory.py`)
- `KNOWLEDGE_INDEX.md` head
- `approaches-tried.md` head
- Latest verification report
- Latest autotuning report
- Recent git log + `git describe --dirty`

This is how "fresh context every iteration" avoids being "starts from nothing
every iteration". The injected context *is* the state, curated and bounded.

---

## 3. The Supporting Cast (the part you asked me to catch)

The core worker doesn't act alone. There are four specialized agents around
it, each with one job and each communicating through files in
`docs/audit/`, never through live chat. The pattern is deliberate:
**measurement, research, cleanup and reporting are separated from doing.**

All four rotate their reports 1→2 so there's always a previous version for
diff:

```
1. If report-2.md exists → delete it
2. If report-1.md exists → rename to report-2.md
3. Write new report as report-1.md
```

### 3.1 `/verify` — the measurement auditor

**Role**: test runner and metrics reporter only. Does not touch code.

**Trigger**: after any change to `src/models/` or `src/modules/`.

**What it does**:

1. Clean release build + clippy `-D warnings`
2. Runs the standardized baseline script (`run_baseline.sh`) with the exact
   same noise files, seed, phrase and parameters every time
3. Runs quality metrics (WER, MCD, SNR, THD)
4. Runs the composite scorer
5. Runs a latency benchmark (TTFA, RTF) unless `--quick`
6. Reads the previous verification report from context and *calculates
   per-metric deltas*
7. Writes `verification-report-1.md` with a tabular summary

**Why it exists separately from `/optimize`**: the worker is biased — it just
made a change and "wants" the change to have helped. A separate measurement
pass, with its own report and its own methodology, protects against
confirmation bias and catches regressions in metrics the worker's composite
score doesn't weight heavily.

### 3.2 `/research` — the advisor with external eyes

**Role**: researcher only. Does not touch code.

**Trigger**: (a) when I'm stuck, or (b) **auto-triggered by the optimize
worker after 3 consecutive no-improvement iterations** (the "research
breakout").

**What it does**:

1. Reads `PORTING_STATUS.md`, `project-story.md`, `KNOWLEDGE_INDEX.md`,
   `approaches-tried.md`, `memory/`, latest verification report
2. Searches the web (Kyutai official docs, Moshi Rust reference
   implementation, Candle GitHub issues, similar porting efforts, HuggingFace
   discussions)
3. Validates the **methodology itself** — asks whether we're even measuring
   the right thing at the right granularity (this section alone saved this
   project once; "correlation is meaningless" was the methodology failure
   that delayed the noise-off-by-one fix)
4. Writes a briefing with High Confidence / Worth Trying / Speculative
   tiers and an explicit "Already Tried (Don't Repeat)" section

**The research breakout loop** is the key auto-triggered behavior: after the
worker plateaus, *one* research briefing is generated, *one* more iteration
is attempted using its top suggestion, and then the session exits whether or
not it worked. Bounded use of a more expensive agent.

### 3.3 `/cleanup` — the technical-debt auditor

**Role**: investigator and reporter only. Never makes changes.

**Trigger**: before commits, every few sessions.

**What it does**: greps for debug statements, commented-out code, unused
imports/functions, test artifacts, output files, duplicate v1/v2
implementations, outdated docs, debug flags. Writes a markdown report with
High/Medium/Low priority buckets and specific file:line references.

**Why a separate agent**: the worker runs under time pressure and leaves
`eprintln!`s and dump hooks behind. This is fine during optimization and
toxic at merge time. A dedicated read-only agent that never fixes anything is
the right shape — I want the *list*, not an autonomous cleanup that might
scrub a diagnostic I still need.

### 3.4 `/progress` — the dashboard

**Role**: aggregator.

**Trigger**: weekly or when I want to reorient.

**What it does**: reads all audit reports, git log, memory, and produces a
single-file dashboard (`progress-dashboard.md`, *not* rotated — it's always
the current snapshot) with:

- Correlation history table across all milestones
- Component-by-component status matrix
- Quality metrics trend vs previous
- Activity this week (commits, files most changed)
- Technical debt from latest cleanup audit
- Current research focus from latest research report
- Recommendations: immediate focus / risk area / quick wins

This is the "boss" view. It doesn't participate in the optimization loop; it
just reads its outputs.

### 3.5 The communication diagram

```
                   ┌───────────────────────┐
                   │   /loop /optimize     │
                   │      (worker)         │
                   └─────────┬─────────────┘
                             │  reads memory.json, approaches-tried.md,
                             │  KNOWLEDGE_INDEX.md, verification-report-1.md
                             │
                             │  writes: src/ edits, memory.json,
                             │          approaches-tried.md, REPORT.md
                             │
         ┌───────────────────┼────────────────────┐
         │                   │                    │
         ▼                   ▼                    ▼
   ┌───────────┐       ┌───────────┐        ┌───────────┐
   │ /verify   │       │ /cleanup  │        │ /research │◀───── auto-trigger
   │(after chg)│       │(pre-comm.)│        │ (stuck/3× │       after 3
   └─────┬─────┘       └─────┬─────┘        │  failures)│       failures
         │                   │              └─────┬─────┘
         ▼                   ▼                    ▼
   verification-        cleanup-audit-       research-advisor-
   report-{1,2}.md      report-{1,2}.md      report-{1,2}.md
         │                   │                    │
         └───────────────────┼────────────────────┘
                             ▼
                    ┌────────────────┐
                    │   /progress    │    weekly
                    └────────┬───────┘
                             ▼
                    progress-dashboard.md
```

---

## 4. The Outer Driver

`/loop <interval> /optimize` is what turns a single-shot skill into continuous
autonomous optimization. Every `<interval>` (e.g. 5m) it re-fires
`/optimize`. Each firing is independent — fresh context, disk-state read,
one iteration, disk-state write, exit.

The loop skill description is worth quoting directly because it's the
user-facing contract:

> Run a prompt or slash command on a recurring interval (e.g. `/loop 5m
> /foo`). Omit the interval to let the model self-pace.

Omitting the interval is a valid mode ("dynamic"/"self-paced"): the model
itself decides when to fire the next iteration. I've found **fixed intervals
are better for this pattern** — self-pacing lets the agent procrastinate or
over-fire, whereas a fixed cadence creates a predictable rhythm I can leave
running unattended.

---

## 5. What the Pattern Is Good For

**Works well when:**

- You have a **fast, reproducible, scalar-measurable** target
- The search space is **bounded enough** that one change at a time makes
  sense (in this project: one file edit or one parameter tweak per iteration)
- Regressions are **safely revertible** (source-only changes, no destructive
  side effects)
- You can stomach **one standardized evaluation per iteration** as the cost
  of safety

**Shines at:**

- Numerical-precision porting projects (this one) — the loop's monotonicity
  guarantee is exactly what you need when ten plausible changes each shift
  three different metrics in opposite directions
- Hyperparameter sweeps in a domain where full grid search is too expensive
- Unblocking human reviewers by turning "keep iterating on this" into a
  background process

---

## 6. Where It Breaks

These are real failure modes I hit, not hypothetical concerns.

### 6.1 The gate is only as good as the eval

The original evaluator ran without noise matching. The composite score was
essentially RNG. The loop happily "improved" by accepting noise-level deltas,
and it took a methodology-review breakout from `/research` to notice the gate
itself was broken. **Lesson:** before trusting a loop like this, verify the
eval is deterministic under zero-change by running it twice and checking
byte-identity of outputs.

### 6.2 Composite scoring can mask regressions

Correlation weighs 50%. That means a +0.04 correlation swing can hide a −30%
WER swing. I added per-metric classification
(`pure_improvement`/`mixed`/`pure_regression`) to catch this, but it's still
possible to ratchet upward on the primary while silently degrading a
secondary. **Mitigation:** the verify agent reports raw per-metric deltas
independently, and mixed results go to `promising_leads` rather than
straight to `dead_ends`.

### 6.3 Diagnostic work doesn't fit the loop

The fix that took correlation from 0.839 → 1.000 — adding a causal +
context-window attention mask to the Mimi decoder transformer — was **found
by per-sub-layer tensor dumps**, not by the loop. The loop can refine once
you know where to look; it cannot localize a cross-component bug on its own.
The dumps were built by hand, then the mechanical fix was applied and
*verified* by the loop's gate. **Lesson:** pair the loop with ad-hoc
diagnostic tooling; don't expect it to discover root causes unaided.

### 6.4 `context: fork` means no in-session learning

Everything has to round-trip through disk. If the worker "notices" something
mid-iteration, it must write it to memory before exiting or it's lost. This
is the right trade-off — it's why the loop stays coherent across hundreds of
iterations — but it forces discipline in the recording step.

### 6.5 Research breakout only fires after three failures

A bad hypothesis trajectory can burn three iterations before the breakout
kicks in. I've considered adding a "confidence score" to pre-flight so the
worker can trigger the breakout proactively on low confidence, but haven't
done it.

### 6.6 Weights are arbitrary

The composite weights (50/20/15/8/7) are defensible but not proven optimal.
Changing them mid-run invalidates historical comparisons. I treat them as
frozen for the duration of a project.

### 6.7 The pattern doesn't choose the right *problem*

It can only optimize the thing you point it at. If the metric is wrong — e.g.
early in this project, correlation was being computed without noise
alignment and was meaningless — the loop will confidently converge on
nothing. Methodology validation (via `/research`) has to be built into the
process, not an afterthought.

---

## 7. Strengths at a Glance (for the cross-project deck)

- **Monotonic-by-construction progress**: the codebase never moves backward on
  the gated metric
- **Compounds unattended**: a 5m cadence for a few hours surfaces
  non-obvious wins overnight
- **Auditable**: every iteration leaves a record in three places
  (`memory.json`, `approaches-tried.md`, `REPORT.md`)
- **Bias-resistant**: fresh context forces every hypothesis to stand on
  disk-state alone
- **Separation of concerns**: worker, measurer, researcher, janitor,
  reporter — five agents, five files, zero chat coupling
- **Human in the loop at the right spot**: improvements are staged, not
  committed; cleanup is flagged, not applied

## 8. Weaknesses at a Glance

- Requires a reproducible, fast, scalar evaluator — not universal
- Composite metric can hide regressions in unweighted dimensions
- No cross-iteration learning except through disk structures you design up
  front
- Won't localize root causes — you still need diagnostic tooling for that
- Sensitive to eval-pipeline bugs (a broken gate is worse than no gate)
- Fixed cadence can be wasteful if a compile is longer than the interval;
  self-pacing trades that for drift

---

## 9. Artifacts in This Repo (for anyone lifting the pattern)

```
.claude/skills/
├── optimize/SKILL.md          ← the worker
├── optimize/evaluate.sh       ← the standardized gate
├── verify/SKILL.md            ← the measurement auditor
├── verify/run_baseline.sh     ← reproducible baseline
├── research/SKILL.md          ← the advisor + methodology validator
├── cleanup/SKILL.md           ← the technical-debt auditor
└── progress/SKILL.md          ← the dashboard aggregator

autotuning/
├── memory.py                  ← ExperimentMemory: dead_ends, leads, ranges
├── memory.json                ← persistent structured state
├── scorer.py                  ← composite score normalization
├── results.tsv                ← flat historical log
├── REPORT.md                  ← session-level arcs
└── program.md                 ← legacy spec (keeps full methodology)

docs/
├── KNOWLEDGE_INDEX.md         ← distilled architectural lessons
├── audit/approaches-tried.md  ← human-readable attempt log
├── audit/*-report-{1,2}.md    ← rotated agent outputs
└── prompts/AGENT_ORCHESTRATION.md ← multi-agent contract
```

## 10. One-Line Definition (for the deck's TOC)

**Iterative Optimization Loop** — a cadence-driven worker that makes one
bounded change per firing, accepts it only if a hard quantitative gate
improves, discards otherwise, and is surrounded by specialized read-only
auditors (measurement, research, cleanup, progress) that communicate entirely
through disk artifacts.
