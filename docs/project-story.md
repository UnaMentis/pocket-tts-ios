# The Pocket TTS Porting Story

## A Tale of Precision, Collaboration, and Systematic Debugging

---

## The Mission

This project aims to bring Kyutai's Pocket TTS—a 117M parameter on-device text-to-speech model—from Python/PyTorch to Rust/Candle for native iOS deployment. The goal isn't just "working" audio; it's **near-identical waveform output** (correlation > 0.95) compared to the Python reference.

Why such a high bar? Because close isn't good enough. A model that produces *different* audio might sound acceptable, but it could exhibit edge-case failures, mispronunciations, or artifacts that the original doesn't. True fidelity means the Rust port can be trusted as a drop-in replacement.

---

## The Challenge

Porting an ML model between frameworks sounds straightforward: translate the architecture, load the weights, run inference. In practice, it's treacherous:

- **Numerical precision differs.** PyTorch and Candle don't compute matrix multiplications in the same order. Floating-point addition isn't associative. Tiny errors accumulate across 6 transformer layers, 6 residual blocks, and temporal upsampling—until the output is unrecognizable.

- **Implicit conventions vary.** Does RoPE use interleaved or split-half format? Does LayerNorm use epsilon 1e-5 or 1e-6? Is the activation ELU or GELU? These aren't always documented. They're discovered by comparing bytes.

- **Architecture documentation is sparse.** The reference implementation works, but *why* it works requires reverse-engineering. Voice conditioning concatenates embeddings in a specific order. The FlowNet averages two time embeddings for Lagrangian Self-Distillation. None of this is obvious from reading code.

To combat this, the project invested heavily in **documenting the Python reference implementation** itself. The `docs/python-reference/` directory contains comprehensive documentation extracted directly from the Python source, including streaming convolution algorithms, state management patterns, and layer-by-layer architecture details. This "source-of-truth" documentation became essential—when stuck, the answer was usually already documented locally.

---

## The Multi-Agent Collaboration Model

Rather than a single agent grinding through an endless debugging session, this project employs a **multi-agent collaboration pattern** with five distinct roles. This architecture evolved during the project, starting with three agents and expanding to five as new needs emerged.

### Agent Overview

```
                         ┌─────────────────────┐
                         │   Implementation    │
                         │       Agent         │
                         │  (primary worker)   │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   Verification  │   │     Cleanup     │   │    Research     │
    │      Agent      │   │     Auditor     │   │     Advisor     │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │   Progress Tracker  │
                         └─────────────────────┘
```

### The Implementation Agent

The primary workhorse. This agent lives in the code—writing Rust, running tests, adding debug instrumentation, comparing tensor values. It maintains a running log of discoveries in `PORTING_STATUS.md`, documenting what's been tried, what's been fixed, and what still diverges.

The implementation agent is allowed to be messy. It leaves `eprintln!` statements everywhere. It creates temporary Python scripts. It modifies test outputs. This is intentional: when hunting for a 5th-decimal-place discrepancy, you need instrumentation, not cleanliness.

### The Cleanup Auditor

A fresh Claude session, invoked periodically to survey the damage. The Cleanup Auditor reads the codebase with no prior context—a **fresh pair of eyes**—and catalogs:

- Debug statements that should eventually be removed
- Unused imports and dead code
- Temporary files that shouldn't be committed
- Documentation drift

Critically, the Cleanup Auditor **doesn't fix anything**. It produces a structured report (`docs/audit/cleanup-audit-report-1.md`) that the human can review. This separation of concerns prevents premature cleanup (removing instrumentation that's still needed) while ensuring nothing is forgotten.

### The Research Advisor

When the implementation agent gets stuck—truly stuck, not just confused—a separate Claude session runs as a Research Advisor. This agent:

1. **Reads the current state** from `PORTING_STATUS.md`
2. **Researches external sources**: Kyutai's official docs, reference implementations, Candle GitHub issues, academic papers
3. **Provides structured suggestions** ranked by confidence level

The Research Advisor doesn't touch code. It produces a briefing (`docs/audit/research-advisor-report-1.md`) with hypotheses to test, approaches to try, and links to relevant resources.

### The Verification Agent (New)

Added to close the feedback loop between code changes and numerical accuracy. After the implementation agent makes changes, the Verification Agent:

1. **Builds the Rust binary** and checks for warnings
2. **Runs the test harness** with a standard test phrase
3. **Compares against Python reference** outputs
4. **Reports metrics with deltas** from the previous run

This is the "Critic" in a Generator-Critic pattern. It catches regressions immediately and provides quantitative feedback: "Correlation improved from 0.01 to 0.013" or "Sample count increased by 500—investigate."

### The Progress Tracker (New)

A weekly dashboard agent that aggregates metrics over time:

- Correlation history showing improvement trajectory
- Issues fixed vs remaining
- Velocity (fixes per week)
- Time-to-target estimates (with appropriate caveats about ML debugging unpredictability)

This agent serves both morale and planning purposes. When you've been debugging for weeks, seeing "14 issues fixed, correlation improved 8x from initial" provides motivation.

### Orchestration and Communication

All agents communicate through **artifacts, not real-time chat**. Each agent produces a structured report in `docs/audit/`. The implementation agent reads these reports to inform its work. This async collaboration pattern allows:

- Parallel investigation streams
- Fresh perspectives without context contamination
- An audit trail of decisions and suggestions

The full orchestration pattern is documented in `docs/prompts/AGENT_ORCHESTRATION.md`.

---

## The Validation Framework

How do you know if a port is correct? This project developed a **three-layer validation approach**:

### Layer 1: Numerical Accuracy

Compare tensors directly:
- **Latent cosine similarity > 0.99**: The internal representations should be nearly identical
- **Audio waveform correlation > 0.95**: The final output should strongly correlate, with tolerance for floating-point differences

These thresholds are strict. A 0.90 correlation might sound fine but could hide systematic errors.

### Layer 2: Intelligibility (ASR Round-Trip)

Run both Python and Rust outputs through Whisper speech-to-text. Compare Word Error Rates. If Rust produces 5% more transcription errors than Python, something is wrong—even if the waveforms look plausible.

This catches a class of bugs that numerical metrics miss: audio that's internally consistent but semantically degraded.

### Layer 3: Signal Health

Sanity checks on the output:
- No NaN or Inf values
- Amplitude in audible range (0.01-1.0)
- DC offset < 0.05
- RMS level computed and logged

These catch catastrophic failures early.

### The Validation Harness

All three layers are orchestrated by `validation/run_tests.sh`:

```bash
./validation/run_tests.sh              # Full suite
./validation/run_tests.sh --quick      # Skip ASR (faster iteration)
./validation/run_tests.sh --regen-ref  # Regenerate Python reference
```

The script builds Rust in release mode, runs Python reference generation, compares outputs, and produces a JSON report. It's designed for rapid iteration: make a change, run the script, see the impact.

---

## The Quality Infrastructure

Beyond validation, the project established a comprehensive quality framework documented in `docs/quality/QUALITY_PLAN.md`. This infrastructure ensures code quality doesn't degrade during active debugging.

**Implementation Status:** ✅ Fully implemented as of January 2026. All hooks, workflows, and configurations are active.

### Pre-Commit Hooks

Local development quality gates:
- **rustfmt** - Code formatting consistency
- **clippy** - Lint with strict warnings
- **gitleaks** - Secrets detection
- **Quick tests** - Fast unit test suite

The hooks log all runs for audit purposes, creating accountability without blocking rapid iteration.

### CI/CD Pipelines

GitHub Actions workflows for:
- **Rust CI** - Lint, test, coverage, build
- **Security** - Dependency audit, secrets scan
- **iOS Build** - XCFramework generation
- **Validation** - Python reference comparison
- **Documentation** - Markdown linting, link checking, rustdoc

### What's Implemented

| Component | Status | Location |
|-----------|--------|----------|
| Pre-commit hooks | ✅ | `.hooks/pre-commit`, `.hooks/pre-push` |
| CI/CD pipelines | ✅ | `.github/workflows/` (6 workflows) |
| Linting config | ✅ | `rustfmt.toml`, `Cargo.toml` |
| Code coverage | ✅ | `codecov.yml` |
| AI review | ✅ | `.coderabbit.yaml` |
| CONTRIBUTING.md | ⏳ | Planned |
| Testing strategy doc | ⏳ | Planned |

### Philosophy: Quality Infrastructure Shouldn't Block Debugging

During active ML debugging, you need to experiment freely. The quality infrastructure is designed to:

1. **Warn, not block** - Pre-commit hooks for critical issues, CI for comprehensive checks
2. **Track technical debt** - The Cleanup Auditor catalogs mess without forcing cleanup
3. **Maintain accountability** - Hook logs track what checks ran (or were bypassed)
4. **Enable cleanup later** - When the port is correct, clean up systematically

---

## The Debugging Journey

### Where We Started

The first Rust synthesis attempt produced correlation **0.0016**—essentially random noise relative to the Python reference. Sample counts were wrong. Amplitudes were off by 10x. Something was fundamentally broken.

### The Discovery Process

Rather than guessing, the approach was systematic: compare intermediate tensors, layer by layer, until divergence is found.

This required building comparison infrastructure:
- `validation/dump_intermediates.py`: Export Python tensors at each layer
- `test-tts --export-latents`: Export Rust tensors in numpy format
- `validation/compare_intermediates.py`: Compute cosine similarity, RMSE, max error

Then, methodically:

1. **Tokenization** (Issue #1): Rust produced 32 tokens, Python produced 17. The tokenizer was wrong. Fixed by switching to SentencePiece.

2. **RoPE format** (Issue #2): Rust used split halves `[first D/2, last D/2]`, Python uses interleaved pairs. Discovered by comparing query vectors after RoPE—they were completely different.

3. **LayerNorm vs RMSNorm** (Issue #3): The model weights have bias terms, which RMSNorm ignores. Python uses LayerNorm. Discovered by checking weight shapes.

4. **Voice conditioning** (Issues #7-8): Rust added voice embeddings; Python concatenates them. And the order matters—voice first, then text, in separate forward passes. This required reading the Python source carefully, not just matching shapes.

5. **FlowNet architecture** (Issues #4-5): Multiple subtle differences—sinusoidal embedding order, MLP activation (SiLU not GELU), AdaLN chunk order, time embedding averaging for LSD.

6. **Latent handling** (Issues #12-14): FlowNet RMSNorm needed proper variance calculation, time embeddings should only add to conditioning (not input), denormalization happens before Mimi (not in the generation loop).

And so on through **14 major issues**, each documented in `PORTING_STATUS.md` with the problem, the fix, and verification that it helped.

### The Current State (Updated January 2026)

After all fixes:
- **All 42 generated latents match Python with cosine similarity = 1.0**
- **Transformer output matches Python to 5+ decimal places** for all phases
- **Sample count matches exactly** (was off by thousands)
- **Short phrase waveform correlation: 0.81** (dramatically improved from initial 0.0016)

The project has achieved a major milestone: **the first public beta release (v0.4.0)** was published on January 24, 2026. Short phrases work excellently with 0.81 correlation—nearly indistinguishable from Python's output. Medium and long phrases (up to ~25 seconds) also generate intelligible audio.

The remaining challenge is **EOS detection divergence for longer sequences**. For phrases longer than ~17 tokens, the Rust implementation detects end-of-speech slightly earlier than Python due to numerical precision accumulation across many transformer layers. This is a refinement issue, not a fundamental architecture problem.

---

## The Techniques That Worked

### 1. Immutable Tracking Document

`PORTING_STATUS.md` is the single source of truth. Every fix gets documented with:
- What the problem was
- What the solution was
- How it was verified
- Quantitative impact (before/after metrics)

This prevents re-discovering the same issue. It also enables handoff—a new Claude session can read the document and understand the full history without starting from scratch.

### 2. Fresh-Eyes Rotation

The auxiliary agents (Cleanup Auditor, Research Advisor, Verification Agent, Progress Tracker) are designed for **fresh Claude sessions with no prior context**. This isn't a limitation—it's a feature. Fresh eyes notice things that a fatigued agent misses.

The prompts are stored in `docs/prompts/` and checked into version control. Anyone (human or AI) can invoke them reproducibly.

### 3. Confidence-Graded Suggestions

The Research Advisor categorizes suggestions as:
- **High Confidence**: Backed by documentation or proven solutions
- **Worth Trying**: Reasonable hypotheses
- **Speculative**: Long shots

This prevents the implementation agent from treating every idea equally. It should try high-confidence approaches first.

### 4. Parallel Investigation Streams

Multiple agents can work simultaneously:
- Implementation agent debugs current blocker
- Research Advisor investigates Candle precision issues
- Cleanup Auditor catalogs accumulated debug code
- Progress Tracker aggregates weekly metrics

They communicate through artifacts (reports, documents), not real-time chat. This is async collaboration at its best.

### 5. Hypothesis-Driven Debugging

Rather than randomly trying things, each debugging session has explicit hypotheses:
- "KV cache values from voice phase may be incorrect"
- "Attention softmax precision differs"
- "Scale factor computed differently"

These are listed in `PORTING_STATUS.md` with current status. When a hypothesis is ruled out, it's documented—preventing future agents from testing it again.

### 6. Generator-Critic Pattern

The Verification Agent implements a feedback loop: after the Implementation Agent generates code changes (Generator), the Verification Agent validates them (Critic). This catches regressions immediately and provides quantitative progress tracking.

### 7. Quantitative Everything

Every debugging session produces numbers:
- Cosine similarity between tensor pairs
- Correlation between waveforms
- Sample counts, amplitudes, RMS levels
- Deltas from previous runs

"It's still wrong" isn't actionable. "Correlation improved from 0.0016 to 0.01" is progress, even if not victory.

### 8. Document the Reference Implementation

Rather than repeatedly diving into undocumented Python source code, the project created **comprehensive documentation of the Python reference** in `docs/python-reference/`. This includes:

- **Streaming algorithms** - The critical overlap-add algorithm for ConvTranspose1d, causal convolution patterns
- **Architecture details** - Layer-by-layer breakdown of SEANet, Mimi, FlowLM, FlowNet
- **State management** - How Python maintains buffers across frames
- **Module implementations** - RoPE, LayerNorm, MLP variants

This investment paid dividends: when the Research Advisor or Implementation Agent needs to understand a component, the answer is already documented locally. No need for external searches or re-reading source code. The Python reference documentation became the authoritative source for "how does Python actually do this?"

---

## The First Release: v0.4.0 Beta

On January 24, 2026, the project achieved its first public release milestone: **Pocket TTS iOS v0.4.0 (Beta)**.

### What's Included

- **Complete TTS pipeline**: FlowLM transformer (~70M params), MLP consistency sampler (~10M params), Mimi VAE decoder (~20M params)
- **8 built-in voices**: Alba, Marius, Javert, Jean, Fantine, Cosette, Eponine, Azelma
- **iOS XCFramework**: Universal binary for device (arm64) and simulator (arm64-sim)
- **Swift bindings**: UniFFI-generated bindings with high-level async/await wrapper
- **Streaming synthesis**: Overlap-add implementation for low-latency playback

### Accuracy Achieved

| Phrase Type | Correlation | Status |
|-------------|-------------|--------|
| Short (~17 tokens) | **0.81** | ✅ Excellent - nearly identical to Python |
| Medium (~50 tokens) | Working | ⚠️ Intelligible, EOS timing differs |
| Long (~100+ tokens) | Working | ⚠️ Up to ~25 seconds per synthesis |

The 0.81 correlation for short phrases represents a **500x improvement** from the initial 0.0016 correlation. The audio is intelligible and sounds nearly identical to Python's output.

### What Made It Possible

The release was enabled by systematic debugging across **14 major issues**:

1. Tokenization format (Character → SentencePiece)
2. RoPE interleaved vs split-half format
3. LayerNorm vs RMSNorm selection
4. FlowNet sinusoidal embedding order
5. MLP activation function (SiLU vs GELU)
6. AdaLN chunk ordering
7. LSD time progression averaging
8. SEANet activation function (ELU not GELU)
9. Voice conditioning concatenation
10. Voice conditioning sequence ordering
11. FinalLayer missing normalization
12. SEANet output activation (removed tanh)
13. FlowNet TimeEmbedding RMSNorm
14. Latent denormalization location

Each fix was documented in `PORTING_STATUS.md` with before/after metrics, creating an audit trail that prevents re-discovering the same issues.

---

## What's Next

The architecture is verified correct. **All latents match Python exactly (cosine similarity = 1.0).** The Mimi decoder streaming implementation is complete and working. The remaining work focuses on EOS detection refinement:

### Primary Blocker: EOS Detection for Longer Phrases

For phrases longer than ~17 tokens, Rust detects end-of-speech earlier than Python:
- **Root cause**: Numerical precision accumulation over many transformer forward passes
- **Impact**: Medium phrases (~50 tokens) generate 21 fewer frames (~1.7 seconds shorter)
- **Workaround available**: Short phrases work excellently; longer content can be chunked at application layer

### Recommended Investigation Path

1. **Log EOS trajectories** - Compare step-by-step EOS logits between Rust and Python
2. **Identify divergence point** - Linear drift vs sudden jump indicates different root causes
3. **Apply targeted precision fixes** - Force Float32 in attention operations if needed
4. **Fallback option** - Accept short-phrase optimization for iOS (notifications, UI feedback)

### Practical Capabilities Today

| Feature | Status |
|---------|--------|
| Short phrases (<5s) | ✅ 0.81 correlation |
| Medium phrases (5-15s) | ✅ Working, intelligible |
| Long paragraphs (15-25s) | ✅ Working, up to 284 latent frames |
| Very long content (>40s) | ⚠️ Requires chunking (max ~512 latents) |

The implementation is **production-ready for iOS use cases** that primarily involve short to medium utterances.

---

## Lessons Learned

1. **Numerical equivalence is hard.** Even "simple" operations like matrix multiplication can differ between frameworks. Plan for layer-by-layer comparison from the start.

2. **Fresh eyes matter.** A new session with no baggage often spots what hours of debugging missed. Build this into your workflow with scheduled agent rotations.

3. **Document obsessively.** The tracking document isn't overhead—it's the difference between progress and circles.

4. **Separate concerns.** The agent doing implementation shouldn't also worry about cleanup, research, or metrics. Let each agent focus on one job.

5. **Quantify everything.** "It's still wrong" isn't actionable. "Correlation improved from 0.0016 to 0.01" is progress.

6. **Trust the process.** With 14 issues fixed and more to find, the approach is working. Stay systematic.

7. **Evolve the architecture.** The multi-agent pattern started with 3 agents and grew to 5 as new needs emerged. Don't be afraid to add agents when you notice gaps.

8. **Quality infrastructure shouldn't block debugging.** Set up hooks and CI, but design them to track technical debt rather than prevent it during active development.

9. **Patterns are reusable.** The multi-agent orchestration, tracking documents, and prompt templates can be extracted for future ML porting projects.

10. **Document the reference implementation.** When porting between frameworks, invest time documenting how the source implementation actually works. This pays dividends throughout the project and reduces context-switching.

---

## Project Artifacts

### Documentation
- `CLAUDE.md` - Project instructions and quick reference
- `PORTING_STATUS.md` - Living technical status document (14 issues fixed, EOS remaining)
- `CHANGELOG.md` - Release notes and version history
- `docs/project-story.md` - This narrative document
- `docs/quality/QUALITY_PLAN.md` - Quality infrastructure specification
- `docs/RELEASE_PROCESS.md` - Release automation documentation

### Python Reference Documentation
- `docs/python-reference/README.md` - Navigation and status overview
- `docs/python-reference/STREAMING/` - Critical streaming convolution algorithms
- `docs/python-reference/ARCHITECTURE/` - SEANet, Mimi, FlowLM, FlowNet details
- `docs/python-reference/MODULES/` - RoPE, LayerNorm, MLP, Transformer
- `docs/python-reference/gap-analysis.md` - Questions for upstream clarification

### Agent Prompts
- `docs/prompts/cleanup-audit.md` - Technical debt inventory
- `docs/prompts/research-advisor.md` - External research and hypotheses
- `docs/prompts/verification-agent.md` - Numerical accuracy validation
- `docs/prompts/progress-tracker.md` - Progress dashboard
- `docs/prompts/AGENT_ORCHESTRATION.md` - Full orchestration guide

### Audit Reports (Latest: 2026-01-24)
- `docs/audit/cleanup-audit-report-2.md` - Technical debt inventory (70+ debug statements)
- `docs/audit/research-advisor-report-2.md` - EOS divergence investigation
- `docs/audit/verification-report-2.md` - 0.69 Mimi correlation, 0.81 short phrase

### Release Artifacts
- `.claude/commands/release.md` - Release automation skill
- `scripts/build-ios.sh` - XCFramework build script
- `scripts/package-release.sh` - Release packaging script

---

### v0.4.1: Streaming Quality Fixes & API Cleanup (January 27, 2025)

Following user reports that streaming mode sounded "wonky" compared to sync mode, investigation revealed three issues:

1. **Broken crossfade logic** - The streaming path applied artificial crossfade at chunk boundaries, blending different audio content together (end of chunk A with start of chunk B). This was removed entirely since Mimi's streaming ConvTranspose1d state already handles continuity via persistent partial buffers.

2. **Callback termination bug** - The callback returned `Stop` on EOS, cutting off the `frames_after_eos` padding needed for natural audio endings. Fixed to return `Continue` and let the generation loop handle termination.

3. **Premature EOS detection** - `min_gen_steps = 3` prevented short phrases from detecting EOS naturally. Changed to 0.

Additionally, the **legacy token-chunked streaming API was removed** (`start_streaming`, `synthesize_streaming`). This method provided no unique value over the other two modes:
- Higher latency than true streaming (chunked by tokens)
- Lower quality than sync mode (artificial boundaries)
- Confusing API with two "streaming" methods

**v0.4.1 API (streamlined to two methods):**
- `synthesize()` - Sync mode for batch processing and single file generation
- `start_true_streaming()` - True streaming (~200ms TTFA, **preferred for on-device**)

---

## The Autotuning System (March 2026)

With the TTS pipeline stable and producing intelligible audio, attention shifted from *implementation correctness* to *systematic quality optimization*. The project needed a way to explore the configuration space methodically rather than through one-off experiments.

### The Autoresearch Architecture

Inspired by Karpathy's autoresearch pattern, the project built a complete autotuning infrastructure:

- **`autotune.py`** — Orchestrator implementing the core loop: modify → evaluate → keep/discard → repeat. Supports three phases: baseline establishment, single-parameter sweeps, and joint optimization with adaptive perturbation.
- **`scorer.py`** — Composite quality scoring reducing five metrics (WER, MCD, SNR, THD, correlation) to a single scalar in [0, 1]. Normalization curves calibrated from actual Rust-vs-Python TTS comparison data.
- **`memory.py`** — Persistent experiment memory tracking dead ends, promising leads, safe parameter ranges, interaction rules, and sensitivity rankings. Prevents the agent from repeating failed experiments and guides it toward productive exploration.

The system uses a 4-phrase test corpus aligned with Python reference audio, enabling MCD (spectral similarity) and correlation measurements against ground truth. A memory-bootstrapping script pre-loads project history so the agent starts with accumulated knowledge rather than from scratch.

This represented a sixth agent role in the multi-agent architecture: the **Autotuning Agent**, operating autonomously within a structured feedback loop.

### The Deterministic Seeding Fix

Building the autotuning system immediately exposed a critical problem: **results weren't reproducible.** Running the same configuration twice produced different audio, different metrics, and different scores. The autotuner couldn't distinguish parameter changes from random noise.

Investigation revealed that the `--seed` CLI parameter was a **complete no-op**. It was parsed, stored in `TTSConfig`, and never used. The seed field existed since the beginning of the project—it just never reached `FlowNet::generate()`, where `Tensor::randn()` used Candle's unseeded global RNG.

The fix required working around a Candle 0.8.4 limitation: no per-call seed API for `Tensor::randn()`. The solution used Rust's `rand` crate with `SeedableRng`:

```rust
let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
let normal = Normal::new(0.0f32, std);
let values: Vec<f32> = (0..n_elements).map(|_| normal.sample(&mut rng)).collect();
Tensor::from_vec(values, shape, device)?
```

Each generation step derives a unique-but-deterministic seed via `seed.wrapping_add(step as u64)`, ensuring different noise at each step while maintaining full reproducibility.

**Verification**: Two runs with `--seed 42` now produce byte-identical WAV files. Without `--seed`, the system still uses Candle's default unseeded RNG. This was the necessary foundation for reliable autotuning—and, as it turned out, for something much more important.

---

## Losing the Plot (March 2026)

This section documents a wrong turn that lasted six weeks—and the course correction that followed. It's included not despite being a mistake, but because recognizing and correcting it was one of the project's most important milestones.

### How the Primary Metric Was Abandoned

The project's mission statement, written at the very beginning, was clear:

> *"The goal isn't just 'working' audio; it's **near-identical waveform output** (correlation > 0.95) compared to the Python reference."*

The project reached **0.81 correlation** at v0.4.0—a 500x improvement from the initial 0.0016, and tantalizingly close to the 0.95 target. Then v0.4.1 enabled random noise in FlowNet to match Python's production behavior. Correlation dropped to approximately zero.

The response was not to fix the noise mismatch. Instead, correlation was declared **"CLOSED — METRIC NO LONGER APPLICABLE"** in PORTING_STATUS.md. The justification: "With random noise enabled, waveform correlation is no longer a meaningful metric since different random number generators produce different (but equally valid) latent trajectories."

This reasoning was locally sound—Python uses PyTorch's Mersenne Twister RNG, Rust uses Candle's thread-local RNG, and they produce incompatible noise sequences. But the conclusion was wrong. Rather than asking "how do we match the noise?", the project asked "how do we score without correlation?"

### The Cascade

Each subsequent step made sense in isolation but drifted further from the mission:

| Date | Correlation | What Happened |
|------|-------------|---------------|
| Jan 24, 2026 | **0.81** | v0.4.0 Beta — peak correlation |
| Jan 27, 2026 | ~0.0 | v0.4.1 enabled random noise, declared correlation "CLOSED" |
| Mar 12, 2026 | ~0.0 | Built autotuning scoring system *without* correlation |
| Mar 13, 2026 (AM) | ~0.0 | Removed correlation from composite scoring entirely |

The autotuning system was built to optimize perceptual quality metrics—WER (intelligibility), MCD (spectral similarity), SNR (signal quality), THD (distortion)—treating them as primary objectives rather than diagnostics. Temperature sweeps, consistency step experiments, and joint optimization runs accumulated. Scores improved from 0.46 to 0.73 through WER normalization fixes and MCD recalibration.

But these improvements were optimizing the wrong thing. The system was getting better at *sounding reasonable* while making no progress toward *matching the reference implementation*.

### The Course Correction

The turning point came from a direct challenge: *why should we accept near-zero correlation?*

The argument was simple and devastating:

- **If correlation = 1.0, every other metric is automatically perfect.** You've matched the reference. WER, MCD, SNR, THD are identical by definition.
- **If correlation < 1.0, the other metrics tell you WHERE you're diverging.** They're diagnostic tools, not objectives.
- **You can have perfect WER, good MCD, good SNR, and still be producing completely different audio.** The other metrics can pass while the implementation is fundamentally wrong.
- **Removing the metric that shows the problem doesn't fix the problem.**

This insight reframed everything. Correlation isn't one metric among many—it's THE metric. The others exist to explain *why* correlation falls short, not to replace it.

### The Root Cause Is Solvable

The ~0.0 correlation isn't an inherent limitation. It's a noise mismatch:

1. The reference audio in `validation/reference_outputs/` was generated by `reference_harness.py` with **no seed**—Python's unseeded `torch.randn()` produced the noise
2. Rust uses a completely different RNG (Candle's `Tensor::randn()` or `StdRng`)
3. Different noise → different latent trajectories → different waveforms → correlation ≈ 0

The fix: capture the exact noise tensors Python used during reference generation, save them as artifacts, and load them in Rust instead of generating new noise. This eliminates the RNG difference entirely, and any remaining correlation gap measures pure implementation differences—which is exactly what we want to know.

With deterministic seeding already implemented, the infrastructure is in place. The path from here:

1. **Re-generate Python references with noise tensor dumps** — hook FlowNet to save per-step noise as `.npy` files
2. **Add noise-loading mode to Rust's FlowNet** — load pre-generated noise instead of sampling
3. **Measure correlation with matched noise** — should return to ~0.81 range
4. **Systematically fix remaining divergences** — frame count, amplitude, precision accumulation
5. **Each fix produces a measurable correlation improvement** — back to the systematic methodology that worked before

### What This Means for Autotuning

The autotuning system isn't wasted work—it just needs to be resequenced:

- **Phase 1 (now)**: Implementation correctness — maximize correlation with matched noise
- **Phase 2 (after correlation > 0.95)**: Quality optimization — tune parameters for best audio quality
- **Phase 3 (production)**: Deployment optimization — tune for iOS-specific constraints

The infrastructure (memory system, composite scoring, parameter sweeps) is all reusable. The scoring weights just need to be restructured with correlation as the dominant signal.

---

## The Story Continues

As of March 2026, Pocket TTS iOS is in an interesting position. The TTS pipeline is production-ready—audio is intelligible, streaming works with ~200ms TTFA, and the iOS XCFramework is deployed. But the project's original mission of > 0.95 waveform correlation was prematurely closed rather than completed.

The detour through perceptual-metric autotuning wasn't wasted. It produced real infrastructure (the autoresearch loop, experiment memory, quality metrics) and real insights (temperature 0.3 minimizes distortion, consistency steps have zero impact, WER normalization matters). But the most valuable outcome was the recognition that **the project had drifted from its primary objective, and the courage to correct course.**

**What's been accomplished:**
- All 14 major architectural issues identified and fixed
- Latents match Python exactly (cosine similarity = 1.0)
- Full streaming Mimi decoder with replicate padding
- Production-quality iOS XCFramework with 8 voices
- Deterministic seeding for reproducible results
- Complete autotuning infrastructure with persistent experiment memory
- Honest recognition of a wrong turn and clear path back

**What was built (same day):**
- Noise tensor capture infrastructure in `reference_harness.py` (`--capture-noise --seed 42`)
- Noise tensor loading in Rust (`FlowNet::generate()` accepts `noise_override`)
- Full pipeline threading through FlowLM and PocketTTSModel
- Test binary `--noise-dir` flag for correlation testing
- Correlation restored as primary metric in `scorer.py` (50% weight)
- Waveform correlation reopened in `PORTING_STATUS.md`

**What remains:**
- Run reference harness with `--capture-noise --seed 42` on a machine with the Python model
- Use captured noise tensors to re-establish ~0.81 correlation baseline
- Fix frame count mismatch (43 vs 45 frames)
- Fix amplitude mismatch (59-77% of Python reference)
- Push toward 0.95 correlation target
- Then: quality optimization with autotuning

**Key metrics from the journey:**

| Milestone | Correlation | Date |
|-----------|-------------|------|
| Initial attempt | 0.0016 | January 2026 |
| After 14 fixes | 0.013 | January 2026 |
| With streaming Mimi | 0.64 | January 2026 |
| With replicate padding | **0.81** | January 2026 |
| **v0.4.0 Beta Release** | **0.81** | January 24, 2026 |
| v0.4.1 (noise enabled) | ~0.0 | January 27, 2026 |
| Autotuning v1 (correlation removed) | ~0.0 | March 12, 2026 |
| **Course correction** (correlation reinstated) | — | March 13, 2026 |
| **Noise capture infrastructure built** | — | March 13, 2026 |

This project also produced a reusable multi-agent collaboration pattern documented in `docs/prompts/`. The six-agent architecture (Implementation, Cleanup Auditor, Research Advisor, Verification, Progress Tracker, Autotuning) can be adapted for future ML porting projects.

---

*This document captures the story of the project as of 2026-03-13. For current technical status, see [PORTING_STATUS.md](../PORTING_STATUS.md). For Python reference documentation, see [docs/python-reference/](python-reference/). For agent prompts, see [docs/prompts/](prompts/). For audit reports, see [docs/audit/](audit/). For quality infrastructure, see [docs/quality/](quality/). For changelog, see [CHANGELOG.md](../CHANGELOG.md).*
