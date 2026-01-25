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
- **Correlation improved from 0.0016 to ~0.013**

The remaining issue is isolated to the **Mimi decoder's SEANet component**. The Rust implementation processes all latent frames in batch mode, producing audio with ~0.12 max amplitude vs Python's ~0.50-0.60. This 5-6x difference is caused by missing **streaming state accumulation** in the convolution layers.

Python's streaming convolutions (`StreamingConv1d`, `StreamingConvTranspose1d`) maintain state buffers across frames, implementing overlap-add for proper signal reconstruction. The Rust batch approach lacks this inter-frame accumulation.

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

## What's Next

The transformer and FlowNet are verified correct. **All 42 generated latents match Python exactly.** The remaining work is implementing streaming convolutions in the Mimi decoder:

1. **Implement streaming state for SEANet** - `previous` buffer for Conv1d, `partial` buffer for ConvTranspose1d
2. **Process latents frame-by-frame** - Instead of batch processing all at once
3. **Implement overlap-add** - Add partial buffer to left edge, store right edge for next frame
4. **Verify amplitude** - Should increase from ~0.12 to ~0.50-0.60

The implementation guidance is now comprehensive. The `docs/python-reference/STREAMING/` directory contains:
- **conv-transpose-overlap-add.md** - The exact algorithm Python uses for overlap-add
- **conv1d-streaming.md** - Causal convolution with context buffers
- **state-management.md** - How StatefulModule manages streaming state

With the validation infrastructure in place and Python behavior fully documented, testing streaming implementation is straightforward: follow the documented algorithm, run the verification agent, check amplitude improvement.

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
- `PORTING_STATUS.md` - Living technical status document
- `docs/project-story.md` - This narrative document
- `docs/quality/QUALITY_PLAN.md` - Quality infrastructure specification

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

### Audit Reports
- `docs/audit/cleanup-audit-report-1.md` - Latest cleanup findings
- `docs/audit/research-advisor-report-1.md` - Latest research briefing
- `docs/audit/verification-report-1.md` - Latest validation metrics
- `docs/audit/progress-dashboard.md` - Progress overview

---

## The Story Continues

As of January 2026, the Pocket TTS Rust port produces latents that match Python exactly. The transformer, FlowNet, and latent generation pipeline are verified correct. The remaining work is implementing streaming convolutions in the Mimi decoder to achieve proper audio amplitude.

The goal of 0.95 correlation is achievable. The infrastructure is in place. The methodology is proven. The multi-agent architecture provides fresh perspectives when needed. It's just a matter of continuing the systematic hunt.

This project also produced a reusable multi-agent collaboration pattern documented in `docs/prompts/`. Future ML porting projects can adapt these prompts and orchestration patterns for their own systematic debugging journeys.

---

*This document captures the story of the project as of 2026-01-24. For current technical status, see [PORTING_STATUS.md](../PORTING_STATUS.md). For Python reference documentation, see [docs/python-reference/](python-reference/). For agent prompts, see [docs/prompts/](prompts/). For audit reports, see [docs/audit/](audit/). For quality infrastructure, see [docs/quality/](quality/).*
