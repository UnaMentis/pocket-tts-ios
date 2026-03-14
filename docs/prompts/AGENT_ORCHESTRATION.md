# Multi-Agent Orchestration Guide

This document describes the multi-agent collaboration pattern used for the Pocket TTS Rust porting project. This pattern is designed to be reusable for similar ML porting work.

---

## Overview

Instead of a single agent grinding through endless debugging, this project uses **specialized agents** with distinct roles. Each agent runs in a fresh session, produces a structured artifact, and communicates through files rather than real-time chat.

### Why Multi-Agent?

1. **Fresh Eyes** - New sessions avoid confirmation bias and tunnel vision
2. **Separation of Concerns** - Implementation doesn't worry about cleanup
3. **Parallel Work** - Research can happen while implementation continues
4. **Quantitative Tracking** - Metrics over feelings
5. **Reproducibility** - Prompts are versioned, results are documented

---

## Agent Registry

| Agent | Prompt File | Output Location | Frequency |
|-------|-------------|-----------------|-----------|
| Implementation | (human session) | Code, PORTING_STATUS.md | Active development |
| Cleanup Auditor | [cleanup-audit.md](cleanup-audit.md) | audit/cleanup-audit-report-1.md | Every 2-3 sessions |
| Research Advisor | [research-advisor.md](research-advisor.md) | audit/research-advisor-report-1.md | When stuck / daily |
| Verification | [verification-agent.md](verification-agent.md) | audit/verification-report-1.md | After code changes |
| Progress Tracker | [progress-tracker.md](progress-tracker.md) | audit/progress-dashboard.md | Weekly |
| Autotuning | [../../autotuning/program.md](../../autotuning/program.md) | autotuning/REPORT.md, memory.json | On demand / continuous |

---

## Communication Flow

```
                         ┌─────────────────────┐
                         │   Implementation    │
                         │       Agent         │
                         │  (your main work)   │
                         └──────────┬──────────┘
                                    │
                    produces: Code, PORTING_STATUS.md
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   Verification  │   │     Cleanup     │   │    Research     │
    │      Agent      │   │     Auditor     │   │     Advisor     │
    │  (after changes)│   │ (before commit) │   │   (when stuck)  │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ verification-   │   │ cleanup-audit-  │   │ research-       │
    │ report-1.md     │   │ report-1.md     │   │ advisor-        │
    │                 │   │                 │   │ report-1.md     │
    └─────────────────┘   └─────────────────┘   └─────────────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                          consumed by
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Progress Tracker  │
                         │     (weekly)        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ progress-           │
                         │ dashboard.md        │
                         └─────────────────────┘
```

---

## When to Run Each Agent

### Verification Agent
- **Trigger:** After any change to files in `src/models/` or `src/modules/`
- **Duration:** ~2-3 minutes
- **Purpose:** Catch numerical regressions immediately
- **Read output:** Before continuing implementation

### Cleanup Auditor
- **Trigger:** Before commits, after 3+ development sessions
- **Duration:** ~5 minutes
- **Purpose:** Track technical debt without blocking development
- **Read output:** Decide what to clean up before committing

### Research Advisor
- **Trigger:** When stuck for >1 hour, or daily for fresh perspective
- **Duration:** ~10-15 minutes (includes web search)
- **Purpose:** Break through blockers with external research
- **Read output:** Try suggested approaches in implementation

### Progress Tracker
- **Trigger:** Weekly, or when you need to assess progress
- **Duration:** ~5 minutes
- **Purpose:** Motivation, timeline estimation, stakeholder updates
- **Read output:** Plan next week's focus

### Autotuning Agent
- **Trigger:** When quality metrics plateau or after major code changes
- **Duration:** Runs indefinitely (autonomous loop)
- **Purpose:** Iteratively optimize TTS quality via autoresearch-style modify → evaluate → keep/discard loop
- **Setup:** Run on a dedicated branch. Start with `--phase baseline` to verify prerequisites
- **Read output:** Check `autotuning/REPORT.md`, `memory.json`, and git log for accumulated improvements
- **Details:** See [autotuning/README.md](../../autotuning/README.md) for full usage

---

## Running an Agent

### Using Skills (Preferred)

Agents are now available as Claude Code skills (slash commands). Just type:

- `/verify` — Run verification tests and report metrics
- `/research` — Deep research and fresh perspective on blockers
- `/research transformer precision` — Focused research on a specific topic
- `/cleanup` — Audit for debug leftovers and technical debt
- `/progress` — Generate progress dashboard

Skills live in `.claude/skills/` and inject dynamic context automatically.

### Legacy: Manual Process

The original prompts in `docs/prompts/` still work if you prefer fresh-session isolation:

1. **Start a fresh Claude Code session**
2. **Paste the prompt** from the relevant file in `docs/prompts/`
3. **Let it run to completion**
4. **The report will be saved** to `docs/audit/`

### Quick Commands

```bash
# Check latest reports
ls -la docs/audit/

# View a report
cat docs/audit/verification-report-1.md
```

---

## Report Rotation

Most agents use a 2-version rotation to prevent unbounded growth while preserving history:

- `-1.md` = most recent report
- `-2.md` = previous report (for comparison)

### Rotation Logic (built into each prompt)

```
1. If report-2.md exists, delete it
2. If report-1.md exists, rename to report-2.md
3. Write new report as report-1.md
```

### Exception

The **Progress Dashboard** uses a single file (`progress-dashboard.md`) that gets overwritten each time, since it's a point-in-time snapshot rather than a comparison tool.

---

## Cross-Agent Context

Agents are designed to read each other's outputs:

| Agent | Reads From |
|-------|------------|
| Research Advisor | PORTING_STATUS.md, verification-report-1.md |
| Progress Tracker | All reports, PORTING_STATUS.md, git log |
| Cleanup Auditor | Git diff, PORTING_STATUS.md |
| Verification | Previous verification report |

This creates a feedback loop where research is informed by metrics, and progress tracking aggregates everything.

---

## Adapting for Other Projects

To use this pattern for a new ML porting project:

### 1. Copy the Structure

```
your-project/
├── CLAUDE.md                    # Project-specific instructions
├── PORTING_STATUS.md            # Living tracking document
├── docs/
│   ├── prompts/
│   │   ├── cleanup-audit.md     # ~100% reusable
│   │   ├── research-advisor.md  # ~80% reusable
│   │   ├── verification-agent.md # Customize validation
│   │   ├── progress-tracker.md  # ~90% reusable
│   │   └── AGENT_ORCHESTRATION.md # Copy and update
│   └── audit/
│       ├── *-report-1.md
│       └── *-report-2.md
└── validation/                  # Project-specific test scripts
```

### 2. Customize Domain-Specific Parts

**research-advisor.md:**
- Update search terms for your framework/model
- Update reference implementation names
- Update technical deep-dive topics

**verification-agent.md:**
- Update test commands and paths
- Update metrics (might not be correlation for your project)
- Update target thresholds

**progress-tracker.md:**
- Update component list
- Update target metric
- Update issue categories

### 3. Create PORTING_STATUS.md

Start with this template:

```markdown
# [Project Name] Port - Status Report

## Overview
[One paragraph describing the port]

## Current Status
**Correlation/Accuracy:** [metric]
**Target:** [target]

## Issues Found and Fixed
### 1. [Issue Name] (FIXED)
**Problem:** [description]
**Solution:** [description]
**Files:** [list]

## Issues Being Investigated
[Current blockers]

## Next Steps
[Planned work]
```

---

## Principles

### 1. Separation of Concerns
Each agent has **one job**. The implementation agent writes code. The cleanup auditor finds cruft. The research advisor does research. No agent tries to do everything.

### 2. Fresh Eyes
New sessions avoid **confirmation bias**. After hours of debugging, you might tunnel-vision on a particular theory. A fresh agent starts clean and might notice something obvious.

### 3. Artifact Communication
Agents communicate through **files, not chat**. This enables async collaboration and creates an audit trail. Reports can be reviewed days later.

### 4. Quantitative Tracking
**Metrics over feelings**. "Correlation improved from 0.01 to 0.013" is progress, even if not victory. The verification agent ensures we always have numbers.

### 5. Don't Repeat Work
Each agent reads **PORTING_STATUS.md** to know what's been tried. The research advisor won't suggest approaches already ruled out. Progress tracker knows what's already fixed.

---

## Troubleshooting

### Agent produces shallow output
- Make sure you're in a **fresh session** (no prior context)
- Check that referenced files exist (PORTING_STATUS.md, etc.)
- The agent needs enough information to work with

### Reports aren't being saved
- Check file permissions in `docs/audit/`
- Verify the rotation logic completed
- Some agents may need manual save if interrupted

### Agents suggest already-tried approaches
- Update PORTING_STATUS.md with what's been tried
- Include specific details about why approaches failed
- Add a "Hypotheses Ruled Out" section

### Verification metrics don't match expectations
- Ensure Python reference is up to date
- Check that test phrase is consistent
- Verify model path is correct

---

## Auto-Trigger Pattern

During long optimization runs (e.g., autotuning), the research advisor can be auto-triggered:

```
AFTER 3 CONSECUTIVE FAILURES (or score stuck ±0.02 for 5+ iterations):
  1. Compose failure context from recent attempts
  2. Invoke /research with context as argument
  3. Wait for report at docs/audit/research-advisor-report-1.md
  4. Extract top suggestion and try ONE more iteration
  5. If still no improvement, EXIT
```

This extends sessions by at most 1 extra iteration after getting fresh research, preventing infinite loops while benefiting from "fresh eyes."

---

## Skills vs Prompts

| Method | Location | Best For |
|--------|----------|----------|
| Skills (`/command`) | `.claude/skills/` | Quick invocation, dynamic context injection, auto-trigger |
| Prompts (copy-paste) | `docs/prompts/` | Fresh-session isolation, maximum "fresh eyes" benefit |

Both produce the same reports in `docs/audit/`. Skills are preferred for routine use; prompts for when you want full session isolation.

---

*Orchestration pattern developed January 2026. Skills conversion March 2026.*
