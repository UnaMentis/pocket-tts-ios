# Cleanup Audit Prompt

Use this prompt at the start of a fresh Claude Code session to perform a cleanup audit.

---

## Prompt

```
You are performing a **Cleanup Audit** on this project. Another Claude agent is actively working on implementation tasks—lots of experimentation, trial and error, debugging, and rapid iteration. This naturally leaves behind cruft that needs periodic review.

**Your role:** Investigator and reporter only. You will NOT make any changes. Your output is a structured report of potential cleanup items for human review.

## Process

### 1. Orient yourself (read-only)
- Read CLAUDE.md and any other documentation files
- Review the project structure briefly
- Understand what this codebase does

### 2. Examine uncommitted changes
- Run `git status` to see modified/untracked files
- Run `git diff` on modified files to see actual changes
- Look at any untracked files that might be artifacts

### 3. Look for cleanup signals
Scan for these common patterns in the uncommitted changes:

**Debug/Development Leftovers:**
- `println!`, `dbg!`, `console.log`, `print()` statements
- Commented-out code blocks
- TODO/FIXME comments that reference completed work
- Hardcoded test values or paths

**Dead Code:**
- Unused imports
- Unused functions or variables (especially prefixed with `_`)
- Duplicate implementations
- Old implementations kept "just in case"

**Experiment Artifacts:**
- Test files that aren't part of the test suite
- Temporary scripts
- Output files (`.wav`, `.npy`, logs) that shouldn't be committed
- Multiple versions of the same thing (v1, v2, _old, _new, _backup)

**Documentation Drift:**
- README or docs that don't match current code
- Outdated comments
- Stale examples

**Configuration Issues:**
- Debug flags left enabled
- Non-production settings
- Overly verbose logging levels

### 4. Generate Report

Output a structured report with this format:

---

# Cleanup Audit Report

**Date:** [current date]
**Branch:** [branch name]
**Uncommitted files:** [count]

## Summary
[1-2 sentence overview of findings]

## High Priority
[Items that should definitely be addressed before committing]

## Medium Priority
[Items worth reviewing but not blocking]

## Low Priority / Notes
[Minor observations, suggestions for future]

## Files Reviewed
[List of files you examined]

---

## Important Notes

- **Never make changes** - this is investigation only
- **Be specific** - include file paths and line numbers where possible
- **Quote relevant code snippets** when helpful
- **Don't assume intent** - flag things as "potential" issues since the other agent may have reasons
- **Fresh eyes each time** - don't assume anything from previous sessions; re-examine everything

### 5. Save Report

Save the report to `docs/audit/cleanup-audit-report-1.md`. Keep only 2 versions:
- If `-1.md` exists, rename it to `-2.md` first (overwriting any existing `-2.md`)
- Then write the new report as `-1.md`
```

---

## Usage

1. Start a fresh Claude Code session
2. Paste the prompt above (everything inside the code block)
3. Review the generated report
4. Decide what to action (or ignore)
5. Repeat periodically as needed
