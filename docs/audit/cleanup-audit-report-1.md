# Cleanup Audit Report

**Date:** 2026-01-24
**Branch:** main
**Uncommitted files:** 0 (working tree clean)

## Summary

The working tree is now clean - all previously flagged changes have been committed. However, the committed codebase still contains significant debug instrumentation (131+ `eprintln!` statements, 4 static atomic counters) and 89 tracked binary files (`.wav`, `.npy`, `.f32`) that should be in `.gitignore`. This is acceptable for active porting work but will need cleanup before release.

---

## High Priority

### 1. Debug Statements in Production Code

Extensive `eprintln!` debug statements (131 total) that should be removed or feature-gated before release:

| File | Count | Notes |
|------|-------|-------|
| `src/models/flowlm.rs` | 38 | Uses static `DEBUG_COUNTER` |
| `src/modules/attention.rs` | 25 | Uses static `DEBUG_ATTN` |
| `src/modules/flownet.rs` | 19 | Uses static `DIAG_COUNT` and `FLOWNET_STEP` |
| `src/models/pocket_tts.rs` | 15 | Synthesis pipeline logging |
| `src/models/mimi.rs` | 12 | Decoder diagnostics |
| `src/modules/embeddings.rs` | 2 | Text embedding stats |
| `src/bin/test_tts.rs` | 20 | (Acceptable - test/CLI tool) |

Additionally, 241 `println!` statements exist (130 in `test_tts.rs` which is acceptable).

### 2. Static Atomic Counters (Global State Leakage)

Four static counters will cause issues in a production library - they never reset between calls:

```rust
// src/modules/attention.rs:230
static DEBUG_ATTN: AtomicUsize

// src/modules/flownet.rs:232
static DIAG_COUNT: AtomicUsize

// src/modules/flownet.rs:386
static FLOWNET_STEP: AtomicUsize

// src/models/flowlm.rs:94
static DEBUG_COUNTER: AtomicUsize
```

### 3. Tracked Binary Files (89 files)

Binary debug/test outputs are committed to git that should be in `.gitignore`:

- `test_output.wav` (root)
- `validation/debug_outputs/*.npy` (9 files)
- `validation/denormalized_latents.npy` and `.f32`
- `validation/mimi_debug/*.npy` (20+ files)
- `validation/rust_outputs/*.wav` and `*.npy`
- `validation/reference_outputs/*.wav` and `*.npy`

---

## Medium Priority

### 1. Incomplete .gitignore

Current `.gitignore` is missing patterns for validation outputs:

```gitignore
# Should be added:
*.wav
*.npy
*.f32
validation/debug_outputs/
validation/mimi_debug/
validation/rust_outputs/
validation/reference_outputs/
```

### 2. Hardcoded Paths in Python Scripts

Some validation scripts may contain hardcoded absolute paths (e.g., `/Users/ramerman/...`). These should be made relative or configurable.

---

## Low Priority / Notes

### 1. No Compiler Warnings

`cargo check` completes cleanly with no warnings - good hygiene.

### 2. No TODO/FIXME Comments

No stale task markers found in the codebase.

### 3. Clean Git State

All changes properly committed. The previous audit (which noted 7 uncommitted files) has been addressed.

---

## Files Reviewed

**Rust Source (for debug patterns):**
- `src/models/flowlm.rs`
- `src/models/mimi.rs`
- `src/models/pocket_tts.rs`
- `src/modules/attention.rs`
- `src/modules/flownet.rs`
- `src/modules/embeddings.rs`
- `src/bin/test_tts.rs`

**Configuration:**
- `.gitignore`

**Previous Audit:**
- `docs/audit/cleanup-audit-report-1.md` (now renamed to `-2.md`)

---

## Recommendations

1. **Keep debug code for now** - Active porting work is ongoing (latents match but waveform correlation is ~0.013)

2. **Add debug feature flag** - When ready, wrap diagnostics in `#[cfg(feature = "debug")]` or use `tracing` crate

3. **Update .gitignore before next commit** - Add patterns for:
   - `*.wav`
   - `*.npy`
   - `*.f32`
   - `validation/*_outputs/`
   - `validation/mimi_debug/`

4. **Remove tracked binaries** - Run:
   ```bash
   git rm --cached '*.wav' '*.npy' '*.f32'
   ```
   Then update .gitignore and commit

---

## Cleanup Activity Log

### 2026-01-24
- Fresh audit performed with clean working tree
- Previous audit archived as `-2.md`
- Identified 89 tracked binary files that should be in .gitignore
- Verified no compiler warnings
- Confirmed debug instrumentation is expected during active porting
