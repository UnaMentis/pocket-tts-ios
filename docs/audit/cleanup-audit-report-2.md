# Cleanup Audit Report

**Date:** 2026-01-24
**Branch:** main
**Uncommitted files:** 7 (5 modified, 2 untracked)

## Summary

The codebase contains extensive debug instrumentation from active porting/debugging work. There are 70+ `eprintln!` debug statements throughout production code, some using static atomic counters that leak global state. This is expected for active development but should be cleaned up before any release.

---

## High Priority

### 1. Debug Statements in Production Code

Extensive `eprintln!` debug statements that should be removed or converted to proper logging before release:

| File | Count | Notes |
|------|-------|-------|
| [src/modules/attention.rs](../../src/modules/attention.rs) | 15 | Lines 240-344. Uses static `DEBUG_ATTN` counter |
| [src/models/flowlm.rs](../../src/models/flowlm.rs) | 22 | Lines 116-477. Uses static `DEBUG_COUNTER` |
| [src/modules/flownet.rs](../../src/modules/flownet.rs) | 11 | Lines 52-445. Time embedding and FlowNet diagnostics |
| [src/models/pocket_tts.rs](../../src/models/pocket_tts.rs) | 15 | Lines 98-296. Synthesis pipeline logging |
| [src/models/mimi.rs](../../src/models/mimi.rs) | 4 | Lines 416-454. Decoder diagnostics |
| [src/modules/embeddings.rs](../../src/modules/embeddings.rs) | 2 | Lines 32-33. Text embedding stats |

### 2. Static Atomic Counters (Global State Leakage)

These will cause issues in a production library - counters never reset between calls:

```rust
// src/modules/attention.rs:236
static DEBUG_ATTN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// src/models/flowlm.rs:109
static DEBUG_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
```

### 3. Config Change Needs Verification

[src/models/flowlm.rs:47](../../src/models/flowlm.rs#L47) - Changed `rms_norm_eps` from `1e-6` to `1e-5`:
```rust
rms_norm_eps: 1e-5,  // Match Python nn.LayerNorm default
```
This is documented in PORTING_STATUS.md but should be verified as intentional before committing.

---

## Medium Priority

### 1. Untracked Debug Script

[validation/dump_attention.py](../../validation/dump_attention.py) (93 lines) - Python script for comparing attention values with Rust. Contains hardcoded path:
```python
sys.path.insert(0, "/Users/ramerman/dev/unamentis/models/kyutai-pocket-ios")
model_dir = "/Users/ramerman/dev/unamentis/models/kyutai-pocket-ios"
```
**Decision needed:** Add to .gitignore, delete, or commit with relative paths?

### 2. Session Notes in PORTING_STATUS.md

70 lines of session-specific debugging notes added. While useful for development, consider whether these belong in version control or should be moved to a separate doc.

### 3. Compiler Warnings

```
warning: unused import: `RMSNorm`
warning: field `kernel_size` is never read
warning: field `dim` is never read
```

### 4. Modified Binary Files

- `validation/rust_outputs/phrase_00_rust.wav` (changed 169KB -> 173KB)
- `validation/rust_outputs/phrase_00_rust_latents.npy` (changed 5.6KB -> 5.8KB)

These are generated test outputs. **Verify:** Should these be in .gitignore instead of committed?

---

## Low Priority / Notes

### 1. New docs/ Directory

`docs/prompts/cleanup-audit.md` - The cleanup audit prompt template. Appears intentional.

### 2. println! Statements in Test Binary

`src/bin/test_tts.rs` has many `println!` statements, but these are appropriate for a test/CLI tool.

---

## Files Reviewed

**Modified:**
- [PORTING_STATUS.md](../../PORTING_STATUS.md) - +70 lines of session notes
- [src/models/flowlm.rs](../../src/models/flowlm.rs) - Debug code + config change
- [src/modules/attention.rs](../../src/modules/attention.rs) - Debug code with static counter
- `validation/rust_outputs/phrase_00_rust.wav` - Binary output
- `validation/rust_outputs/phrase_00_rust_latents.npy` - Binary output

**Untracked:**
- `docs/` - New documentation directory
- `validation/dump_attention.py` - Debug script

**Also Scanned (not modified, but contain debug code from previous commits):**
- `src/modules/flownet.rs`
- `src/models/mimi.rs`
- `src/models/pocket_tts.rs`
- `src/modules/embeddings.rs`

---

## Recommendations

1. **Keep debug code for now** - Active porting work is ongoing. Create a cleanup task for when porting is complete.

2. **Consider debug feature flag** - Wrap diagnostics in `#[cfg(feature = "debug")]` or use `tracing` crate for proper logging levels.

3. **Add to .gitignore** (if not already):
   - `validation/rust_outputs/*.wav`
   - `validation/rust_outputs/*.npy`
   - `validation/dump_attention.py` (or fix paths if keeping)

4. **Run `cargo fix`** - Fixes the unused import automatically.

---

## Cleanup Activity Log

### 2026-01-24
- **Ran `cargo fix`** - Removed unused `RMSNorm` import from `src/models/flowlm.rs`
- **Skipped unused fields** - `kernel_size` (mimi.rs) and `dim` (rotary.rs) left as-is; likely needed for future use
- **Skipped debug statements** - Porting still active; keeping diagnostics per report recommendation
