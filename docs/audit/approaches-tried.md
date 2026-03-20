# Approaches Tried — Correlation Improvement Log

Structured log of optimization approaches attempted, with results.
The research agent MUST read this before suggesting new approaches.

Last updated: 2026-03-19

---

## Approach: Noise tensor off-by-one fix (2026-03-18)
- **What**: Changed `noise_tensors[step]` → `noise_tensors[step + 1]` in flowlm.rs
- **Why**: Python's `generate_audio()` makes a text-prompting FlowNet call before autoregressive generation. This call generates noise captured as `noise_step_000.npy`, but **discards the FlowNet output**. Autoregressive step 0 (BOS) uses `noise_step_001.npy`. Rust was using `noise_step_000` for step 0 — the wrong tensor.
- **Result**: **MASSIVE IMPROVEMENT**. End-to-end correlation: ~0 → **0.839**. 10/45 frames > 0.9.
- **Status**: APPLIED (both sync and streaming paths in flowlm.rs)
- **Files**: `src/models/flowlm.rs` (lines ~584, ~721)

## Approach: Switch to `softmax_last_dim()` in attention (2026-03-17)
- **What**: Changed `candle_nn::ops::softmax(&attn_weights, D::Minus1)` → `candle_nn::ops::softmax_last_dim(&attn_weights)` in both MultiHeadAttention and FusedMultiHeadAttention
- **Why**: Kyutai's Moshi Rust implementation uses `softmax_last_dim`. Research advisor recommended it as potentially more numerically stable for the last dimension.
- **Result**: **ZERO measurable impact**. Bit-identical output when tested with same noise.
- **Status**: APPLIED (no harm, matches Moshi reference)
- **Files**: `src/modules/attention.rs`

## Approach: Switch to `rope_i()` from candle_nn (2026-03-17)
- **What**: Replaced custom `apply_rotary()` with `candle_nn::rotary_emb::rope_i()`. Required transposing `(B,T,H,D)` → `(B,H,T,D)` before call and back after.
- **Why**: Research advisor recommended using Candle's built-in interleaved RoPE. Kyutai's Moshi Rust uses candle_nn RoPE.
- **Result**: **ZERO measurable impact**. Bit-identical output when tested with same noise.
- **Status**: APPLIED (no harm, matches Moshi reference)
- **Files**: `src/modules/rotary.rs`

## Approach: Layer-by-layer transformer comparison (2026-03-17)
- **What**: Built diagnostic scripts (`validation/capture_step0_all_layers.py`, `capture_flownet_seeded.py`) to compare per-layer transformer outputs and FlowNet intermediates between Python and Rust.
- **Why**: To locate where transformer divergence originates.
- **Result**: Discovered that transformer layer outputs match to ~1e-6 when comparing Rust vs a **non-seeded** Python run, but diverge significantly vs the **seeded reference** run. This led to discovering the noise off-by-one bug. Also confirmed `input_proj` matches perfectly but `cond_embed` diverges (because it depends on transformer hidden state which depends on noise alignment).
- **Status**: DIAGNOSTIC COMPLETE — led to the noise offset discovery
- **Files**: `validation/capture_step0_all_layers.py`, `validation/capture_flownet_seeded.py`, `validation/capture_flownet_step0.py`

---

## Not Yet Tried

These are potential approaches that have NOT been attempted:

- **f64 validation pass**: Run critical operations (matmul, softmax, RMSNorm) in f64 to measure f32 accumulation error
- **Per-layer intermediate tensor dump**: Compare Rust vs Python hidden states at each transformer layer for step 0, using correctly-aligned noise
- **Weight loading audit**: Verify all weights loaded correctly (transposes, reshapes, dtype)
- **KV cache drift analysis**: Check if KV cache introduces cumulative precision drift over 125+ voice positions
- **Text embedding comparison**: Verify `conditioner(tokens)` in Python matches `text_embedding.forward(token_ids)` in Rust
- **Matmul accumulation order**: Test if Candle and PyTorch produce different results for the same matmul due to summation order
