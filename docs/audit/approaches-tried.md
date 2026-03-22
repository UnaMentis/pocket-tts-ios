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

## Approach: Per-layer + multi-step intermediate tensor dump (2026-03-21)
- **What**: Dumped all intermediate tensors (input, norm1, attn, post_attn, norm2, mlp, output) for all 6 transformer layers at step 0, plus input_linear/out_norm/latent for steps 0-2. Created matching Python and Rust dump scripts with .npy file comparison.
- **Why**: To determine whether the 0.839→1.0 correlation gap comes from (a) transformer divergence, (b) FlowNet divergence, or (c) Mimi decoder divergence.
- **Result**: **CRITICAL FINDING — Transformer and FlowNet match perfectly.**
  - All 44 per-layer tensors at step 0: cosine similarity = 1.0000, RMSE < 1e-6
  - Latent outputs at steps 0-2: max error grows from 1.9e-6 → 2.5e-6 → 5.6e-6 (normal float32 accumulation)
  - **The entire 0.839→1.0 gap is in the Mimi decoder**, not the transformer or FlowNet
  - This invalidates the research advisor's #1 hypothesis (F.scaled_dot_product_attention vs manual attention)
- **Status**: DIAGNOSTIC COMPLETE — redirects all optimization effort to Mimi decoder
- **Files**: `validation/dump_intermediates.py`, `src/models/flowlm.rs` (dump_npy helper), `validation/compare_intermediates.py`
- **Key implication**: The Mimi decoder streaming implementation (replicate padding, ConvTranspose1d overlap-add, decoder transformer KV cache, SEANet streaming) is the sole source of the remaining correlation gap.

---

## Not Yet Tried

These are potential approaches that have NOT been attempted:

- **Mimi decoder intermediate comparison**: Dump per-block outputs from Python Mimi decoder vs Rust to find which specific operation (output_proj, upsample, decoder_transformer, SEANet blocks) introduces the most divergence
- **SEANet streaming vs batch comparison**: Run Rust SEANet in batch mode (all latents at once) to isolate streaming-specific errors
- **Decoder transformer comparison**: Compare Python's non-causal attention with full KV cache vs Rust implementation
- **ConvTranspose1d overlap-add validation**: Verify the streaming ConvTranspose1d state management and overlap-add logic matches Python
- **Replicate padding deep-dive**: Test whether the first-frame replicate padding introduces systematic bias that persists through subsequent frames
