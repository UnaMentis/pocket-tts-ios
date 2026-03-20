# Research Advisor Briefing

**Date:** 2026-03-13
**Current Blocker:** Transformer divergence -- frame 0 latent correlation 0.72 with matched noise
**Research Focus:** Transformer precision, Kyutai Moshi Rust reference, Candle precision issues, methodology validation

---

## Situational Summary

The Rust/Candle port of Pocket TTS is marked "production ready" with random noise enabled, producing intelligible speech with 91% amplitude ratio and 98% RMS ratio vs Python. However, when noise tensors are matched (captured from Python and loaded in Rust), waveform correlation remains at ~0.72 for frame 0 and drops to ~0 by frame 2+. This means the **transformer produces different 1024-dim hidden states** than Python, and autoregressive compounding amplifies the error.

The Mimi decoder alone achieves ~0.74 correlation when given identical latents, confirming the transformer is the bottleneck.

### What Changed Since Last Report (2026-01-30)

1. Previous report declared "production ready" with random noise, noting correlation is meaningless with different RNGs
2. Noise-matched testing was implemented (loading Python .npy files in Rust) to isolate pure implementation differences
3. Frame 0 correlation of 0.72 confirms a real transformer-level discrepancy exists
4. The issue compounds autoregressively, destroying correlation by frame 2+

---

## Methodology Validation

### Is noise-matched correlation the right approach?

**Yes, but with caveats.** Noise-matched testing is the gold standard for isolating implementation differences. If Rust and Python receive identical noise tensors and identical model weights, any correlation gap is a pure implementation bug. The 0.72 frame-0 correlation is strong evidence of a real difference.

**However**, correlation at the waveform level conflates multiple sources of error:
- Transformer hidden state differences
- FlowNet processing differences
- Mimi decoder differences

### Better diagnostic approach: Layer-by-layer intermediate comparison

Instead of only comparing final waveforms, the most effective approach is to **dump and compare intermediate tensors at each stage**:

1. **After each transformer layer**: Compare hidden states layer-by-layer to find where divergence first appears
2. **After out_norm**: Compare the 1024-dim conditioning vector fed to FlowNet
3. **After FlowNet**: Compare the 32-dim latent
4. **After Mimi**: Compare the waveform

This narrows the search from "somewhere in the transformer" to a specific layer and operation.

### Recommended: Per-attention-head comparison

Within each transformer layer, compare:
- Q, K, V projections (before and after RoPE)
- Attention weights (softmax output)
- Attention output (after value weighting)
- MLP output

This identifies whether the issue is in attention computation, RoPE, or MLP.

---

## Key Research Findings

### From Official Kyutai Sources (Moshi Rust)

Kyutai maintains their own Rust implementation of the Moshi system at `github.com/kyutai-labs/moshi/tree/main/rust/moshi-core`. Key findings from examining their code:

**1. RoPE: They use `candle_nn::rotary_emb::rope_i()` (interleaved)**
```rust
candle_nn::rotary_emb::rope_i(&qk.to_dtype(DType::F32)?, &self.cos, &self.sin)?
    .to_dtype(qk_dtype)
```
Critical details:
- They convert to F32 before applying RoPE, then convert back to original dtype
- They use Candle's built-in `rope_i` function which expects shape `(B, H, T, D)` (post-transpose)
- Our implementation applies RoPE pre-transpose with shape `(B, T, H, D)` using a custom `apply_rotary` function

**Our custom RoPE implementation is mathematically equivalent but differs in computation order**, which can produce different floating-point results. Switching to `candle_nn::rotary_emb::rope_i()` would align with the official reference and eliminate this as a variable.

**2. Softmax: They use `candle_nn::ops::softmax_last_dim()`**
```rust
let ws = candle_nn::ops::softmax_last_dim(&pre_ws)?;
```
Our code uses `candle_nn::ops::softmax(&attn_weights, D::Minus1)`. According to the Candle source:
- `softmax()` is a generic implementation using high-level tensor ops (max, sub, exp, sum, div)
- `softmax_last_dim()` uses **backend-specific optimized kernels** with `rayon::par_chunks` on CPU

These two functions compute the same math but with different accumulation patterns and parallelism, producing slightly different floating-point results. The `softmax_last_dim` variant is both faster and matches what Kyutai uses.

**3. Attention scaling: They use `f64` for the scale factor**
```rust
let pre_ws = (pre_ws * (head_dim as f64).powf(-0.5))?;
```
Our code uses `f32`:
```rust
scale: 1.0 / (head_dim as f32).sqrt(),
// applied as:
(attn_weights * self.scale as f64)?
```
The scale value `1/sqrt(64) = 0.125` is exactly representable in both f32 and f64, so this is not a source of error. However, computing `(head_dim as f64).powf(-0.5)` vs `1.0 / (head_dim as f32).sqrt()` could differ for non-power-of-2 head dims.

**4. KV cache: They use scatter-based pre-allocated cache**
Kyutai's `ScatteredKvCache` pre-allocates a fixed-size tensor and uses `scatter_set` to insert new K/V values. Our code uses `Tensor::cat` to concatenate, which creates new tensors each step. Functionally equivalent but different memory patterns.

**5. Python uses `F.scaled_dot_product_attention`**
The Python Pocket TTS code uses PyTorch's fused SDPA kernel, which has its own numerical characteristics:
- On CPU with float32, it typically uses the "math" backend
- The "math" backend computes: `attn = softmax(Q @ K^T / sqrt(d) + mask) @ V`
- This may differ from our manual implementation in operation ordering

### From Candle/Framework Research

**Known precision issue (GitHub #3032)**:
- MSE between Candle and PyTorch matmul: ~4.1e-10 to ~8.2e-11
- MAE: ~0.000006 to ~0.000015 per operation
- Maintainer confirms: "errors accrue layer by layer for each token"
- Root cause: floating-point arithmetic is not associative; different operation orders produce different results

**Output divergence issue (#2031)**:
- Candle vs PyTorch diverge on Mistral-7B token generation even with greedy decoding
- Maintainer (Laurent Mazare): "we accumulate with f32 in the softmax whereas pytorch may well do something slightly different"
- Different CUDA algorithm implementations and accumulation strategies
- Considered expected behavior -- exact token matching is unrealistic

**Key insight**: For autoregressive models, even tiny per-step errors compound. A 1e-5 error in attention weights, passed through 6 layers, then fed back as input for the next step, can diverge significantly after 10-20 steps. This is consistent with the observed 0.72 frame-0 correlation dropping to ~0 by frame 2+.

### From Pocket TTS Community

- **wasm-pocket-tts** by LaurentMazare (Candle maintainer): Uses XN framework, not Candle. Repo not found (may be private or renamed).
- **pocket-tts-candle** by babybirdprd: Our code derives from this. No published correlation metrics.
- **PocketTTS.cpp** by VolgaGerm: C++ runtime using ONNX. ONNX export from Python ensures exact numerical match by definition.
- No public reports of anyone achieving >0.9 correlation between a Candle port and PyTorch for an autoregressive transformer of this size.

---

## Suggested Approaches

### High Confidence

**1. Switch to `candle_nn::ops::softmax_last_dim()` for attention**
- **Why**: Matches Kyutai's Moshi Rust implementation. Uses optimized backend-specific kernels. The generic `softmax()` uses different operation ordering that may accumulate differently.
- **How**: Replace `candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)` with `candle_nn::ops::softmax_last_dim(&attn_weights)` in `src/modules/attention.rs` (both `MultiHeadAttention` and `FusedMultiHeadAttention`).
- **Expected impact**: Small per-operation improvement, but compounds over 6 layers x N steps.

**2. Use Candle's built-in `rope_i()` instead of custom RoPE**
- **Why**: Kyutai's Moshi Rust uses `candle_nn::rotary_emb::rope_i()`. It processes data in `(B, H, T, D)` layout using a tight loop over pairs, while our custom implementation uses reshape/narrow/cat operations that produce different float accumulation.
- **How**: After transposing Q, K to `(B, H, T, D)`, call `candle_nn::rotary_emb::rope_i()` with precomputed cos/sin tensors. Ensure cos/sin have shape `(1, 1, T, D/2)` matching what `rope_i` expects.
- **Expected impact**: Could be significant. RoPE errors feed into every subsequent attention computation.

**3. Layer-by-layer intermediate tensor comparison**
- **Why**: The current approach compares only final waveforms. We need to find exactly WHERE divergence starts.
- **How**:
  1. Add a Python script that runs the model and dumps tensors after every transformer layer, after RoPE, after softmax, etc. (extend `validation/dump_intermediates.py`)
  2. Load these tensors in Rust and compare at each stage
  3. Find the first operation where correlation drops below 0.99
- **Expected impact**: Directly identifies the root cause operation.

**4. Compute RoPE frequencies on-the-fly like Python (not precomputed cache)**
- **Why**: Python computes `freqs = exp(ds * (-log(max_period) * 2 / D))` and `cos(freqs * ts)` at each call. Our Rust code precomputes frequencies via `positions.matmul(&inv_freq)`. The matmul-based computation has different float accumulation order than element-wise multiplication.
- **How**: Change `RotaryEmbedding` to compute cos/sin on-the-fly using element-wise operations matching Python exactly: `cos_val = cos(inv_freq[i] * position)` for each element.
- **Expected impact**: Eliminates RoPE as a precision variable.

### Worth Trying

**5. Remove all debug logging from attention.rs and flowlm.rs**
- **Why**: The extensive debug logging with `AtomicUsize` counters and conditional tensor reads adds overhead and could affect tensor evaluation ordering in Candle's lazy evaluation model. Debug code should not be in production paths.
- **How**: Remove or gate behind a compile-time feature flag.

**6. Match Python's KV cache pattern (pre-allocated scatter)**
- **Why**: `Tensor::cat` creates new tensors each step, potentially with different memory alignment. Scatter-based cache modifies in place.
- **How**: Pre-allocate KV cache to max_seq_len and scatter-write new values.
- **Expected impact**: Likely minimal for correctness, but better for performance.

**7. Convert Q/K to F32 before RoPE (like Moshi)**
- **Why**: Kyutai's code explicitly converts to F32 before RoPE application: `qk.to_dtype(DType::F32)`. If the model weights are already F32, this is a no-op, but it ensures consistency.
- **How**: Add `.to_dtype(DType::F32)?` before RoPE, `.to_dtype(original_dtype)?` after.

### Speculative

**8. Use ONNX export path for bit-exact matching**
- **Why**: ONNX runtimes can reproduce PyTorch results exactly. Export the Python model to ONNX, run in onnxruntime-rs, compare outputs to isolate whether the issue is in our transformer logic or in Candle's numerics.
- **How**: Use `torch.onnx.export()` on the Python model, load with `ort` crate in Rust.

**9. Double-precision (f64) validation pass**
- **Why**: Running both Python (with `.double()`) and Rust (with `DType::F64`) in f64 would show if the divergence is a precision issue or a logic bug. If f64 gives correlation ~1.0, the issue is purely float32 accumulation. If f64 still diverges, there is a logic bug.
- **How**: Temporarily cast all model weights and computations to f64 in both Python and Rust.

---

## Already Tried (Don't Repeat)

From PORTING_STATUS.md, these have been fixed and verified:
1. RoPE interleaved vs split halves - FIXED
2. RoPE applied before vs after transpose - FIXED
3. LayerNorm vs RMSNorm for out_norm - FIXED
4. FlowNet sinusoidal order, activation, AdaLN chunk order, SiLU placement - ALL FIXED
5. LSD time progression (two time values) - FIXED
6. SEANet activation (ELU not GELU) - FIXED
7. Voice conditioning concatenation order (voice first, then text) - FIXED
8. Two-phase forward pass (voice phase, text phase, generation) - FIXED
9. FinalLayer missing norm_final - FIXED
10. SEANet output tanh removal - FIXED
11. FlowNet TimeEmbedding RMSNorm (proper variance normalization) - FIXED
12. FlowNet time embedding addition (only to conditioning) - FIXED
13. Latent denormalization (only before Mimi, not in loop) - FIXED
14. Weight loading - VERIFIED CORRECT

---

## Useful Links & References

- Kyutai Moshi Rust source: https://github.com/kyutai-labs/moshi/tree/main/rust/moshi-core/src
  - `transformer.rs` - Reference transformer with `rope_i()` and `softmax_last_dim()`
  - `kv_cache.rs` - ScatteredKvCache implementation
  - `nn.rs` - Quantization wrappers and `matmul_dtype()` helper
- Candle precision issue #3032: https://github.com/huggingface/candle/issues/3032
- Candle divergence issue #2031: https://github.com/huggingface/candle/issues/2031
- Candle `softmax` vs `softmax_last_dim`: https://github.com/huggingface/candle/blob/main/candle-nn/src/ops.rs
- Candle `rope` vs `rope_i`: https://github.com/huggingface/candle/blob/main/candle-nn/src/rotary_emb.rs
- Kyutai Pocket TTS official: https://github.com/kyutai-labs/pocket-tts

---

## Priority Action Plan

| Priority | Action | Expected Impact | Effort |
|----------|--------|-----------------|--------|
| 1 | Layer-by-layer intermediate comparison | Identifies root cause | Medium |
| 2 | Switch to `softmax_last_dim()` | Matches Moshi reference | Low |
| 3 | Switch to `rope_i()` from candle_nn | Matches Moshi reference | Medium |
| 4 | Compute RoPE frequencies on-the-fly | Eliminates freq precision gap | Low |
| 5 | f64 validation pass | Logic bug vs precision triage | Medium |
| 6 | Remove debug logging | Clean code, no eval interference | Low |
