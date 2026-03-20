# Research Advisor Briefing

**Date:** 2026-03-19
**Current Blocker:** Transformer hidden state divergence (~16% correlation gap)
**Research Focus:** Python source code audit, F.scaled_dot_product_attention semantics, RoPE frequency computation, LayerNorm variance, FlowNet RMSNorm, GELU precision, causal mask format, KV cache accumulation
**Triggered By:** Manual invocation after establishing noise-matched baseline at 0.839 correlation

---

## Situational Summary

The Rust/Candle port achieves 0.839 end-to-end waveform correlation with Python (noise-matched, phrase_00), up from ~0 after fixing the noise off-by-one bug on 2026-03-18. The remaining ~16% gap is real transformer hidden state divergence. Frames 0-1 are weakest (~0.18, -0.09) while frames 3+ recover well (0.88-0.98), and 10/45 frames exceed 0.9 correlation. The pattern suggests the divergence is worst during early conditioning (where the transformer processes BOS + FlowNet noise for the first time) and diminishes as the autoregressive loop self-corrects.

This report is based on a thorough reading of the **actual Python source code** installed at `.venv/lib/python3.14/site-packages/pocket_tts/`, the Kyutai Moshi Rust reference at `github.com/kyutai-labs/moshi/tree/main/rust/moshi-core/src/transformer.rs`, Candle precision issue reports, and the current Rust implementation. Previous reports (2026-03-13) recommended switching to `softmax_last_dim` and `rope_i`, both of which were applied with zero measurable impact.

The primary new finding in this report is a **critical difference in how Python and Rust compute attention**: Python uses `F.scaled_dot_product_attention()` which is a fused kernel with specific numerical behavior, while Rust uses manual matmul + softmax + matmul. Additionally, there are several smaller but compounding differences in RoPE frequency computation, attention mask format, and the autoregressive state management.

---

## Methodology Validation

### Is noise-matched waveform correlation the right primary metric?

**Yes, emphatically.** The jump from ~0 to 0.839 after fixing the noise off-by-one bug validated this approach. With identical noise, correlation directly measures implementation fidelity. The remaining 0.161 gap is real and attributable to transformer hidden state divergence.

### Should we target 0.95 or settle for 0.839?

The per-frame analysis is informative: frames 3+ achieve 0.88-0.98, suggesting the implementations converge once the autoregressive loop is established. The early-frame weakness (0.18, -0.09) suggests a conditioning-phase divergence that compounds in the first 2-3 steps then attenuates. Achieving >0.95 overall requires fixing the early-frame divergence specifically.

### Recommended diagnostic approach

The most effective next step is a **step-0 intermediate tensor dump** comparing Python vs Rust at every stage:
1. After `input_linear(bos)` - should match exactly (same weights, same input)
2. After each transformer layer's norm1, attention, norm2, MLP
3. After `out_norm`
4. After FlowNet `cond_embed`
5. After FlowNet velocity prediction

This narrows "transformer divergence" to a specific operation within a specific layer.

---

## Key Research Findings

### 1. F.scaled_dot_product_attention vs Manual Attention (HIGH IMPACT)

**Finding:** Python uses `F.scaled_dot_product_attention(q, k, v, attn_mask)` (line 117 of `pocket_tts/modules/transformer.py`). Rust uses manual `q.matmul(&k.t()) * scale + mask -> softmax -> matmul(&v)`.

**Why this matters:** PyTorch's SDPA on CPU with float32 uses the "math" backend, which is a fused C++ implementation. Key differences:

1. **Scaling is fused**: SDPA computes `softmax(Q @ K^T / sqrt(d) + mask) @ V` in a single kernel. The Rust code first computes `Q @ K^T`, then multiplies by scale, then adds mask, then softmax, then matmul V. Each step materializes an intermediate tensor, accumulating floating-point rounding at each step.

2. **Mask format differs**: Python's `_materialize_causal_mask` produces `log(tril(ones))` which gives `0.0` for allowed positions and `-inf` for masked positions. The Rust `create_causal_mask` produces the same values. However, SDPA may handle the mask addition differently internally (it can accept both additive masks and boolean masks with different codepaths).

3. **Softmax internal precision**: SDPA's math backend keeps all intermediates in the input dtype. When input is float32, this is the same as the Rust code. But the accumulation order within the fused kernel differs from Candle's `softmax_last_dim`.

**Estimated impact:** Small per-operation (~1e-5 MAE per matmul per Candle issue #3032), but compounds across 6 layers, 16 heads, and hundreds of autoregressive steps.

**How to test:** Replace the manual attention in Rust with a single fused operation. Unfortunately Candle does not have a direct SDPA equivalent. However, you can measure the divergence by dumping Q, K, V from Python (just before SDPA) and computing attention manually in Python to compare against the fused result.

### 2. RoPE Frequency Computation Order (MEDIUM IMPACT)

**Finding:** Python computes frequencies **on-the-fly** each forward call:
```python
ds = torch.arange(D // 2, dtype=torch.float32)
freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
ts = torch.arange(T, dtype=torch.float32) + offset
rotr = torch.cos(freqs * ts)
```

Rust **precomputes** a cache:
```rust
let inv_freq: Vec<f32> = (0..half_dim)
    .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
    .collect();
let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
let cos_cache = freqs.cos()?;
let sin_cache = freqs.sin()?;
```

While mathematically equivalent, there are subtle precision differences:
- `exp(d * C)` vs `1.0 / base.powf(2*d/D)` - different function call paths for computing the same frequency
- `freqs * ts` (element-wise) vs `positions.matmul(&inv_freq)` (outer product) - for this specific case (outer product of 1D vectors), the results should be identical since there's no summation

But Candle's `rope_i()` function operates on the precomputed cos/sin cache, while Python computes cos/sin fresh each call from element-wise operations. The cos/sin values themselves should be identical (both are applied to `position * frequency` pairs), but any difference in the frequency values propagates to every subsequent attention computation.

**How to test:** Dump the cos/sin values from Python for positions 0-200 and compare against Rust's precomputed cache. Any difference > 1e-7 indicates a frequency computation discrepancy.

### 3. Python KV Cache Uses Pre-allocated Scatter, Rust Uses Tensor::cat (LOW-MEDIUM IMPACT)

**Finding:** Python pre-allocates a NaN-filled cache tensor and uses array slicing to insert new K/V values:
```python
cache[0, :, current_end:current_end + k.shape[1]] = k
valid = cache[:, :, :current_end + k.shape[1]]
```

Rust uses `Tensor::cat` to concatenate:
```rust
let k_new = Tensor::cat(&[k_cache, &k], 2)?;
```

**Why this matters:** `Tensor::cat` creates a new tensor each step, copying all previous cache entries. This doesn't change values, but the memory layout and alignment may differ from Python's in-place update. More importantly, if there are any NaN propagation issues or if the cache retrieval order affects attention computation, this could introduce differences.

**Estimated impact:** Likely zero for correctness, but worth validating that the cached K/V values are bit-identical to what Python produces at each step.

### 4. Python's QKV Split Uses `view + unbind`, Not `narrow` (ZERO IMPACT - VERIFIED)

The Python code uses `projected.view(b, t, 3, h, d)` then `torch.unbind(packed, dim=2)`. Rust uses `narrow` on the last dimension. I verified these produce identical results since the reshape interprets the flat 3072-dim output as `[3][16][64]` in C-contiguous order, making Q = first 1024, K = next 1024, V = last 1024 -- same as Rust's narrow.

### 5. Python Uses `nn.LayerNorm` (Built-in), Rust Uses Custom Implementation (LOW IMPACT)

**Finding:** The FlowLM transformer layers use `nn.LayerNorm(d_model, eps=1e-5)` which is PyTorch's highly optimized C++ implementation. The Rust code uses a custom implementation:
```rust
let mean = x.mean_keepdim(D::Minus1)?;
let x_centered = x.broadcast_sub(&mean)?;
let variance = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
let x_normed = x_centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
```

Both use biased variance (dividing by N). However, PyTorch's built-in LayerNorm is a single fused C++ kernel that computes mean, variance, normalization, and affine transform in one pass. The Rust code performs 5 separate tensor operations, each materializing an intermediate. The accumulation order for `mean_keepdim` and the variance computation may differ.

**How to test:** Dump the output of `norm1` from Python layer 0 at step 0 and compare to Rust.

### 6. Python Transformer Uses `F.gelu()` (Default = Exact erf), Rust Uses `gelu_erf()` (ZERO IMPACT - MATCHES)

**Finding:** The Python `StreamingTransformerLayer._ff_block` uses:
```python
update = self.linear2(F.gelu(self.linear1(x)))
```

`F.gelu()` without `approximate` parameter defaults to `approximate='none'`, which uses the exact erf-based GELU: `x * 0.5 * (1.0 + erf(x / sqrt(2.0)))`.

The Rust code uses `x.gelu_erf()` which also computes the exact erf-based GELU.

**Impact:** These should produce identical results. No change needed.

### 7. Python FlowNet MLP Uses SiLU, Rust Matches (ZERO IMPACT)

**Finding:** The FlowNet's ResBlock MLP and AdaLN modulation both use `nn.SiLU()` in Python. The Rust code uses `candle_nn::ops::silu()`. These should match.

### 8. Layer Scale Not Used in FlowLM Transformer (VERIFIED)

**Finding:** The `StreamingTransformer.from_pydantic_config()` does NOT pass `layer_scale`:
```python
return cls(
    d_model=config.d_model, num_heads=config.num_heads,
    num_layers=config.num_layers, dim_feedforward=dim_feedforward,
    max_period=float(config.max_period), kind="flow_lm",
)
```

This means `layer_scale=None`, so `self.layer_scale_1 = nn.Identity()`. The Rust code also has no layer_scale. Matches.

### 9. Kyutai Moshi Rust Reference Findings (INFORMATIONAL)

From examining `github.com/kyutai-labs/moshi/blob/main/rust/moshi-core/src/transformer.rs`:

- **Attention**: Uses manual matmul + softmax (NOT fused SDPA), with `softmax_last_dim()` and scaling via `(head_dim as f64).powf(-0.5)`. For BF16 + CUDA, it uses flash_attn.
- **RoPE**: Uses `candle_nn::rotary_emb::rope_i()` with explicit F32 cast before applying, then cast back.
- **RMSNorm**: Uses `candle_nn::ops::rms_norm()` with `alpha` parameter.
- **LayerScale**: Has a `LayerScale` struct with learnable per-channel scaling.
- **KV Cache**: Uses pre-allocated `ScatteredKvCache` with scatter-write, unlike our `Tensor::cat` approach.

The Moshi reference uses manual attention like our code, not SDPA. This validates our approach but doesn't help close the gap to Python's SDPA.

### 10. Candle Precision Issues (FROM GitHub)

From Candle issue #3032:
- MSE between Candle and PyTorch matmul: ~4e-10 (MAE ~1.5e-5)
- Maintainer Robert Knight: "floating point addition and multiplication are not associative. There will be very small differences depending on the order in which calculations are done."
- Issue closed as "expected behavior"

From Candle issue #2031:
- Candle vs PyTorch diverge on token generation even with greedy decoding
- Maintainer Laurent Mazare: "we accumulate with f32 in the softmax whereas pytorch may well do something slightly different"
- Divergence is expected and considered inherent to framework differences

**Key insight:** No public report exists of anyone achieving >0.9 correlation between a Candle port and PyTorch for an autoregressive transformer. Our 0.839 may already be near the practical ceiling for this framework combination.

---

## Suggested Approaches

### High Confidence

**1. Step-0 Per-Layer Intermediate Tensor Dump (DIAGNOSTIC)**
- **Why**: This is the single most impactful thing to do next. Instead of guessing, measure exactly where divergence starts.
- **How**:
  1. In Python, add hooks to capture tensors after norm1, after attention, after norm2, after MLP, after final_norm for step 0 of autoregressive generation. Save as .npy files.
  2. In Rust, dump the same tensors at the same points for step 0.
  3. Compare per-element: compute correlation and max absolute difference at each stage.
  4. Find the FIRST operation where correlation drops below 0.999.
- **Expected outcome**: Identifies whether divergence starts in attention (SDPA vs manual) or elsewhere.
- **Effort**: Medium (a few hours to add capture hooks in both Python and Rust)

**2. Match Python's On-the-fly RoPE Frequency Computation**
- **Why**: Eliminate RoPE frequencies as a precision variable. Even tiny frequency differences (1e-7) get multiplied by large position offsets (125+) and compound through cos/sin.
- **How**: Replace `RotaryEmbedding::compute_inv_freq` and `compute_cache` to exactly mirror Python's computation:
  ```rust
  // Instead of precomputing via matmul, compute element-wise per position:
  let ds: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
  let freqs: Vec<f32> = ds.iter()
      .map(|d| f32::exp(d * (-f32::ln(base) * 2.0 / dim as f32)))
      .collect();
  // Then for each position t:
  // cos_val[t][d] = cos(freqs[d] * (t + offset))
  // sin_val[t][d] = sin(freqs[d] * (t + offset))
  ```
- **Key difference**: Python uses `exp(d * C)` while Rust uses `1.0 / base.powf(2*d/D)`. These are algebraically identical but `exp` and `powf` use different numerical codepaths. Also, Python computes `freqs * ts` element-wise while Rust uses a matmul for the outer product.
- **Expected impact**: Small per-element (~1e-7) but matters at high positions.
- **Effort**: Low (rewrite ~20 lines in rotary.rs)

**3. f64 Validation Pass to Triage Logic Bug vs Precision**
- **Why**: If running both Python and Rust in f64 produces correlation ~1.0, the entire gap is float32 accumulation. If it still diverges, there is a logic bug.
- **How**:
  1. In Python: Cast model to `.double()`, cast all inputs to float64, run generation.
  2. In Rust: Change all `DType::F32` to `DType::F64`, load weights as f64, run generation.
  3. Compare f64 outputs.
- **Expected impact**: Definitive answer on whether ~0.84 is the float32 ceiling or if there's a bug.
- **Effort**: Medium-High (requires modifying both Python and Rust pipelines)
- **Risk**: Some PyTorch operations may not support f64 on all codepaths.

### Worth Trying

**4. Capture and Compare Attention Weights at Step 0**
- **Why**: SDPA vs manual attention is the biggest suspected source of divergence. Comparing the attention weight matrices (post-softmax) between Python and Rust would quantify this directly.
- **How**:
  1. In Python, temporarily replace `F.scaled_dot_product_attention` with manual attention to get the attention weights matrix.
  2. Compare these weights to what Rust computes.
  3. Also compare the SDPA output to the manual attention output in Python alone -- this quantifies the SDPA vs manual gap.
- **Expected impact**: Directly measures the SDPA effect.
- **Effort**: Medium

**5. Replace Rust Manual Attention with Candle's SDPA (if available)**
- **Why**: If Candle has an SDPA implementation or if one can be added, it would reduce the divergence from Python.
- **How**: Check if `candle-flash-attn` or `candle-nn` has an SDPA function. If not, implement a tighter fused attention loop in Rust that matches SDPA's computation order.
- **Expected impact**: Could be significant if SDPA accounts for a large portion of the divergence.
- **Effort**: High

**6. Remove Debug Logging from Hot Path**
- **Why**: The `attention.rs` and `flowlm.rs` files contain extensive debug logging with `AtomicUsize` counters, tensor reads (`.to_vec1::<f32>()`), and conditional eprintln. These force Candle to materialize tensors that might otherwise be lazily evaluated, potentially affecting computation order and performance.
- **How**: Gate all debug logging behind `#[cfg(feature = "debug_tensors")]` or remove entirely.
- **Expected impact**: Likely zero for correctness, but improves performance and code clarity.
- **Effort**: Low

### Speculative

**7. Implement Two-Pass Attention: Compute in f64, Store in f32**
- **Why**: If the f64 validation pass (approach #3) shows correlation ~1.0, the gap is purely precision. Selectively computing attention in f64 (just the softmax and/or the Q@K^T matmul) while keeping everything else in f32 might close the gap without the full f64 overhead.
- **How**: Cast Q, K to f64 for the matmul, compute softmax in f64, cast result back to f32 before matmul with V.
- **Expected impact**: Unknown -- depends on whether softmax precision is the bottleneck.
- **Effort**: Low

**8. Pre-compute Python Attention Outputs for Validation**
- **Why**: Instead of trying to match SDPA exactly, capture the Python transformer's output at each autoregressive step and inject it into Rust as "ground truth conditioning" for FlowNet. This isolates FlowNet divergence from transformer divergence.
- **How**: Dump Python `out_norm` output at every step. In Rust, load these instead of computing through the transformer. Compare FlowNet output.
- **Expected impact**: Definitively separates transformer vs FlowNet contribution to the gap.
- **Effort**: Medium

---

## Already Tried (Don't Repeat)

From `docs/audit/approaches-tried.md`:

1. **Noise tensor off-by-one fix** (2026-03-18) -- MASSIVE IMPROVEMENT (0 -> 0.839). APPLIED.
2. **Switch to `softmax_last_dim()`** (2026-03-17) -- ZERO impact. APPLIED.
3. **Switch to `rope_i()` from candle_nn** (2026-03-17) -- ZERO impact. APPLIED.
4. **Layer-by-layer transformer comparison** (2026-03-17) -- DIAGNOSTIC only, led to noise offset discovery.

From KNOWLEDGE_INDEX.md, these are all FIXED and verified:
- RoPE interleaved vs split halves
- RoPE before vs after transpose
- LayerNorm vs RMSNorm for out_norm
- FlowNet sinusoidal order, activation, AdaLN chunk order, SiLU placement
- LSD time progression (two time values)
- SEANet activation (ELU not GELU)
- Voice conditioning concatenation order
- Two-phase forward pass
- FinalLayer missing norm_final
- SEANet output tanh removal
- FlowNet TimeEmbedding RMSNorm
- FlowNet time embedding addition
- Latent denormalization
- Weight loading -- VERIFIED CORRECT

---

## Specific Questions to Investigate

1. **What is the MAE between Python's `F.scaled_dot_product_attention` output and manual attention (Q@K^T*scale+mask -> softmax -> @V) when run on the SAME inputs in Python?** This quantifies the SDPA vs manual attention divergence ceiling.

2. **Does the precomputed cos/sin cache in Rust exactly match Python's on-the-fly computation for positions 0-200?** Dump and compare.

3. **Is there an off-by-one in the causal mask?** Python: `shift = num_keys - num_queries` (overrides the passed shift argument). Rust: `if j <= i + (kv_len - q_len)`. Verify these produce identical masks for the specific (q_len, kv_len) values used during generation.

4. **Does `Tensor::cat` for KV cache produce bit-identical results to Python's scatter-based cache?** Dump K/V from cache at step 0 and compare.

5. **At step 0, does `input_linear(bos_emb)` match between Python and Rust?** This should be exact since both use the same weights and same input. If it diverges, there's a weight loading issue.

6. **How does Python handle the NaN -> bos_emb replacement?** Python: `sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)`. Rust: starts with `bos_emb.unsqueeze(0).unsqueeze(0)`. These should be equivalent but verify the tensor values match.

---

## Useful Links & References

- [Kyutai Moshi Rust transformer source](https://github.com/kyutai-labs/moshi/blob/main/rust/moshi-core/src/transformer.rs) - Reference attention implementation with `rope_i()`, `softmax_last_dim()`, `matmul_dtype()`
- [Candle precision issue #3032](https://github.com/huggingface/candle/issues/3032) - MSE ~4e-10 per matmul, MAE ~1.5e-5
- [Candle divergence issue #2031](https://github.com/huggingface/candle/issues/2031) - Expected divergence in autoregressive generation
- [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) - Math backend keeps all intermediates in input dtype
- [PyTorch SDPA different output issue #119188](https://github.com/pytorch/pytorch/issues/119188) - Different output between reference and fused implementations
- [Candle porting tutorial](https://github.com/ToluClassics/candle-tutorial) - Recommended verification techniques
- [Kyutai Pocket TTS official](https://github.com/kyutai-labs/pocket-tts) - Python source for `modules/transformer.py`, `modules/rope.py`, `modules/mlp.py`
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

---

## Priority Action Plan

| Priority | Action | Expected Impact | Effort | Confidence |
|----------|--------|-----------------|--------|------------|
| 1 | Step-0 per-layer intermediate dump | Identifies root cause | Medium | Very High |
| 2 | Match Python RoPE frequency computation | Eliminates freq precision gap | Low | Medium |
| 3 | f64 validation pass | Logic bug vs precision triage | Medium-High | Very High |
| 4 | Compare SDPA vs manual attention in Python | Quantifies SDPA ceiling | Medium | High |
| 5 | Remove debug logging from hot path | Clean code | Low | Low |
| 6 | Higher-precision attention (f64 softmax) | Close precision gap | Low | Low-Medium |

---

## Summary of Status

The 0.839 correlation is a strong result that may be approaching the practical ceiling for Candle-vs-PyTorch autoregressive transformers. The key unknown is whether the remaining 16% gap is:

**(a) Inherent framework precision** -- float32 accumulation differences between Candle and PyTorch, compounded across 6 layers x ~45 autoregressive steps. This would be validated by the f64 experiment (approach #3). If f64 gives ~1.0, the gap is purely precision and may be unclosable without selective f64 computation.

**(b) A specific implementation difference** -- such as SDPA vs manual attention, RoPE frequency precision, or LayerNorm computation order. This would be identified by the step-0 intermediate dump (approach #1).

**(c) A combination** -- some small logic difference that's amplified by precision accumulation.

The recommended next step is approach #1 (step-0 intermediate dump) because it answers whether there's a single big divergence source or a distributed precision issue. This determines whether approaches #2-#6 are worth pursuing.

*Previous report archived as research-advisor-report-2.md*
