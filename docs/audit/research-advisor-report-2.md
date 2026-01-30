# Research Advisor Briefing

**Date:** 2026-01-24
**Current Blocker:** Hidden state divergence at step 36+ causing EOS detection mismatch; fundamental numerical differences between Candle and PyTorch
**Research Focus:** Root causes of autoregressive hidden state divergence, Candle vs PyTorch precision differences, cumulative error patterns in transformers

---

## Situational Summary

The implementation agent has made significant progress with the Rust/Candle port of Pocket TTS. **Short phrases (17 tokens) achieve 0.74-0.81 correlation**, and the pipeline produces intelligible speech. However, a critical issue has been identified:

**Hidden states diverge starting around step 36** of autoregressive generation. This is NOT just an EOS threshold issue - the actual hidden state vectors are fundamentally different:

| Python Step | Rust Step | Python EOS | Rust EOS | Delta |
|-------------|-----------|------------|----------|-------|
| 38          | 36        | -10.62     | -8.45    | +2.17 |
| 39          | 37        | -10.19     | -6.15    | +4.04 |
| 40          | 38        | -10.22     | -0.46    | +9.76 |
| 41          | 39        | -9.58      | +4.39    | +13.97|

The hidden state VALUES themselves are completely different vectors, even though their statistics (mean ~0, std ~0.35) appear similar. This explains why the EOS divergence follows - the `out_eos` linear head is just amplifying hidden state differences.

**Key contradiction:** PORTING_STATUS.md claims "All 42 latents match with cos_sim=1.0", but if hidden states diverge at step 36+, this can only be true for steps 0-35. The latent comparison may have been from an earlier version, or the FlowNet is somehow producing matching outputs despite hidden state differences (unlikely).

**Current state:**
- Frame count: Rust 43 vs Python 45 (2 frames short)
- Aligned correlation: 0.74 (target: >0.95)
- All components verified correct individually for short sequences
- Divergence appears to be cumulative, not from a single operation

---

## Key Research Findings

### From Official Kyutai Sources

**Architecture Confirmation** ([GitHub](https://github.com/kyutai-labs/pocket-tts), [DeepWiki](https://deepwiki.com/kyutai-labs/pocket-tts)):
- 6-layer transformer with 1024 hidden dim, 16 heads
- RoPE for position encoding (interleaved, not split halves)
- LSD (Lagrangian Self Distillation) for flow matching with 1 decode step in practice
- Default EOS threshold: -4.0
- Frame rate: 12.5 Hz (80ms per frame)

**Critical dtype handling** from Python source:
```python
# Explicit float32 conversion after transformer
transformer_out = transformer_out.to(torch.float32)
```
This ensures FlowNet receives float32 regardless of transformer's internal dtype.

### From Candle Precision Research

**Known Precision Discrepancies** ([GitHub Issue #3032](https://github.com/huggingface/candle/issues/3032)):
- MSE between Candle and PyTorch matmul: ~4.1e-10 to ~8.2e-11
- MAE: 0.000006 to 0.000015 per operation
- Maintainer confirms: "errors accrue layer by layer for each token"
- Root cause: Floating-point arithmetic isn't associative; different operation orders produce different results

**ToluClassics Tutorial** ([GitHub](https://github.com/ToluClassics/candle-tutorial)):
- Recommended: "almost identical results with +/-1% for floating point operations"
- Key pattern for numerical stability in LayerNorm:
```rust
let internal_dtype = match x_dtype {
    DType::F16 | DType::BF16 => DType::F32,
    d => d,
};
```

### From Autoregressive Error Propagation Research

**Closed-Loop Transformers Paper** ([arXiv:2511.21882](https://arxiv.org/html/2511.21882v1)):
> "Contemporary autoregressive transformers operate in open loop: each hidden state is computed in a single forward pass and never revised, causing **errors to propagate uncorrected through the sequence**."

Key insight: Once a hidden state is computed, any representational error propagates irreversibly through subsequent tokens. This manifests as:
- Cumulative hallucination in long-form generation
- Fragility under distribution shift
- Catastrophic failure in multi-step reasoning

**For our case:** Each step through the transformer introduces small precision errors. Over 36+ steps, these compound to produce fundamentally different hidden state vectors.

### From LayerNorm Precision Research

**PyTorch GitHub Issue #66707** and NVIDIA docs:
- LayerNorm needs to be done in fp32 for fp16 inputs, "otherwise overflow happens and there is a significant divergence that starts to add up over multiple chained uses"
- TensorRT warning: "Running layernorm after self-attention in FP16 may cause overflow. Forcing layernorm layers to run in FP32 precision can help with preserving accuracy."

**Accumulation Error Formula:**
- Unit round-off for IEEE-754 float32: ~1.19e-7
- Each layer introduces at most a relative slip
- Chaining L layers gives compound error through multiplication of (1+delta) terms

### From RoPE Position Encoding Research

**Known RoPE Issues** ([LearnOpenCV](https://learnopencv.com/rope-position-embeddings/), [EleutherAI](https://blog.eleuther.ai/rotary-embeddings/)):
- "RoPE's practical weak spots include **phase drift of fast clocks** and **floating-point precision**"
- At high positions (100+), sine/cosine computations may differ between frameworks
- Modern solutions (NTK, YaRN) address these for 128k+ context

For Pocket TTS at position 132+ (125 voice + 7 text + latents):
- Small differences in trigonometric function implementations compound
- Order of floating-point operations affects results
- Intermediate precision differences (BFloat16 vs Float32) matter

---

## Suggested Approaches

### High Confidence

**1. Layer-by-Layer Hidden State Comparison at Step 36** (DIAGNOSTIC PRIORITY)
- **Why:** Pinpoint exactly where divergence first appears within the transformer
- **Confidence:** Very High - essential diagnostic
- **How:**
  1. Add logging in Python to dump hidden state after EACH of the 6 transformer layers at step 36
  2. Add matching logging in Rust
  3. Compare: If layer 0 matches but layer 1 diverges, the issue is in layer 1's attention or MLP
  4. Continue narrowing: attention vs MLP, Q/K/V computation vs softmax, etc.
  5. The divergence point will reveal the root cause operation

**2. Force Float32 Throughout Transformer** (PRECISION FIX)
- **Why:** Python explicitly converts to float32 before FlowNet; Rust may not be doing equivalent
- **Confidence:** High - documented best practice
- **How:**
  1. Ensure all attention computations explicitly use Float32
  2. Ensure LayerNorm internal computations use Float32
  3. Ensure matmul accumulations use Float32
  4. Check: `transformer_out.to_dtype(DType::F32)?` before FlowNet

**3. Compare KV Cache at Step 36** (VERIFICATION)
- **Why:** KV cache stores all previous K/V values; if these drift, all future attention is affected
- **Confidence:** High - directly relevant to autoregressive accumulation
- **How:**
  1. Dump K cache for layer 0, positions 130-136 in both implementations
  2. Compare values to 6+ decimal places
  3. If K values diverge, the issue is in earlier attention computations
  4. If K values match but hidden states diverge, the issue is in current step's computation

### Worth Trying

**4. Verify RoPE at High Positions** (ISOLATION TEST)
- **Why:** RoPE is applied to every Q/K at every step; small errors compound
- **Confidence:** Medium - may not be the specific issue
- **How:**
  1. Compute RoPE for position=135 in both Python and Rust (standalone, not in model)
  2. Compare sine/cosine values to 8+ decimal places
  3. If they differ, focus on RoPE implementation
  4. Check for differences in: base frequency, dimension ordering, precision of powf/exp operations

**5. Use Python Weights on Rust Hidden States** (CROSS-VALIDATION)
- **Why:** Isolates whether weights or computation differs
- **Confidence:** Medium - useful diagnostic
- **How:**
  1. At step 36, save Rust hidden state to file
  2. Load in Python and run through Python's `out_eos` head
  3. If Python's EOS matches Python's original → Rust hidden state is the problem
  4. If Python's EOS on Rust hidden state differs → Rust hidden state + Python weights = different result (weight issue)

**6. Implement "Sync Point" at Step 35** (WORKAROUND)
- **Why:** Force Rust to use Python's hidden state, see if subsequent steps match
- **Confidence:** Medium - experimental
- **How:**
  1. Save Python hidden state at step 35
  2. Load in Rust and inject into KV cache
  3. Continue Rust generation from step 36
  4. If outputs match → confirms cumulative error theory
  5. If outputs still diverge → there's an operation-level bug

### Speculative

**7. Accept Current Correlation for MVP**
- **Why:** 0.74 correlation produces intelligible speech; iOS users may not notice the difference
- **Confidence:** Pragmatic fallback
- **Tradeoffs:**
  - Short phrases work excellently (0.81 correlation)
  - Longer phrases produce slightly shorter audio (2 frames fewer)
  - Audio quality is still acceptable for many use cases
  - Document limitation and ship MVP while continuing investigation

**8. Try Alternative Attention Implementation**
- **Why:** Candle's softmax may accumulate differently than PyTorch's
- **Confidence:** Low - speculative
- **How:**
  1. Implement manual softmax with explicit Float64 accumulation
  2. Compare results to standard candle_nn::ops::softmax
  3. If different, use manual implementation

---

## Things That Have Been Tried (DO NOT REPEAT)

**Verified Correct:**
- Tokenization (SentencePiece) - exact match
- RoPE (interleaved pairs, applied before transpose) - verified at short positions
- LayerNorm (eps=1e-5, affine=true) - verified
- All 6 transformer layers - layer-by-layer verification passed for SHORT sequences
- FlowNet (sinusoidal order, SiLU, AdaLN, time embedding) - latent cosine similarity = 1.0 for short sequences
- Voice conditioning (concatenation, two-phase forward) - verified
- Latent denormalization - moved before Mimi, verified
- EOS threshold = -4.0 - matches Python DEFAULT_EOS_THRESHOLD
- Replicate padding for SEANet streaming convolutions
- Full streaming mode for all SEANet Conv1d layers
- Non-causal attention for Mimi decoder transformer
- min_gen_steps = 0 for natural EOS detection
- Mimi decoder order (upsample before transformer)
- Mimi decoder RoPE
- Upsample padding fix (no padding, then trim)

**Recent Session Activity:**
- Added `capture_hidden_states.py` for comparing hidden states at critical steps
- Added EOS trajectory logging at steps 36-42
- Verified hidden state statistics match but VALUES diverge

---

## Specific Questions to Investigate

1. **At which LAYER does the hidden state divergence first appear at step 36?**
   - Is it layer 0 (first attention), or does it compound through layers?
   - Does the attention output diverge, or the MLP output?

2. **Are the KV cache values at position 130+ identical between Rust and Python?**
   - If K/V drift in the cache, all subsequent attention is wrong
   - The cache stores values from ALL previous steps

3. **Does Candle's softmax implementation differ from PyTorch's for large attention matrices?**
   - At step 36, the attention matrix is [1, 16, 1, 168] (168 = 132 + 36)
   - Large matrices may accumulate differently

4. **Is there a dtype mismatch at any point in the Rust pipeline?**
   - Python explicitly converts to float32 before FlowNet
   - Are all Rust operations using consistent precision?

5. **Could the BOS embedding handling introduce drift?**
   - Python uses NaN replacement; Rust may compute differently
   - Small differences in BOS handling propagate through all 36 steps

---

## Useful Links & References

### Official Kyutai
- [GitHub: kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) - Python reference
- [GitHub: kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) - Production Rust Mimi implementation
- [DeepWiki: pocket-tts](https://deepwiki.com/kyutai-labs/pocket-tts) - Architecture analysis
- [HuggingFace: kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) - Model card

### Precision and Numerical Stability
- [Candle Issue #3032: Precision differences](https://github.com/huggingface/candle/issues/3032) - Direct evidence of matmul differences
- [ToluClassics Candle Tutorial](https://github.com/ToluClassics/candle-tutorial) - Porting best practices
- [PyTorch Issue #66707: LayerNorm fp32](https://github.com/pytorch/pytorch/issues/66707) - LayerNorm precision
- [PyTorch Numerical Accuracy Docs](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) - Official guidance

### Autoregressive Error Propagation
- [Closed-Loop Transformers Paper](https://arxiv.org/html/2511.21882v1) - Error propagation theory
- [Lil'Log: LLM Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) - KV cache details

### RoPE Position Encoding
- [LearnOpenCV: Inside RoPE](https://learnopencv.com/rope-position-embeddings/) - Practical guide
- [EleutherAI: Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/) - Numerical precision issues

### Local Documentation (Already in Repo!)
- `docs/python-reference/ARCHITECTURE/flowlm.md` - FlowLM architecture details
- `docs/python-reference/MODULES/rope.md` - RoPE implementation
- `docs/python-reference/MODULES/transformer.md` - Transformer attention details

---

## Critical Insight

**This is a cumulative precision error problem, not an architectural bug.**

The evidence strongly suggests:
1. Individual operations are correct (verified for short sequences)
2. Small precision differences between Candle and PyTorch exist (~1e-7 to 1e-5 per operation)
3. These differences compound across 6 layers x 36 steps = 216 layer applications
4. By step 36, the accumulated error produces measurably different hidden states
5. The EOS divergence is a symptom, not the cause

**The fundamental limitation:** Candle and PyTorch will NEVER produce bit-identical results for deep networks with many autoregressive steps. The question is whether we can reduce the divergence enough to match EOS timing, or accept the current correlation and ship.

---

## Recommended Next Steps

1. **Immediate (Diagnostic):**
   - Add per-layer hidden state logging at step 36 in both implementations
   - Compare to identify first divergent layer
   - If layer 0 diverges, focus on attention or MLP
   - If later layers diverge more, it's cumulative error

2. **If cumulative error confirmed:**
   - Force Float32 throughout transformer
   - Consider double-precision (Float64) for critical accumulations (softmax, layernorm)
   - Accept that perfect match is impossible

3. **Fallback (MVP):**
   - Ship with current 0.74 correlation
   - Document max recommended phrase length
   - Implement chunking at application level for longer content
   - The audio is intelligible and production-ready for many use cases

**Time estimate:** Diagnostic (1 session), precision fixes (1-2 sessions), or accept limitation and document.

---

*Previous report archived as research-advisor-report-2.md*
