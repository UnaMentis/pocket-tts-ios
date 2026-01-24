# Research Advisor Briefing

**Date:** 2026-01-24
**Current Blocker:** Autoregressive generation step 0 diverges from Python despite text phase matching perfectly
**Research Focus:** KV cache handling, single-token attention, Candle precision, Pocket TTS architecture

---

## Situational Summary

The Rust/Candle port of Kyutai Pocket TTS has made significant progress with 11 architectural issues fixed (tokenization, RoPE, LayerNorm, FlowNet, SEANet, voice conditioning, etc.). Correlation improved from 0.0016 to ~0.01, but target is >0.95.

**Critical Discovery:** The text processing phase (7 tokens through transformer with 125-position voice KV cache) produces outputs that match Python exactly at every layer. However, the very next step—**step 0 of autoregressive latent generation** (single BOS token through transformer with 132-position KV cache)—diverges immediately:
- Python step 0 hidden: `[-0.24085297, -0.47014824, 0.23839632, 1.0895451, ...]`
- Rust step 0 hidden: `[-0.2274104, -0.35906476, 0.05141802, 0.5091589, ...]`

This is not a gradual drift—it's a significant mismatch from the first autoregressive step. The divergence occurs when transitioning from multi-token (7-token text) to single-token (1-token BOS) attention over the same KV cache.

---

## Key Research Findings

### From Official Kyutai Source

- **Architecture**: FlowLM is a 6-layer transformer (~70M params) at 12.5 Hz frame rate with 16 audio tokens per frame
- **Voice Conditioning**: Audio prompt → `MimiModel.encode_to_latent()` → project via `speaker_proj_weight` → initialize via `_run_flow_lm_and_increment_step()`
- **State Management**: Uses `StatefulModule` base class with cache dictionaries containing `current_end` (position) and `transformer_layer_N_self_attn_cache`
- **LSD Sampling**: Lagrangian Self-Distillation with 1-step integrator (`lsd_decode_steps = 1`)
- **Key URLs**:
  - [GitHub: kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
  - [HuggingFace Model](https://huggingface.co/kyutai/pocket-tts)
  - [DeepWiki Technical Overview](https://deepwiki.com/kyutai-labs/pocket-tts)

### From Reference Implementation (babybirdprd/pocket-tts)

- Published on [crates.io as pocket-tts-cli](https://crates.io/crates/pocket-tts-cli) with 1,309 SLoC
- Documentation mentions "Full-pipeline stateful streaming (KV-caching for Transformer, overlap-add for Mimi)" but provides no implementation details on porting challenges
- No documented issues or PRs about numerical divergence were found
- The repo claims near-identical output but doesn't document how divergence was debugged

### From Technical Deep-Dives

**Candle Precision Issues** ([GitHub Issue #3032](https://github.com/huggingface/candle/issues/3032)):
- Confirmed precision differences between Candle and PyTorch for `matmul` and linear layers
- MSE ~0.0000000004, MAE ~0.00001550 for F32 on CPU
- Maintainer notes: "floating point addition and multiplication are not associative"
- **No fix available** - this is inherent to different computation orderings
- Error accumulates across layers until outputs diverge

**Prefill vs Decode Phase Differences**:
- Prefill (multi-token): Highly parallelized, compute-bound, write-intensive to KV cache
- Decode (single-token): Sequential, memory-bound, read-intensive from KV cache
- The attention computation fundamentally differs: prefill uses matrix-matrix ops, decode uses matrix-vector ops
- Position tracking is critical: new queries must align with KV cache positions

---

## Suggested Approaches

### High Confidence

**1. Verify BOS Embedding Input for Step 0**
- Why: The divergence starts at the very first autoregressive step, suggesting the INPUT might be different
- How:
  1. Add Python hook to capture the exact tensor fed into layer 0 at step 0 (BOS embedding after projection)
  2. Compare with Rust's `bos_projected` tensor entering the transformer
  3. Check: Is this the same BOS embedding used during text phase, or is it fetched differently?

**2. Compare RoPE Offset at Step 0**
- Why: Python debug shows offset should be 132 (125 voice + 7 text). Rust logs show `RoPE offset: 132` but needs verification
- How:
  1. In Python, hook RoPE and log the position indices used for step 0
  2. Verify Rust uses same offset (132) and same frequency computation
  3. Check: Do cos/sin tables match exactly at position 132?

**3. Verify KV Cache State Before Step 0**
- Why: If text phase is correct, KV cache should contain valid K/V for positions 0-131. Any corruption would cause step 0 to diverge
- How:
  1. After text phase, dump K[131] and V[131] (last text position) in both Python and Rust
  2. Compare these values - they should match since text phase matches
  3. If they don't match, the cache update mechanism has a bug

### Worth Trying

**4. Check Attention Scale Factor Application**
- Why: Rust applies scale as `f64` cast (`self.scale as f64`), PyTorch may handle differently
- How:
  1. Log pre-scale and post-scale attention scores for step 0
  2. Verify scale = 1/sqrt(64) = 0.125 in both implementations
  3. Try: Apply scale before matmul vs after (order can affect precision)

**5. Verify Causal Mask for Single-Token Query**
- Why: With seq_len=1 and kv_len=133, the causal mask logic might behave differently
- How:
  1. In Python, check if a causal mask is applied at step 0 (often not needed for single-token)
  2. In Rust, check if `causal_mask` parameter is being handled correctly for single-token case
  3. If Python skips masking and Rust applies an incorrect mask, outputs would differ

**6. BFloat16 vs Float32 Accumulation**
- Why: Python uses BFloat16, Rust uses Float32. While F32 is higher precision, the accumulation order differs
- How:
  1. Try running Python in F32 mode to compare
  2. Check if Candle's matmul accumulation order matches PyTorch's
  3. Consider: Would using lower precision in Rust paradoxically give better correlation?

### Speculative

**7. Investigate StatefulModule Position Tracking**
- Why: Python's StatefulModule tracks `current_end` for each module. If Rust's position tracking differs, RoPE would be wrong
- How: Trace `increment_steps()` calls in Python to understand exact position management

**8. Check for Off-by-One in KV Cache Length**
- Why: KV cache length feeds into RoPE offset. Off-by-one would cause position mismatch
- How: After text phase, verify `cache.seq_len()` returns exactly 132 in Rust

---

## Things That Have Been Tried (DO NOT REPEAT)

From PORTING_STATUS.md:
1. Fixed tokenization (SentencePiece)
2. Fixed RoPE (interleaved vs split halves, applied before transpose)
3. Fixed LayerNorm vs RMSNorm (eps=1e-5)
4. Fixed FlowNet (sinusoidal order, SiLU activation, AdaLN chunk order)
5. Fixed LSD time progression (two time values, averaged embeddings)
6. Fixed SEANet activation (ELU instead of GELU)
7. Fixed voice conditioning (concatenation vs addition, two-phase processing)
8. Fixed FinalLayer norm_final (LayerNorm before AdaLN modulation)
9. Fixed SEANet output (removed tanh)
10. Verified weight loading (manual matmul matches)
11. Verified all layer 0 text phase intermediates (all match Python)

---

## Specific Questions to Investigate

1. **What is the exact input to layer 0 at step 0?** (BOS embedding after projection - compare Python vs Rust)

2. **Does Python apply any special treatment for single-token queries in attention?** (Different code path for seq_len=1?)

3. **What are the exact K and V values at position 132 after step 0 completes?** (If step 0 diverges, subsequent positions will be wrong)

4. **Is there any state reset or reinitialization between text phase and generation phase?** (Could explain sudden divergence)

5. **Does Python's `_sample_next_latent()` do anything before calling the transformer?** (Pre-processing of BOS?)

---

## Useful Links & References

### Official Kyutai
- [GitHub: kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
- [HuggingFace: kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
- [DeepWiki Technical Overview](https://deepwiki.com/kyutai-labs/pocket-tts)
- [Blog: Pocket TTS Release](https://kyutai.org/blog/2026-01-13-pocket-tts)

### Reference Implementation
- [GitHub: babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts)
- [crates.io: pocket-tts-cli](https://crates.io/crates/pocket-tts-cli)
- [docs.rs: pocket_tts](https://docs.rs/pocket-tts/latest/pocket_tts/)

### Candle & Technical
- [Candle Issue #3032: Precision differences](https://github.com/huggingface/candle/issues/3032)
- [Candle Layer Norm](https://github.com/huggingface/candle-layer-norm)
- [HuggingFace: KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching)
- [Medium: KV Caching Deep Dive](https://medium.com/@joaolages/kv-caching-explained-276520203249)
- [Tutorial: Porting PyTorch to Candle](https://github.com/ToluClassics/candle-tutorial)

### LSD / Flow Matching
- [ArXiv: Learning Flow Maps via Self-Distillation](https://arxiv.org/html/2505.18825)
- [ArXiv: Continuous Audio Language Models](https://arxiv.org/pdf/2509.06926)

---

## Key Insight

The breakthrough finding is that text phase works perfectly but step 0 diverges immediately. This strongly suggests the issue is **NOT** in the core transformer math (which is validated by text phase matching) but rather in:
- How the BOS token is prepared/projected for step 0
- How the transition from multi-token to single-token attention is handled
- How position tracking works between phases
- A subtle difference in how Python and Rust handle the decode phase vs prefill phase

**The implementation agent should focus debugging efforts on the exact moment of transition from text phase to generation phase, not on core transformer operations which are proven correct.**
