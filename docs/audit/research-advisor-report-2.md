# Research Advisor Briefing

**Date:** 2026-01-24
**Current Blocker:** EOS detection diverges for longer phrases (21 frames early); Short phrase correlation 0.81, Long phrase correlation 0.05
**Research Focus:** EOS logit divergence, numerical precision accumulation, RoPE position drift at high positions, Candle vs PyTorch precision differences

---

## Situational Summary

The Rust/Candle port of Pocket TTS has achieved **excellent results for short phrases** - 0.81 waveform correlation with replicate padding implemented. The FlowLM transformer, FlowNet, latent generation, and Mimi SEANet streaming are all working. Audio is intelligible and sounds nearly identical to Python for short inputs.

**The new blocker is EOS detection divergence for longer sequences.** Testing revealed:

| Phrase Type | Tokens | Rust Frames | Python Frames | Correlation | Status |
|-------------|--------|-------------|---------------|-------------|--------|
| Short (17 tokens) | 17 | 43 | 43 | **0.81** | Working |
| Medium (~50 tokens) | ~50 | 145 | ~166 | **0.05** | EOS mismatch |
| Long (>100 tokens) | >100 | N/A | N/A | N/A | Exceeds 4096 KV limit |

**Key finding:** The EOS threshold is correctly set to `-4.0` in both implementations, but the underlying EOS logit distributions diverge over longer sequences. Rust triggers EOS 21 frames (~1.7 seconds) earlier than Python for medium-length inputs.

**Root cause hypotheses:**
1. Numerical precision accumulation over many transformer forward passes
2. KV cache differences at higher position indices
3. RoPE position encoding drift at longer sequences (positions 100+)
4. Float32 vs BFloat16 accumulation patterns

**Additional discovery:** The decoder transformer has a max sequence length of 4096 frames. With 16x upsample, this limits audio to ~20 seconds maximum (256 latents).

---

## Key Research Findings

### From Official Kyutai Documentation

**EOS Detection Mechanism** ([Kyutai docs](https://github.com/kyutai-labs/pocket-tts/blob/main/docs/generate.md)):
- Default EOS threshold: **-4.0** (verified in `default_parameters.py`)
- EOS is predicted by `out_eos` linear head: `out_eos(transformer_out) > eos_threshold`
- `frames_after_eos` parameter controls post-EOS generation (default: auto-calculated)
- Each frame is 80ms of audio

**Autoregressive Generation** ([DeepWiki analysis](https://deepwiki.com/kyutai-labs/pocket-tts)):
- `FlowLMModel._sample_next_latent()` generates at 12.5 Hz until EOS
- Multiple KV caches: FlowLM transformer (6 layers) + Mimi decoder transformer (2 layers)
- State accumulates through `current_end` position tracking

### From RoPE Position Encoding Research

**RoPE Numerical Stability Issues** ([EleutherAI Blog](https://blog.eleuther.ai/rotary-embeddings/), [LearnOpenCV](https://learnopencv.com/rope-position-embeddings/)):
- RoPE's practical weak spots include **phase drift of fast clocks** and **floating-point precision**
- Length generalization is a known pain point - models degrade when pushed beyond training context length
- Modern solutions (NTK, YaRN, sliding-window) address these issues for 128k+ context

**Key insight:** At position 150+ in the KV cache, RoPE's sine/cosine values may be computed differently between PyTorch and Candle due to:
1. Order of floating-point operations
2. Intermediate precision (BFloat16 vs Float32)
3. Trigonometric function implementations

### From Candle Framework Research

**PyTorch to Candle Precision** ([ToluClassics Tutorial](https://github.com/ToluClassics/candle-tutorial)):
- Expected difference: "almost identical results with ±1% for floating point operations"
- Recommendation: "Use F32 for internal computations in LayerNorm for numerical stability"
- Weight loading verified - differences compound over many operations

**Known Candle Limitation:**
- Candle uses different BLAS backends than PyTorch
- Floating-point operation order may differ
- No built-in `torch.compile()` equivalent for fusion

### From TTS Stop Detection Research

**EOS Detection Challenges** ([MELLE paper](https://arxiv.org/html/2412.06602v1)):
- "Each utterance has only one positive frame indicating 'stop'" - extreme class imbalance
- Small changes to logit distributions can dramatically shift stop timing
- EOS head sensitivity increases with sequence length

---

## Suggested Approaches

### High Confidence

**1. Compare EOS Logit Trajectories Between Rust and Python** (DIAGNOSTIC PRIORITY)
- **Why:** This will reveal exactly WHERE and HOW MUCH the logits diverge
- **Confidence:** Very High - essential diagnostic
- **How:**
  1. Add logging to dump EOS logit at EVERY step in both Rust and Python
  2. Use the same medium-length phrase in both
  3. Plot logit values vs step number
  4. Identify the step where divergence begins (likely around step 40-60)
  5. The divergence pattern will indicate root cause:
     - Linear drift → accumulating precision errors
     - Sudden jump → specific layer/operation mismatch
     - Oscillating → RoPE phase issues

**2. Force Float32 Throughout FlowLM Transformer** (PRECISION FIX)
- **Why:** Python may use BFloat16 internally, Candle may have different defaults
- **Confidence:** High - common source of divergence
- **How:**
  1. Check current dtype in KV cache creation
  2. Ensure all attention computations use Float32
  3. Convert outputs explicitly: `transformer_out.to_dtype(DType::F32)?`
  4. Pay special attention to softmax and LayerNorm operations

**3. Verify RoPE Computation at High Positions** (VERIFICATION)
- **Why:** RoPE errors compound over long sequences
- **Confidence:** High - known issue area
- **How:**
  1. Compute `rope_freqs(position=150)` in both Rust and Python
  2. Compare sine/cosine values to 6+ decimal places
  3. Check if interleaved dimension ordering is consistent
  4. Verify head dimension ordering in reshape operations

### Worth Trying

**4. Use Python EOS Detection on Rust Hidden States** (ISOLATION TEST)
- **Why:** Isolates whether the issue is in EOS head or hidden state generation
- **Confidence:** Medium - may reveal root cause
- **How:**
  1. Save Rust hidden states at each step to file
  2. Load in Python and run through Python's `out_eos` head
  3. Compare predicted EOS steps
  4. If they match Python's original → issue is in hidden state generation
  5. If they don't match → issue is in EOS head weights/computation

**5. Adjust EOS Threshold for Rust** (PRACTICAL WORKAROUND)
- **Why:** If logits are systematically lower, a lower threshold may compensate
- **Confidence:** Medium - workaround, not fix
- **How:**
  1. Based on diagnostic from Approach 1, calculate average logit offset
  2. Adjust Rust threshold: `eos_threshold = -4.0 + offset`
  3. Test across multiple phrase lengths
  4. Document as known difference

**6. Implement Explicit Float32 Accumulation in Attention** (PRECISION FIX)
- **Why:** Attention score accumulation is where precision matters most
- **Confidence:** Medium - may not be the specific issue
- **How:**
  1. In `src/modules/attention.rs`, force QK matmul to Float32
  2. Keep softmax in Float32 throughout
  3. Only downcast after attention output is computed

### Speculative

**7. Accept Short-Phrase-Only Support for iOS**
- **Why:** Short phrases work well (0.81 correlation), iOS use cases often involve short utterances
- **Confidence:** Pragmatic fallback
- **Tradeoffs:**
  - Document max recommended phrase length (~40 tokens, ~3 seconds)
  - Long texts would need chunking at application level
  - Audio quality for short phrases is excellent

---

## Things That Have Been Tried (DO NOT REPEAT)

**Verified Correct:**
- Tokenization (SentencePiece) - exact match
- RoPE (interleaved, applied before transpose) - verified at short positions
- LayerNorm (eps=1e-5) - verified
- All 6 transformer layers - layer-by-layer verification passed for short sequences
- FlowNet (sinusoidal order, SiLU, AdaLN) - latent cosine similarity = 1.0
- Voice conditioning (concatenation, two-phase forward) - verified
- Latent denormalization - moved before Mimi, verified
- EOS threshold = -4.0 - matches Python DEFAULT_EOS_THRESHOLD exactly

**Recent Fixes Applied:**
- Replicate padding for SEANet streaming convolutions
- Full streaming mode for all SEANet Conv1d layers
- Non-causal attention for decoder transformer
- min_gen_steps = 0 for natural EOS detection

**Current Implementation Status:**
- Short phrases: 0.81 correlation - WORKING
- Audio quality: Intelligible, nearly identical to Python
- Real-time factor: ~4x on CPU - GOOD

---

## Specific Questions to Investigate

1. **At what step does the EOS logit divergence begin?**
   - Is it gradual (precision accumulation) or sudden (specific operation)?
   - Does it correlate with KV cache filling past a certain threshold?

2. **What is the EOS logit trajectory for Python on the medium phrase?**
   - Does Python's EOS logit stay below -4.0 until step ~166?
   - How close to the threshold does it get at step 145 (when Rust triggers)?

3. **Is the hidden state divergence uniform across all dimensions?**
   - Some dimensions may diverge more than others
   - The EOS head's weight distribution may amplify certain dimensions

4. **Does the KV cache show divergence at high positions?**
   - Compare K[position=100] and V[position=100] between implementations
   - Check if older cached values drift due to accumulated operations

5. **Would using a sliding window attention help?**
   - If KV cache position is the issue, limiting to recent context may help
   - Pocket TTS may not need full context for later generation steps

---

## Useful Links & References

### Official Kyutai
- [GitHub: kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) - Python reference
- [docs/generate.md](https://github.com/kyutai-labs/pocket-tts/blob/main/docs/generate.md) - EOS parameters
- [GitHub: kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) - Production Rust Mimi
- [Kyutai TTS Blog](https://kyutai.org/blog/2026-01-13-pocket-tts) - Technical overview
- [HuggingFace: kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) - Model card

### Precision and Numerical Stability
- [ToluClassics Tutorial](https://github.com/ToluClassics/candle-tutorial) - PyTorch to Candle porting
- [KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching) - HuggingFace tutorial
- [Codegenes: KV Cache PyTorch](https://www.codegenes.net/blog/kv-cache-pytorch/) - Implementation details

### RoPE Position Encoding
- [EleutherAI: Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/) - Original blog
- [LearnOpenCV: Inside RoPE](https://learnopencv.com/rope-position-embeddings/) - Practical guide
- [RoFormer Paper](https://arxiv.org/abs/2104.09864) - Original paper

### Streaming Audio
- [Andrew Gibiansky: Streaming Audio Synthesis](https://andrew.gibiansky.com/streaming-audio-synthesis/) - Overlap-add tutorial
- [cached_conv Library](https://acids-ircam.github.io/cached_conv/) - Post-training streaming

### Local Documentation (Already in Repo!)
- `docs/python-reference/ARCHITECTURE/flowlm.md` - FlowLM architecture
- `docs/python-reference/MODULES/rope.md` - RoPE implementation
- `docs/python-reference/STREAMING/` - Streaming convolution algorithms

---

## Critical Insight

**The implementation is correct for short phrases (0.81 correlation). The EOS divergence for longer phrases is likely a precision accumulation issue, not an architectural bug.**

The path forward is:
1. **Diagnostic first**: Log EOS logits at every step to understand the divergence pattern
2. **Targeted fix**: Based on pattern, apply precision fixes to the specific operations causing drift
3. **Fallback**: If unfixable, document max phrase length and implement chunking at app level

For iOS use cases, short phrases (notifications, UI feedback, brief messages) work excellently. Longer content can be chunked at the application layer if needed.

---

## Recommended Next Steps

1. **Immediate**: Add step-by-step EOS logit logging to both Rust and Python
2. **Compare**: Generate same medium phrase, plot logit trajectories
3. **Identify**: Find the divergence inflection point
4. **Fix**: Apply targeted precision improvements to the diverging operations
5. **Validate**: Re-test with various phrase lengths

Time estimate: Diagnostic (1 session), fix attempt (1-2 sessions), or accept limitation and document.
