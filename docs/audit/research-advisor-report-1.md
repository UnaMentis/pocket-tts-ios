# Research Advisor Briefing

**Date:** 2026-01-24
**Current Blocker:** Waveform correlation ~0.13 (aligned) vs target >0.95; batch processing in Mimi decoder differs fundamentally from Python streaming
**Research Focus:** Kyutai Moshi Rust implementation patterns, streaming convolution state management, frame-by-frame processing architecture

---

## Situational Summary

The Rust/Candle port of Pocket TTS has achieved **perfect latent generation** - all 42 latents match Python with cosine similarity = 1.0. The FlowLM transformer, FlowNet, and latent denormalization are verified correct. The audio output is **intelligible and recognizable** with amplitude within 73% of reference.

**The sole remaining blocker is the Mimi decoder's streaming behavior.** The verification report shows:
- Latent cosine similarity: **1.0** (perfect)
- Waveform correlation (aligned): **0.1285** (target: >0.95)
- Max amplitude: 0.44 (Rust) vs 0.61 (Python)
- Real-time factor: 3.17x (good performance)

**Key architectural gap**: Python processes one latent frame at a time through the *entire* Mimi pipeline with persistent state. The current Rust implementation has streaming infrastructure (state structs, `forward_streaming` methods) but the **calling pattern is wrong**:

- **Current Rust**: All latents → batch upsample → batch transformer → SEANet with streaming convtr
- **Python pattern**: For each latent → streaming upsample → streaming transformer (KV cache) → streaming SEANet

The streaming convolution code exists and appears correct (`forward_streaming` with overlap-add and bias subtraction). The issue is how it's invoked.

---

## Key Research Findings

### From Kyutai Moshi Rust Implementation (PRIMARY SOURCE)

The [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) repository contains production Rust code for streaming Mimi in `rust/moshi-core/src/conv.rs`:

**StreamableConv1d pattern:**
```
- state_prev_xs: StreamTensor (previous input buffer)
- left_pad_applied: bool (first frame flag)
- Causal mode: pad1d(xs, padding_total, 0) left-only on first frame
- xs.narrow() and xs.split() manage active window vs state
```

**StreamableConvTranspose1d pattern:**
```
- state_prev_ys: StreamTensor (output history buffer for overlap-add)
- offset = num_frames * stride separates valid output from state
- invalid_steps = kernel_size - stride
- Output split: ot - invalid_steps for valid results
```

**Key insight from Moshi**: They use `StreamTensor::cat2()` for concatenation and `mask.where_cond()` for variable-length processing. The pattern is cleaner than the current Rust implementation.

**rustymimi** ([PyPI](https://pypi.org/project/rustymimi/)) provides Python bindings to the Rust Mimi - this confirms the Rust implementation works correctly in production.

### From Pocket TTS Documentation

The [DeepWiki analysis](https://deepwiki.com/kyutai-labs/pocket-tts) confirms:
- `StatefulModule` pattern with dictionaries containing "current_end" position tracking
- Multithreaded architecture: `latents_queue` → `decode_from_latent()` → `result_queue`
- State includes module-specific KV caches for the decoder transformer
- Frame rate: 12.5 Hz (80ms per frame)

### From Streaming Audio Research

[Streaming Speech Decoder Architectures](https://www.emergentmind.com/topics/streaming-speech-decoder) confirms:
- KV caching is essential for streaming transformers
- Chunk-based processing with causal masking
- Buffer management separates completed output from pending state

[KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching):
- "At each generation step we are recalculating the same previous token attention"
- KV cache: "only calculate attention for the new token"
- Cache grows linearly with sequence length

### From Candle Porting Guides

[ToluClassics Tutorial](https://github.com/ToluClassics/candle-tutorial) recommends:
- **Unit test every module** for shape consistency
- **Load PyTorch weights directly** for comparison (±1% for floating point)
- **Use F32 for internal computations** in LayerNorm for numerical stability
- Save PyTorch weights to safetensor for cross-framework testing

---

## Suggested Approaches

### High Confidence

**1. Implement True Frame-by-Frame Processing in MimiDecoder** (HIGHEST PRIORITY)
- **Why:** The streaming code EXISTS but is called wrong. This is the last mile.
- **Confidence:** Very High - the algorithms are already implemented in `src/models/mimi.rs`
- **How:**
  1. In `forward_true_streaming()`, process one latent at a time through:
     - `output_proj` (k=1, stateless - already works)
     - `upsample_streaming` using `StreamableConvTranspose1d::step()` (already implemented!)
     - `decoder_transformer.forward_streaming()` with KV cache (already implemented!)
     - `seanet.forward_streaming()` with overlap-add (already implemented!)
  2. **Critical**: Ensure state persists across ALL iterations in the loop
  3. The `forward_true_streaming` method at line 998 already attempts this but uses batch SEANet
  4. **Fix**: Make SEANet also use streaming convolutions per-frame

**2. Verify Streaming State Persistence Across Frames**
- **Why:** State may be getting reset between frames
- **Confidence:** High - common bug in streaming implementations
- **How:**
  1. Add debug logging of state buffer values between frames
  2. Verify `state.partial` accumulates (not zeros after each frame)
  3. Compare partial buffer values to Python intermediates
  4. Use `validation/dump_intermediates.py` to capture Python state between frames

**3. Study Kyutai's moshi-core/src/conv.rs Directly**
- **Why:** Production-quality reference implementation
- **Confidence:** Very High - it's the authoritative source
- **How:**
  1. Clone [github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
  2. Read `rust/moshi-core/src/conv.rs` line by line
  3. Compare `StreamableConvTranspose1d::step()` implementation
  4. Note the `StreamTensor` pattern - cleaner than Option<Tensor>
  5. Adapt patterns to current codebase

### Worth Trying

**4. Process Just SEANet Frame-by-Frame (Partial Streaming)**
- **Why:** Isolate whether SEANet is the bottleneck
- **Confidence:** Medium - might give partial improvement
- **How:**
  1. Keep batch for: output_proj, upsample, decoder_transformer
  2. Split output into 16-sample chunks (one upsampled latent)
  3. Process each chunk through SEANet with streaming state
  4. Measure correlation improvement

**5. Compare Decoder Transformer KV Cache Behavior**
- **Why:** The transformer may need streaming even if attention is causal
- **Confidence:** Medium - Python passes `mimi_state` to transformer
- **How:**
  1. Check if Python's decoder_transformer uses KV cache
  2. Verify `DecoderTransformerLayer::forward_streaming()` KV cache logic
  3. Ensure offset calculation is correct for RoPE

### Speculative

**6. Accept 0.64+ Correlation as "Good Enough"**
- **Why:** Audio is intelligible, amplitude is reasonable
- **Confidence:** Pragmatic fallback
- **Tradeoffs:**
  - Pro: Ship sooner, audio works
  - Con: Doesn't match reference perfectly, may have subtle artifacts
  - Consider: For mobile use cases, might be acceptable

---

## Things That Have Been Tried (DO NOT REPEAT)

**Verified Correct (from verification-report-1.md):**
- Latent generation: cosine sim = 1.0
- FlowLM transformer: all 6 layers verified
- FlowNet: LSD decode correct
- Tokenization: exact match
- RoPE: interleaved, applied correctly
- LayerNorm: eps=1e-5

**Recent Fixes Applied:**
- Processing order: upsample BEFORE transformer (correct)
- Removed tanh activation from SEANet output
- Added streaming infrastructure (state structs, forward_streaming methods)
- StreamableConvTranspose1d with overlap-add and bias subtraction
- StreamableConv1d with causal context buffer

**What's Already Implemented (just needs correct calling):**
- `StreamingConv1dState` and `StreamingConvTr1dState` structs
- `Conv1d::forward_streaming()` - causal context buffer
- `ConvTranspose1d::forward_streaming()` - overlap-add with bias subtraction
- `DecoderTransformerLayer::forward_streaming()` - KV cache
- `MimiDecoder::forward_true_streaming()` - partial implementation

---

## Specific Questions to Investigate

1. **Does `forward_true_streaming` maintain state across the entire loop?**
   - Check line 1014: `upsample_streaming` is created fresh inside the method
   - Should state persist across ALL latent frames, not reset per-call?

2. **Is the SEANet being processed in streaming mode in forward_true_streaming?**
   - Line 1051: `self.seanet.forward(&x)` uses batch mode
   - Should be `self.seanet.forward_streaming(&x, &mut seanet_state)`

3. **Why doesn't forward_streaming match Python better than 0.13?**
   - The `forward_streaming` at line 864 processes chunks through SEANet with streaming
   - But upsampler is processed frame-by-frame THEN concatenated
   - Python likely processes each latent through ENTIRE pipeline with state

4. **What's the correct chunk size for SEANet streaming?**
   - Current: 16 samples (one upsampled latent frame)
   - Python: Processes frame-by-frame through entire SEANet
   - May need smaller or larger chunks

5. **Is the decoder transformer actually stateless in Python?**
   - Python passes `mimi_state` everywhere but might not use it for transformer
   - Verify if KV cache is needed or if batch mode is correct for transformer

---

## Useful Links & References

### Official Kyutai
- [GitHub: kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) - **Production Rust Mimi** (PRIMARY SOURCE)
- [rust/moshi-core/src/conv.rs](https://github.com/kyutai-labs/moshi/blob/main/rust/moshi-core/src/conv.rs) - Streaming conv implementation
- [GitHub: kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) - Python reference
- [PyPI: rustymimi](https://pypi.org/project/rustymimi/) - Python bindings for Rust Mimi
- [Kyutai TTS Blog](https://kyutai.org/blog/2026-01-13-pocket-tts) - Technical overview
- [Kyutai TTS Technical Report](https://kyutai.org/pocket-tts-technical-report) - Architecture details

### DeepWiki Analysis
- [DeepWiki: kyutai-labs/pocket-tts](https://deepwiki.com/kyutai-labs/pocket-tts) - StatefulModule pattern details

### Streaming Audio Research
- [Streaming Speech Decoder Architectures](https://www.emergentmind.com/topics/streaming-speech-decoder) - Overview
- [KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching) - HuggingFace tutorial
- [cached_conv](https://acids-ircam.github.io/cached_conv/) - Alternative streaming approach

### Candle Porting
- [ToluClassics Tutorial](https://github.com/ToluClassics/candle-tutorial) - PyTorch to Candle guide
- [Rust for AI 2025](https://markaicode.com/rust-ai-frameworks-candle-pytorch-comparison-2025/) - Framework comparison

### Local Documentation (Already in Repo!)
- `docs/python-reference/STREAMING/conv-transpose-overlap-add.md` - **ROOT CAUSE DOCUMENT**
- `docs/python-reference/STREAMING/conv1d-streaming.md` - Causal convolution
- `docs/python-reference/STREAMING/state-management.md` - StatefulModule pattern

---

## Critical Insight

**The streaming code is written. The algorithms are implemented. The issue is the integration.**

Current `forward_true_streaming()` creates fresh streaming state inside the method, processes latents one at a time through upsample and transformer (good!), but then uses batch mode for SEANet (bad!).

**The fix is straightforward:**
1. Create SEANet streaming state ONCE at the start
2. Process EACH latent through the ENTIRE pipeline including streaming SEANet
3. Ensure ALL state buffers persist across the loop

This is ~20 lines of code change, not a major refactor. The hard work (implementing streaming convolutions) is already done.

---

## Recommended Next Steps

1. **Read `forward_true_streaming()` carefully** (lines 998-1071 in mimi.rs)
2. **Identify where SEANet switches to batch mode** (line 1051)
3. **Create persistent SEANet streaming state** before the loop
4. **Call `seanet.forward_streaming()` inside the loop** with persistent state
5. **Verify with intermediate dumps** against Python frame-by-frame output

Time estimate: Implementation agent should be able to fix this in one focused session.
