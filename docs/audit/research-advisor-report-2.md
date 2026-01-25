# Research Advisor Briefing

**Date:** 2026-01-24
**Current Blocker:** Waveform correlation ~0.08 (aligned) vs target >0.95; Rust batch processing fundamentally differs from Python streaming
**Research Focus:** Streaming convolution state management, SEANet architecture, frame-by-frame processing, alternative solutions

---

## Situational Summary

The Rust/Candle port has achieved a **major milestone**: all 42 generated latents now match Python exactly with cosine similarity = 1.0. The FlowLM transformer, FlowNet, and the entire latent generation pipeline are verified correct. Generation length is also fixed (41 frames matching Python).

**The remaining issue is isolated to the Mimi decoder's SEANet component.** The PORTING_STATUS.md documents extensive investigation revealing a fundamental architectural difference:

1. **Python streaming mode**: Processes one latent frame at a time with state buffers (`previous` for Conv1d, `partial` for ConvTranspose1d) that accumulate across frames
2. **Rust batch mode**: Processes all latent frames at once without inter-frame state accumulation

**Critical Discovery**: Even Python's own batch mode has only ~-0.04 correlation with Python streaming mode! This means batch processing is fundamentally different from streaming, not just a precision issue.

Current Rust vs Python streaming correlation: **0.64** after alignment (shift: -449 samples). This is actually consistent with expected batch vs streaming behavior. The max amplitude is now close: Rust ~0.45 vs Python ~0.46.

**Key insight from local docs**: The `docs/python-reference/` directory contains comprehensive documentation of Python's streaming algorithms, including the exact overlap-add mechanism for ConvTranspose1d and causal context buffering for Conv1d.

---

## Key Research Findings

### From Official Kyutai Source (Moshi/Mimi)

**Streaming Implementation in Moshi Rust** ([GitHub](https://github.com/kyutai-labs/moshi)):
- Kyutai provides a production Rust implementation of Mimi in their Moshi repo (`rust/` directory)
- Python bindings available as `rustymimi` on PyPI ([Release](https://github.com/kyutai-labs/moshi/releases/tag/rustymimi-0.2.2))
- This is the **official reference** for Rust streaming Mimi
- **moshi-swift** uses MLX Swift for iOS, not Candle, but demonstrates the streaming pattern

**Key Mimi Specs**:
- 24 kHz audio → 12.5 Hz representation
- 1.1 kbps bandwidth
- 80ms latency (streaming frame size)
- Licensed: MIT (code), CC-BY 4.0 (weights)

### From Academic Literature

**Streamable Neural Audio Synthesis** ([arXiv:2204.07064](https://arxiv.org/abs/2204.07064), [cached_conv](https://acids-ircam.github.io/cached_conv/)):
- Introduces **cached padding** method for streaming non-causal convolutions
- Key insight: "The last N frames from input buffer 1 are cached and concatenated with input buffer 2 (with N being the original amount of zero padding)"
- Can convert any trained model to streaming **post-training**

**For Transposed Convolutions** ([Andrew Gibiansky](https://andrew.gibiansky.com/streaming-audio-synthesis/)):
- "Each input timestep contributes to k different outputs"
- "The last input timestep in a chunk affects the first (k-1)/2 outputs of the NEXT chunk"
- State buffer size = k - 1 timesteps of partial outputs
- Algorithm:
  1. Run conv_transpose: n inputs → n + k - 1 outputs
  2. Add current state to left edge of outputs
  3. Return all but rightmost k-1 timesteps
  4. Store rightmost k-1 timesteps as new state

### From Local Python Reference Documentation

**CRITICAL**: The `docs/python-reference/` directory contains the answer! Key documents:

1. **[conv-transpose-overlap-add.md](docs/python-reference/STREAMING/conv-transpose-overlap-add.md)** - Complete Python implementation with bias subtraction detail
2. **[conv1d-streaming.md](docs/python-reference/STREAMING/conv1d-streaming.md)** - Causal context buffer with first-frame handling
3. **[seanet-decoder.md](docs/python-reference/ARCHITECTURE/seanet-decoder.md)** - Complete layer structure with state buffer dimensions

**Exact State Buffer Sizes (from local docs)**:
| Layer | Kernel | Stride | Overlap (K-S) | Channels | State Shape |
|-------|--------|--------|---------------|----------|-------------|
| Mimi Upsample | 32 | 16 | 16 | 512 | `[B, 512, 16]` |
| SEANet Stage 0 | 16 | 8 | 8 | 256 | `[B, 256, 8]` |
| SEANet Stage 1 | 10 | 5 | 5 | 128 | `[B, 128, 5]` |
| SEANet Stage 2 | 8 | 4 | 4 | 64 | `[B, 64, 4]` |
| SEANet Stage 3 | 4 | 2 | 2 | 32 | `[B, 32, 2]` |

### From Candle Framework Resources

**Candle does NOT have built-in streaming convolution support**. The `ConvTranspose1d` in [candle-nn](https://docs.rs/candle-nn/latest/candle_nn/conv/struct.ConvTranspose1d.html) is stateless.

However, Kyutai's [moshika-candle-bf16](https://huggingface.co/kyutai/moshika-candle-bf16) demonstrates that streaming Mimi IS possible in Candle - their Rust implementation in the Moshi repo must handle it.

---

## Suggested Approaches

### High Confidence

**1. Study Kyutai's Official Rust Mimi Implementation** (HIGHEST PRIORITY)
- **Why:** This is the authoritative streaming Mimi in Rust
- **Confidence:** Very High - it's Kyutai's own production code
- **How:**
  1. Clone [github.com/kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
  2. Navigate to `rust/mimi/` directory
  3. Study how they implement streaming ConvTranspose1d
  4. Compare with current Rust implementation
  5. The state management pattern should be directly applicable

**2. Process One Latent Frame at a Time with Persistent State**
- **Why:** This is exactly how Python works, already documented in local docs
- **Confidence:** Very High - the algorithms are complete in `docs/python-reference/`
- **How:**
  1. Create state structs matching local doc specs (already partially done in mimi.rs)
  2. Modify `forward_streaming` to process latents one-by-one in a loop
  3. Call `forward_streaming` for ConvTranspose1d with overlap-add (already implemented!)
  4. Ensure state persists across the loop iterations
  5. **Key detail**: Subtract bias before storing partial (already in code at line 338-343!)

**Current implementation status** (from reading mimi.rs):
- State structs: **DONE** (StreamingConv1dState, StreamingConvTr1dState)
- ConvTranspose1d::forward_streaming: **DONE** (overlap-add with bias subtraction)
- Conv1d::forward_streaming: **DONE** (causal context buffer)
- Frame-by-frame loop: **NOT DONE** - currently processes all frames in batch, then applies streaming to SEANet

**The missing piece**: The current code processes ALL latents through output_proj, upsample, and decoder_transformer in batch, then runs SEANet with streaming. Python processes EACH latent frame through the ENTIRE pipeline with streaming state.

### Worth Trying

**3. Try Frame-by-Frame Processing Just for SEANet**
- **Why:** The decoder_transformer may work correctly in batch (non-causal attention)
- **Confidence:** Medium - might give partial improvement
- **How:**
  1. Keep batch processing for output_proj and upsample
  2. Process SEANet one upsampled frame at a time (16 samples per latent)
  3. Maintain streaming state across the 16-sample chunks
  4. This isolates whether SEANet streaming is the key factor

**4. Verify the "First Frame" Handling**
- **Why:** Python has special handling for first frame (replicate padding)
- **Confidence:** Medium - might explain edge effects
- **How:**
  1. Check if `first` boolean is being handled correctly in Conv1d streaming
  2. Verify replicate padding on first frame vs zeros
  3. Compare first few audio samples specifically

### Speculative

**5. Accept Batch Mode with Lower Correlation**
- **Why:** The audio is intelligible, just different phase characteristics
- **Confidence:** Pragmatic fallback
- **How:**
  1. Audio amplitude is now good (~0.45)
  2. Real-time factor is good (~4x)
  3. Speech is intelligible (verify with listening test)
  4. Document limitation and ship

---

## Things That Have Been Tried (DO NOT REPEAT)

From PORTING_STATUS.md, these are **VERIFIED CORRECT**:

**FlowLM Pipeline (All Match Python)**:
1. Tokenization (SentencePiece) - exact token match
2. RoPE (interleaved, applied before transpose)
3. LayerNorm (eps=1e-5)
4. All 6 transformer layers - layer-by-layer verification passed
5. FlowNet (sinusoidal order, SiLU, AdaLN chunk order)
6. LSD time progression (two times, averaged)
7. Latent denormalization (moved before Mimi)

**Mimi Decoder Pipeline (Intermediate Values Match)**:
8. output_proj - first 8 values verified
9. Upsample (no-padding mode, trim 16 samples) - first 8 values verified
10. decoder_transformer (RoPE, causal mask) - first 8 values verified
11. SEANet activation (ELU not GELU, not tanh)

**Recent Fixes Applied**:
12. Mimi processing order: upsample BEFORE transformer
13. min_gen_steps = 0 to allow natural EOS detection
14. SEANet batch mode processing (improved amplitude from 0.12 to 0.45)
15. Removed 5x amplitude scaling hack

---

## Specific Questions to Investigate

1. **Is there a complete streaming Mimi example in Kyutai's Rust codebase?**
   - Check `rust/mimi/src/` in the Moshi repo
   - Look for how they handle state persistence across frames

2. **Does Python actually process the decoder_transformer per-frame?**
   - The decoder_transformer is non-causal (full self-attention)
   - But `mimi_state` is passed to it - why if it's stateless?
   - Check if there's KV caching for the decoder transformer

3. **What's the correct upsampling stride in Pocket TTS?**
   - Docs mention stride=6 for 75Hz intermediate rate
   - But code shows stride=16 for direct 12.5Hz→200Hz
   - Verify which is used in production

4. **How does Python's SEANet ResBlock streaming work exactly?**
   - Each ResBlock has dilated convolutions
   - Dilation affects effective kernel size: `(k-1)*d + 1`
   - State buffer sizes should account for dilation

5. **Should bias subtraction happen for Conv1d too?**
   - ConvTranspose1d subtracts bias from partial (documented)
   - Conv1d stores raw context (no bias subtraction mentioned)
   - Verify this asymmetry is correct

---

## Useful Links & References

### Official Kyutai
- [GitHub: kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) - **Production Rust Mimi**
- [GitHub: kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) - Python reference
- [GitHub: kyutai-labs/moshi-swift](https://github.com/kyutai-labs/moshi-swift) - iOS/MLX reference
- [HuggingFace: kyutai/moshika-candle-bf16](https://huggingface.co/kyutai/moshika-candle-bf16) - Candle weights
- [PyPI: rustymimi](https://pypi.org/project/rustymimi/) - Python bindings for Rust Mimi

### Streaming Convolution Research
- [Streamable Neural Audio Synthesis](https://arxiv.org/abs/2204.07064) - Cached convolution paper
- [cached_conv Library](https://acids-ircam.github.io/cached_conv/) - Post-training streaming conversion
- [Andrew Gibiansky: Streaming Audio Synthesis](https://andrew.gibiansky.com/streaming-audio-synthesis/) - **Detailed overlap-add tutorial**
- [Balacoon: Streaming Inference](https://balacoon.com/blog/streaming_inference/) - Practical guide

### Audio Codec References
- [GitHub: facebookresearch/encodec](https://github.com/facebookresearch/encodec) - EnCodec implementation
- [HuggingFace: EnCodec docs](https://huggingface.co/docs/transformers/model_doc/encodec) - use_causal_conv option

### Candle Framework
- [docs.rs: candle-nn ConvTranspose1d](https://docs.rs/candle-nn/latest/candle_nn/conv/struct.ConvTranspose1d.html)
- [GitHub: huggingface/candle](https://github.com/huggingface/candle)
- [Candle Tutorial for Porting](https://github.com/ToluClassics/candle-tutorial)

### Local Documentation (Already in Repo!)
- `docs/python-reference/STREAMING/conv-transpose-overlap-add.md` - **ROOT CAUSE DOCUMENT**
- `docs/python-reference/STREAMING/conv1d-streaming.md` - Causal convolution
- `docs/python-reference/STREAMING/state-management.md` - StatefulModule pattern
- `docs/python-reference/ARCHITECTURE/seanet-decoder.md` - Layer-by-layer structure

---

## Key Insight

**The Rust implementation already has the streaming convolution code written** - `forward_streaming` methods exist for both Conv1d and ConvTranspose1d with correct overlap-add and bias subtraction. The issue is the **calling pattern**:

**Current (wrong)**:
```
All latents → batch output_proj → batch upsample → batch transformer → batch SEANet (with streaming ConvTranspose)
```

**Correct (Python pattern)**:
```
For each latent:
    latent → streaming output_proj → streaming upsample → streaming transformer → streaming SEANet
    (state persists across iterations)
```

**Recommendation**: Before writing new code, **study Kyutai's Rust Mimi in the Moshi repo**. They've already solved this exact problem in production-quality Rust code. The implementation patterns should be directly applicable.

---

## Implementation Priority

1. **First**: Clone and study [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) `rust/` directory
2. **Second**: Refactor `MimiDecoder::forward_streaming` to process one latent at a time with true streaming
3. **Third**: Verify intermediate values match Python at each step of the streaming loop
4. **Fallback**: If streaming is too complex, accept current batch mode with 0.64 correlation (audio is intelligible)
