# Research Advisor Briefing

**Date:** 2026-01-24
**Current Blocker:** Mimi decoder produces 5-6x lower amplitude than Python due to batch vs streaming convolution processing
**Research Focus:** Streaming ConvTranspose1d, overlap-add accumulation, Mimi/Moshi SEANet implementation

---

## Situational Summary

The Rust/Candle port has achieved a major milestone: **all 42 generated latents now match Python exactly with cosine similarity = 1.0**. The FlowLM transformer, FlowNet, and latent generation pipeline are verified correct. The generation length issue was also fixed by removing the `min_gen_steps = 40` debug value.

The remaining problem is isolated to the **Mimi decoder's SEANet component**. The current implementation processes all latent frames in batch mode, producing audio with ~0.12 max amplitude vs Python's ~0.50-0.60. This 5-6x difference is directly caused by the lack of **inter-frame state accumulation** that Python's streaming convolutions provide.

The PORTING_STATUS.md documents this clearly: Python's `StreamingConv1d` keeps a `previous` buffer for causal context, while `StreamingConvTranspose1d` does overlap-add with a `partial` buffer. These state buffers accumulate signal contributions across frames, building up higher intermediate values that then get scaled down to proper audio amplitude by the tiny output conv weights (~0.003).

---

## Key Research Findings

### From Official Kyutai Source (Moshi/Mimi)

**Streaming Module Implementation** (from `moshi/moshi/modules/conv.py`):

The Mimi codec uses two key streaming classes with explicit state management:

**1. StreamingConv1d:**
```python
def _init_streaming_state(self, batch_size):
    # previous buffer stores (kernel - stride) samples from prior chunks
    previous = torch.zeros(batch_size, in_channels, kernel - stride)
    return _StreamingConv1dState(previous, first=True)

def forward(self, x):
    # Concatenate previous samples with new input
    x = torch.cat([state.previous, x], dim=-1)
    y = self.conv(x)
    # Store trailing samples for next frame
    state.previous[:] = x[..., -TP:]
    return y
```

**2. StreamingConvTranspose1d:**
```python
def _init_streaming_state(self, batch_size):
    # partial buffer stores (kernel - stride) overlapping output samples
    partial = torch.zeros(batch_size, out_channels, K - S)
    return _StreamingConvTr1dState(partial)

def forward(self, x):
    y = self.convtr(x)
    # Add partial buffer to left edge of output
    y[..., :PT] += state.partial
    # Store right edge (minus bias) for next frame
    for_partial = y[..., -PT:]
    if bias is not None:
        for_partial -= bias[:, None]
    state.partial[:] = for_partial
    # Trim overlapping region from output
    return y[..., :-PT]
```

**Key Insight**: The `partial` buffer implements the overlap-add algorithm. Each ConvTranspose1d output has overlapping contributions from adjacent input frames. By accumulating these in the `partial` buffer, the streaming implementation produces correct (higher) amplitude output.

### From Technical Resources

**Andrew Gibiansky's Streaming Audio Synthesis** ([link](https://andrew.gibiansky.com/streaming-audio-synthesis/)):

For transposed convolutions with kernel size k:
1. Each input timestep contributes to k different outputs
2. The last input timestep in a chunk affects the first (k-1)/2 outputs of the NEXT chunk
3. State buffer size = k - 1 timesteps of partial outputs
4. Algorithm:
   - Run conv_transpose: n inputs → n + k - 1 outputs
   - Add current state to left edge of outputs
   - Return all but rightmost k-1 timesteps
   - Store rightmost k-1 timesteps as new state

**Concrete Example** (kernel_size=7, stride=1):
- Chunk 1: 4 inputs → 10 outputs; return 4, store 6 as state
- Chunk 2: 4 inputs → 10 outputs; add state to left, return 4, store 6
- Final: return first 3 of remaining state
- Total: 4 + 4 + 4 + 3 = 15 outputs (matching 12 inputs with proper overlap)

### From SEANet/SoundStream Literature

**SoundStream Paper** ([arXiv:2107.03312](https://arxiv.org/abs/2107.03312)):
- All components use **causal operations** for streaming
- Decoder mirrors encoder with transposed convolutions for upsampling
- Same strides as encoder but in reverse order
- ELU activations throughout

**EnCodec** adds two small LSTM layers after SEANet to improve sequence modeling, which may contribute to state accumulation that Mimi achieves through its Transformer layers.

---

## Suggested Approaches

### High Confidence

**1. Implement Streaming Mimi Decoder**
- **Why:** This directly replicates Python's behavior and is the most correct solution
- **Confidence:** Very High - this is exactly how Python works
- **How:**
  1. Create `StreamingConv1dState` struct with `previous: Tensor` field
  2. Create `StreamingConvTranspose1dState` struct with `partial: Tensor` field
  3. Modify `Conv1d::forward()` to accept optional state, concatenate previous, store trailing
  4. Modify `ConvTranspose1d::forward()` to accept optional state, add partial to left edge, store right edge
  5. Process latents one frame at a time in a loop
  6. After loop, flush remaining partial buffers

**2. Compute Overlap-Add in Batch Mode (Mathematical Approach)**
- **Why:** Avoids frame-by-frame processing overhead while achieving correct output
- **Confidence:** High - mathematically equivalent, but more complex to implement correctly
- **How:**
  1. For each ConvTranspose1d with kernel K, stride S:
  2. After conv_transpose, identify overlapping regions: positions where multiple input frames contribute
  3. For output position `i`, sum contributions from all input frames that affect it
  4. This is essentially a sparse matrix multiply that can be vectorized
  5. The overlap pattern is deterministic based on K and S

### Worth Trying

**3. Apply Amplitude Scaling Heuristic**
- **Why:** Quick fix if streaming is complex to implement
- **Confidence:** Medium - may work for speech but could distort dynamics
- **How:**
  1. Compute the ratio between expected and actual RMS amplitudes
  2. Scale final audio by this factor (approximately 5-6x)
  3. Risk: Distorts quiet/loud dynamics, may not generalize to all voices

**4. Check for Missing Post-Processing**
- **Why:** Python might have normalization or gain stages not in Rust
- **Confidence:** Low - PORTING_STATUS.md says Python has no tanh at output
- **How:**
  1. Compare final audio statistics (RMS, peak, dynamic range)
  2. Check if Python applies any audio normalization before output
  3. Verify sample rate handling (24kHz in both?)

### Speculative

**5. Try Moshi's Rust Implementation (rustymimi)**
- **Why:** Kyutai provides `rustymimi` Python bindings for their Rust Mimi implementation
- **Confidence:** Unknown - may have same architecture as this port
- **How:**
  1. Check if rustymimi source is available
  2. Compare their ConvTranspose1d implementation
  3. Look for streaming state management patterns

---

## Things That Have Been Tried (DO NOT REPEAT)

From PORTING_STATUS.md, these are VERIFIED CORRECT:
1. Tokenization (SentencePiece) - matches Python
2. RoPE (interleaved, applied before transpose) - matches Python
3. LayerNorm (eps=1e-5) - matches Python
4. FlowNet (sinusoidal order, SiLU, AdaLN chunk order) - matches Python
5. LSD time progression (two times, averaged) - matches Python
6. SEANet activation (ELU not GELU) - changed
7. Voice conditioning (concatenation, two-phase) - matches Python
8. FinalLayer norm_final (LayerNorm before modulation) - added
9. SEANet output (no tanh) - removed
10. FlowNet RMSNorm (proper variance calculation) - fixed
11. FlowNet time embedding (only add to conditioning) - fixed
12. Latent denormalization (moved before Mimi) - fixed
13. Generation length (min_gen_steps = 0) - fixed

**Recent fixes in mimi.rs:**
- ResidualBlock ELU placement (ELU before each conv, not after)
- SEANetDecoder order (Conv → ELU → ConvTranspose → ResBlock → ELU)

---

## Specific Questions to Investigate

1. **What are the kernel sizes and strides for each ConvTranspose1d in SEANet?**
   - These determine the overlap-add buffer sizes (K - S for each layer)
   - PORTING_STATUS.md mentions 16x temporal upsampling

2. **Does Moshi's rustymimi have streaming state management?**
   - If so, what data structures do they use?
   - Is the implementation simpler than Python's?

3. **What is the exact upsampling factor at each SEANet stage?**
   - The decoder should mirror encoder strides in reverse
   - Mimi uses strides (4,5,6,8,2) in encoder → (2,8,6,5,4) in decoder

4. **Are there any Transformer layers in Mimi's decoder?**
   - PORTING_STATUS.md mentions "2 transformer layers" in Mimi
   - How do these interact with streaming convolutions?

5. **What happens to partial buffers at sequence end?**
   - Is there a flush/finalize step that returns remaining samples?
   - Python's streaming context manager handles this automatically

---

## Useful Links & References

### Official Kyutai
- [GitHub: kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
- [GitHub: kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) - Contains Mimi implementation
- [HuggingFace: kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)
- [HuggingFace: kyutai/mimi](https://huggingface.co/kyutai/mimi)
- [Blog: Pocket TTS Release](https://kyutai.org/blog/2026-01-13-pocket-tts)
- [DeepWiki: Moshi Technical Overview](https://deepwiki.com/kyutai-labs/moshi)

### Streaming Convolution Resources
- [Andrew Gibiansky: Streaming Audio Synthesis](https://andrew.gibiansky.com/streaming-audio-synthesis/) - **Excellent detailed implementation guide**
- [ACIDS-IRCAM: cached_conv](https://acids-ircam.github.io/cached_conv/) - Streamable neural audio synthesis
- [ArXiv: Streamable Neural Audio Synthesis](https://arxiv.org/abs/2204.07064) - Paper on cached convolutions
- [Balacoon: Streaming Inference with Convolutional Layers](https://balacoon.com/blog/streaming_inference/) - Practical guide

### Audio Codec References
- [ArXiv: SoundStream](https://arxiv.org/abs/2107.03312) - Original SEANet-based codec
- [GitHub: facebookresearch/encodec](https://github.com/facebookresearch/encodec) - EnCodec implementation
- [AudioCraft Modules: conv.py](https://huggingface.co/spaces/facebook/MusicGen/blob/main/audiocraft/modules/conv.py) - StreamableConv classes

### Candle Framework
- [docs.rs: candle_nn::conv](https://docs.rs/candle-nn/latest/candle_nn/conv/index.html) - ConvTranspose1d API
- [GitHub: huggingface/candle](https://github.com/huggingface/candle)

---

## Key Insight

The breakthrough discovery is that **all latents now match Python perfectly** - the transformer and flow matching components are verified correct. The remaining amplitude issue is entirely due to **missing overlap-add state accumulation in the SEANet decoder**.

The fix requires implementing streaming convolution state management:
- `previous` buffer for Conv1d (kernel - stride samples of input history)
- `partial` buffer for ConvTranspose1d (kernel - stride samples of overlapping outputs)

**The implementation agent should focus on implementing streaming state for the Mimi decoder, not on further debugging the latent generation pipeline which is now verified correct.**

### Recommended Implementation Order

1. Identify all ConvTranspose1d layers in SEANet decoder and their kernel/stride values
2. Create state structs to hold partial buffers for each layer
3. Modify forward pass to process one latent frame at a time
4. Implement overlap-add: add partial to left edge, store right edge as new partial
5. Add flush step to return remaining partial samples at end
6. Verify amplitude matches Python (~0.50-0.60 vs current ~0.12)
