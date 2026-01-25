# Gap Analysis: Information for Kyutai Outreach

## Purpose

This document identifies information gaps that cannot be resolved from the available Python source code and may require outreach to Kyutai Labs.

---

## Available Information (Complete)

Everything below is fully documented from the Python source:

| Area | Status | Documentation |
|------|--------|---------------|
| StreamingConv1d | ✅ Complete | `STREAMING/conv1d-streaming.md` |
| StreamingConvTranspose1d | ✅ Complete | `STREAMING/conv-transpose-overlap-add.md` |
| StatefulModule pattern | ✅ Complete | `STREAMING/state-management.md` |
| SEANet architecture | ✅ Complete | `ARCHITECTURE/seanet-decoder.md` |
| Mimi decode flow | ✅ Complete | `ARCHITECTURE/mimi-decode.md` |
| FlowLM architecture | ✅ Complete | `ARCHITECTURE/flowlm.md` |
| FlowNet/LSD decode | ✅ Complete | `ARCHITECTURE/flownet-lsd.md` |
| Transformer + KV cache | ✅ Complete | `MODULES/transformer.md` |
| RoPE implementation | ✅ Complete | `MODULES/rope.md` |
| Layer normalization | ✅ Complete | `MODULES/layer-norm.md` |

---

## Gaps Requiring External Input

### 1. Numerical Precision Tolerance

**Question:** What numerical divergence is acceptable between bfloat16 (Python) and float32 (Rust)?

**Context:**
- Python uses `torch.bfloat16` for model weights and computation
- Rust/Candle uses `f32` (float32)
- Current Rust port has latent cosine similarity = 1.0 (perfect match)
- But audio waveform correlation is only 0.013 (target: >0.95)

**What we know:**
- The latent mismatch is not due to precision (they match exactly)
- The audio mismatch is due to streaming vs batch processing
- But after fixing streaming, what tolerance should we expect?

**Suggested question for Kyutai:**
> "We're porting Pocket TTS to Rust with float32. After implementing streaming correctly, what numerical tolerance should we expect vs the bfloat16 reference? Is >0.95 waveform correlation a reasonable target, or should we aim higher?"

---

### 2. Streaming Edge Cases

**Question:** Are there edge cases at sequence boundaries that aren't obvious from the code?

**Context:**
- First frame: State buffers are initialized to zeros (clear in code)
- Last frame: What happens when we stop generating?
- Partial frames: What if the last frame doesn't complete a full stride?

**What we know:**
- `StreamingConvTranspose1d`: First frame adds zeros, subsequent frames add overlap
- No explicit "flush" or "finalize" operation visible in code
- The streaming code seems to work for arbitrary-length sequences

**Suggested question for Kyutai:**
> "For streaming mode, are there any initialization or finalization steps beyond what's in the code? Specifically, should the last audio frame be handled differently (e.g., flushing remaining overlap buffers)?"

---

### 3. Official Test Vectors

**Question:** Are there official intermediate tensor values for validation?

**Context:**
- We're debugging by comparing Python vs Rust outputs
- Having known-correct values at key points would speed debugging
- Current validation uses Whisper ASR as a sanity check

**What would help:**
- Input/output pairs for each major component
- Intermediate tensor values at key points (after transformer, after SEANet stages, etc.)
- Expected numerical precision at each stage

**Suggested question for Kyutai:**
> "Are there official test vectors (intermediate tensors at key points) that third-party implementations can use for validation? Even a single 'golden' example with saved intermediates would be extremely helpful."

---

### 4. Known Implementation Gotchas

**Question:** Are there undocumented quirks that third-party ports commonly encounter?

**Context:**
- We've fixed 15+ issues during porting (documented in PORTING_STATUS.md)
- Examples: RoPE interleaving, AdaLN chunk order, activation functions
- Some of these took significant debugging to discover

**What would help:**
- List of common porting pitfalls
- Differences between training and inference code
- Any behavior that differs from "standard" implementations

**Suggested question for Kyutai:**
> "Is there any internal documentation about implementation details that could help third-party ports? We've encountered several non-obvious details (RoPE interleaving, AdaLN modulation order, etc.) that weren't obvious from the code."

---

### 5. Decoder Transformer State

**Question:** Does the decoder transformer in Mimi need streaming state (KV cache)?

**Context:**
- Mimi has a 2-layer transformer after the upsample
- The code passes `mimi_state` to it
- But it uses `causal=False` (full self-attention)

**What we know:**
- Non-causal attention doesn't typically need KV cache for streaming
- But the transformer does have `StatefulModule` infrastructure
- Unclear if this is for streaming or just for consistency

**Suggested question for Kyutai:**
> "In the Mimi decoder, the 2-layer transformer uses non-causal attention. Does it need to maintain state across frames for correct streaming, or is it stateless within each decode call?"

---

## Information Extracted from Source

### Configuration Values

From `seanet.py` defaults:
```python
ratios = [8, 5, 4, 2]      # Total: 320x upsampling
n_filters = 32              # Base channel count
dimension = 128             # Latent dimension (before Mimi projection)
n_residual_layers = 3       # ResBlocks per stage
kernel_size = 7             # Initial/final conv
residual_kernel_size = 3    # ResBlock convs
dilation_base = 2           # Dilation growth
compress = 2                # Channel compression in ResBlocks
pad_mode = "reflect"        # Non-streaming padding
```

From `mimi.py`:
```python
sample_rate = 24000
frame_rate = 12.5           # Hz (latent frame rate)
channels = 1                # Mono audio
```

From `flow_lm.py`:
```python
ldim = 32                   # Latent dimension
dim = 1024                  # Transformer hidden dimension
# Transformer is 6 layers, 16 heads, 64 dim per head
```

### Weight Mapping

Main weight file: `model.safetensors`

Key prefixes:
- `flow_lm.` - FlowLM weights
- `mimi.encoder.` - Mimi encoder (not used for TTS)
- `mimi.decoder.` - Mimi decoder
- `mimi.quantizer.` - Quantizer projections
- `mimi.encoder_transformer.` - Encoder transformer
- `mimi.decoder_transformer.` - Decoder transformer

---

## Summary

### What We Have
- **Complete source code** - Every implementation detail is visible
- **Working validation** - Can compare Python vs Rust outputs
- **Clear architecture** - Documentation now covers all components

### What Would Help
1. **Precision guidance** - Expected tolerance for float32 port
2. **Test vectors** - Official intermediate values for validation
3. **Common pitfalls** - List of known porting issues
4. **Edge case documentation** - Streaming boundary behavior

### Recommended Outreach Approach

If reaching out to Kyutai:

1. **Introduce the project** - Rust/Candle port for iOS
2. **Share progress** - 95% complete, latents match exactly
3. **Specific questions** - The 5 questions above
4. **Offer contribution** - Share documentation/learnings back

### Contact

- GitHub: https://github.com/kyutai-labs/pocket-tts/issues
- Email: (check kyutai.org for contact info)
- HuggingFace: Discussion thread on model card
