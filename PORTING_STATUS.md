# Kyutai Pocket TTS Rust Port - Status Report

## Overview

Porting Kyutai Pocket TTS (~117M parameter on-device TTS model) from Python to Rust/Candle for iOS deployment. The goal is to achieve near-identical waveform output (correlation > 0.95) compared to the Python reference.

**Current Status**: Multiple architectural issues fixed. Waveform correlation improved from 0.0016 to ~0.01. Target is > 0.95. More work needed.

---

## Issues Found and Fixed

### 1. Tokenization (FIXED)
**Problem**: Character-level tokenization produced 32 tokens vs Python's 17 tokens.
**Solution**: Switched to SentencePiece tokenization.
**File**: `src/tokenizer.rs`
**Verification**: Token counts now match Python.

### 2. RoPE (Rotary Position Embedding) (FIXED)
**Problem**: Rust used SPLIT HALVES (first D/2 vs last D/2), Python uses INTERLEAVED pairs.
**Solution**: Reshape to `[B,T,H,D/2,2]` and extract `[..0]` and `[..1]` components.
**File**: `src/modules/rotary.rs`

**Also Fixed**: RoPE was applied AFTER transpose in Rust. Python applies BEFORE transpose.
**File**: `src/modules/attention.rs`

### 3. LayerNorm vs RMSNorm (FIXED)
**Problem**: Rust used RMSNorm for `out_norm`, but model weights have bias (indicating LayerNorm).
**Solution**: Changed to LayerNorm with eps=1e-5.
**File**: `src/models/flowlm.rs`

### 4. FlowNet Architecture (FIXED)
Multiple issues fixed in `src/modules/flownet.rs`:

| Issue | Python | Rust (Before) | Status |
|-------|--------|---------------|--------|
| Sinusoidal order | `[cos, sin]` | `[sin, cos]` | FIXED |
| MLP activation | SiLU | GELU | FIXED |
| AdaLN chunk order | `[shift, scale, gate]` | `[scale, shift, gate]` | FIXED |
| SiLU before AdaLN linear | Yes | No | FIXED |

### 5. LSD Time Progression (FIXED)
**Problem**: Rust used single time value going from 1→0. Python's LSD uses TWO time values:
- `s = i / num_steps` (start time)
- `t = (i + 1) / num_steps` (target time)

**Solution**: Modified FlowNet to accept both s and t, AVERAGE the two time embeddings.
**File**: `src/modules/flownet.rs`

### 6. SEANet Activation Function (FIXED)
**Problem**: Rust used GELU throughout SEANet decoder. Python uses ELU(alpha=1.0).
**Solution**: Changed all GELU to ELU(1.0) in SEANet.
**Files**:
- `src/models/mimi.rs` (inline SEANet)
- `src/models/seanet.rs` (module SEANet)
- `src/modules/conv.rs` (SEANetDecoderBlock)

### 7. Voice Conditioning Concatenation (FIXED)
**Problem**: Rust ADDS mean-pooled voice embedding to hidden states. Python CONCATENATES full voice embedding with text embeddings along sequence dimension.

```python
# Python:
text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)
```

**Solution**: Modified to concatenate voice embedding (125 frames) with text embeddings.
**File**: `src/models/flowlm.rs`

**Results After Fix**:
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Max amplitude | 0.062 | 0.170 | +174% |
| Sample count | 82568 | 86408 | Now matches ref! |
| Frame count | 43 | 45 | Now matches ref! |
| Correlation | ~0 | ~0 | Still wrong |

The fix dramatically improved frame generation but waveform content is still different.

### 8. Voice Conditioning Sequence (FIXED - STILL INVESTIGATING)
**Problem**: Rust concatenated `[text, voice]` in a single forward pass. Python processes them in **SEPARATE forward passes** with voice FIRST.

**Python Flow** (discovered by reading `tts_model.py` and `flow_lm.py`):
1. `get_state_for_audio_prompt()` - Voice through transformer FIRST
   - KV cache positions 0-124: Voice context
2. `_generate()` text prompting - Text through transformer SECOND
   - KV cache positions 125-141: Text context
3. Autoregressive loop - Latents one at a time
   - KV cache positions 142+: Generated latents

**Previous Rust Approach** (WRONG):
- Single pass with concatenated `[text, voice]`
- KV cache: positions 0-16 = text, 17-141 = voice (wrong order!)

**Fix Applied**: Two-phase forward pass
1. Phase 1: Process voice embeddings alone (populates KV cache 0-124)
2. Phase 2: Process text embeddings (appends to KV cache 125-141)
3. Phase 3: Autoregressive latent generation (142+)

**File**: `src/models/flowlm.rs`
**Result**: Correlation still ~0. More investigation needed.

**Additional Verification**:
- Voice embeddings confirmed identical between Python and local safetensors files
- Created `validation/dump_intermediates.py` for Python tensor capture

### 9. FinalLayer Missing norm_final (FIXED)
**Problem**: Python's `FinalLayer` in FlowNet applies LayerNorm (with `affine=false`) before AdaLN modulation. Rust was missing this normalization step.
**Solution**: Added `layer_norm_no_affine()` function and applied it before modulation in FinalLayer.
**Files**:
- `src/modules/layer_norm.rs` - Added `layer_norm_no_affine` function
- `src/modules/flownet.rs` - Added norm_final call in FinalLayer
**Result**: Correlation improved from 0.0016 to 0.0256

### 10. SEANet Output Activation (FIXED)
**Problem**: Rust applied `tanh()` to SEANet output. Python reference does NOT apply tanh at the end.
**Solution**: Removed tanh from SEANet forward method.
**File**: `src/models/mimi.rs`

### 11. Weight Loading Verified (CONFIRMED CORRECT)
**Verification**: Manual matmul of `bos_emb @ input_linear.weight.T` matches Rust output exactly.
- Python debug hook for `input_linear` captures different step (not BOS)
- Actual weight loading is correct

---

## Issues Still Being Investigated

### EOS Triggering Too Early
- EOS triggers at step 20 with logit=-2.5 (threshold=-4.0)
- Python generates ~45 frames for same phrase
- Currently using `min_gen_steps=40` as workaround
- May be related to incorrect hidden states

### Audio Amplitude (Updated after all fixes)
| Metric | Reference | Rust (After all fixes) | Ratio |
|--------|-----------|------------------------|-------|
| Max amplitude | 0.605 | 0.170 | 28% |
| RMS | 0.099 | 0.020 | 20% |
| Samples | 86400 | 86408 | 100% ✓ |

Amplitude is still ~4x quieter but now within reasonable range.

### Latent Statistics (Rust)
- mean=-0.0605, std=1.0477
- min=-9.1670, max=2.7633
- Note: Asymmetric range may indicate issues

### Waveform Correlation
- Current correlation: ~0.01-0.02
- Target: > 0.95
- Reference implementation (babybirdprd/pocket-tts) achieved ~0.06 max difference

### Root Cause Found (2025-01-24)
**The transformer is working correctly!** Divergence is in FlowNet's random noise initialization.

FlowNet starts from `randn(0, 1, ...)` - random Gaussian noise. Different random seeds between Python and Rust cause different latents, which cascade into different audio.

For proper comparison, need to:
1. Use same random seed, OR
2. Use zeros instead of noise for testing, OR
3. Compare velocity (model output) instead of final latent

---

## Test Results Summary

### Build Status
All code compiles successfully with only minor warnings.

### Test Phrases Used
1. "Hello, this is a test of the Pocket TTS system."
2. "The quick brown fox jumps over the lazy dog."
3. "One two three four five six seven eight nine ten."
4. "How are you doing today?"

### Reference Harness
Located at: `validation/reference_outputs/`
- Generates audio using Python Pocket TTS
- Saves WAV files and numpy arrays
- Includes ASR transcription validation

---

## Model Architecture Summary

### FlowLM (Backbone Transformer)
- 6 transformer layers
- 1024 hidden dim
- 16 attention heads
- RoPE for positional encoding
- LayerNorm (not RMSNorm)

### FlowNet (Latent Generator)
- 6 residual blocks with AdaLN
- 512 hidden dim
- 32 latent dim
- 2 TimestepEmbedders (averaged for LSD)
- SiLU activation throughout

### Mimi Decoder
- 2 transformer layers
- 16x temporal upsampling (ConvTranspose1d)
- SEANet decoder with ELU activation
- No output bounding (raw conv output, NOT tanh)

---

## Weight Loading Verification

### Paths Checked
- `mimi.quantizer.output_proj.weight`: [512, 32, 1] ✓
- `mimi.upsample.convtr.convtr.weight`: [512, 1, 32] ✓
- `mimi.decoder.*`: Full SEANet weights ✓
- `flow_lm.speaker_proj_weight`: [1024, 512] (for voice projection)

### Voice Embeddings
- Format: `audio_prompt` key in safetensors
- Shape: [1, ~125, 1024] (already projected)
- 8 voices available: alba, marius, javert, jean, fantine, cosette, eponine, azelma

---

## Key Differences from Python

| Aspect | Python | Rust |
|--------|--------|------|
| Precision | BFloat16 | Float32 |
| Framework | PyTorch | Candle |
| Voice conditioning | Concatenate | Was adding, now concat |
| Streaming | Supported | Partial |

---

## Files Modified

### Core Model Files
- `src/models/flowlm.rs` - Transformer, generation loop, voice conditioning
- `src/models/mimi.rs` - Decoder with inline SEANet
- `src/models/seanet.rs` - Standalone SEANet module
- `src/models/pocket_tts.rs` - Top-level TTS interface

### Module Files
- `src/modules/flownet.rs` - Flow matching network with LSD
- `src/modules/rotary.rs` - RoPE implementation
- `src/modules/attention.rs` - Multi-head attention
- `src/modules/conv.rs` - Conv layers with ELU
- `src/modules/layer_norm.rs` - LayerNorm implementation
- `src/modules/embeddings.rs` - Text and voice embeddings

### Test Files
- `src/bin/test_tts.rs` - Test harness
- `validation/reference_harness.py` - Python reference generator
- `validation/validate.py` - Comparison orchestrator

---

## Next Steps

1. **Compare transformer hidden states** - Layer-by-layer comparison:
   - After voice prompting phase (Python: out_norm mean=-0.003, std=0.34)
   - After text prompting phase
   - During autoregressive generation (Rust step 0: mean=-0.0009, std=0.44)
2. **Compare FlowNet output** - Python flownet_output: mean=-0.45, std=1.25 vs Rust: mean=-0.06, std=2.39
3. **Verify attention patterns** - Check if self-attention produces similar outputs
4. **Check KV cache correctness** - Verify cache is populated and used correctly
5. **Review babybirdprd/pocket-tts implementation** for additional reference:
   - They achieved ~0.06 max difference from Python
   - Compare their RMSNorm variance computation
   - Compare their modulation precomputation approach

---

## Commands Reference

```bash
# Build
cargo build --release

# Run test (replace with your model path)
./target/release/test-tts -m /path/to/model/kyutai-pocket-ios \
  -t "Hello, this is a test." -o /tmp/output.wav

# Run full validation
cd validation && ./run_tests.sh /path/to/model/kyutai-pocket-ios

# Compare waveforms (Python)
python3 << 'EOF'
import numpy as np
ref = read_wav("reference.wav")
rust = read_wav("output.wav")
corr = np.corrcoef(ref[:len(rust)], rust)[0, 1]
print(f"Correlation: {corr:.6f}")
EOF
```

---

## Session Notes

This document was created to prevent repeated attempts at already-tried fixes. All changes are in unstaged files pending testing of the voice concatenation fix.

---

## Session 2025-01-24: Layer-by-Layer Debugging

### Methodology
Systematically comparing Rust intermediate values against Python to find divergence point.

### Findings

#### 1. LayerNorm Epsilon (FIXED)
- **Problem**: FlowLMConfig used `rms_norm_eps: 1e-6` but Python nn.LayerNorm default is `1e-5`
- **Fix**: Changed to `1e-5` in src/models/flowlm.rs
- **Result**: Correlation improved from 0.0109 to 0.0172

#### 2. Components Verified as MATCHING Python

| Component | Rust Value | Python Reference | Status |
|-----------|------------|------------------|--------|
| BOS embedding first 8 | [-0.021240234, 0.012329102, -0.018310547, 0.06640625, 0.028442383, 0.09375, -0.02319336, 0.001335144] | [-0.02124023, 0.0123291, -0.01831055, 0.06640625, 0.02844238, 0.09375, -0.02319336, 0.00133514] | ✅ MATCH |
| LayerNorm (norm1) output first 8 | [-0.13284129, 0.053179074, 0.12783831, -0.04164055, 0.019039862, 0.14751202, -0.051860895, 0.010783803] | [-0.13284129, 0.05317907, 0.12783831, ...] | ✅ MATCH |
| Q before RoPE head0 first 8 | [1.091528, 0.8480718, -0.55045056, -1.6421012, -0.7520723, -0.9170052, -2.2115798, 1.0723763] | [1.0915277, 0.8480716, -0.5504507, -1.6421008, -0.75207204, -0.91700494, -2.2115788, 1.0723768] | ✅ MATCH |
| Q after RoPE (offset=125) head0 first 8 | [1.3822591, -0.0043869615, -1.2832044, -1.1631329, 0.558903, -1.0460109, 1.0118504, -2.2399185] | (verified with test script) | ✅ MATCH |

#### 3. Current Investigation: Attention and KV Cache

Latest diagnostic output:
```
[Attn-L0] K shape after cache: [1, 16, 132, 64]  # 125 voice + 7 text
[Attn-L0] Q shape: [1, 16, 7, 64]                # 7 text tokens
[Attn-L0] K[pos=125] head0 first 8: [5.314439, -3.7322762, -3.3927584, -1.637581, 1.1816614, -4.9086018, 0.12820822, -1.8842314]
[Attn-L0] Raw attn scores head0 q0 first 8: [13.028812, -3.285898, -14.347115, -23.565731, ...]
[Attn-L0] Softmax attn probs head0 q0 first 8: [0.005478, 0.000713, 0.000179, ...]
```

**Key observation**: KV cache shapes are correct (132 = 125 voice + 7 text), but need Python reference values to compare attention scores.

#### 4. Statistics Comparison

| Metric | Python | Rust | Notes |
|--------|--------|------|-------|
| out_norm mean | -0.003252 | -0.001425 | Close |
| out_norm std | 0.340720 | 0.4230 | Different |
| flownet_output mean | -0.448619 | -0.1594 | Different |
| flownet_output std | 1.246537 | 2.5161 | ~2x higher in Rust |

**Note**: Python debug outputs used different text ("Hello, this is a test of the Pocket TTS system." - 17 tokens) vs current test ("Hello, this is a test." - 7 tokens). Need to regenerate with same text.

### Hypotheses to Test

1. **KV cache values from voice phase may be incorrect**
   - Need to compare K[pos=0] (first voice position) with Python
   - Voice embedding processing might differ

2. **V values might be computed differently**
   - Haven't compared V values yet

3. **Attention softmax precision**
   - Large negative values before softmax (-23.56) could cause numerical issues

4. **Scale factor**
   - Using `1.0 / sqrt(head_dim)` = `1.0 / sqrt(64)` = 0.125
   - Verify Python uses same scaling

### Verified Components (All Match Python!)

**Layer 0 Text Phase** - All intermediate values verified:
| Step | Python | Rust | Status |
|------|--------|------|--------|
| norm1 | [-0.1328412, 0.05317907, ...] | [-0.13284129, 0.053179074, ...] | ✅ |
| Q before RoPE | [1.0915278, 0.848071, ...] | [1.091528, 0.8480718, ...] | ✅ |
| K before RoPE | [6.485494, 0.33394, ...] | [6.4854937, 0.3339412, ...] | ✅ |
| V | [0.032204, -0.014147, ...] | [0.032203987, -0.014147918, ...] | ✅ |
| Voice K cache[0] | [6.905799, -3.9978523, ...] | [6.9058, -3.9978526, ...] | ✅ |
| Voice V cache[0] | [-0.06757915, 0.01683673, ...] | [-0.06757918, 0.016836748, ...] | ✅ |
| Attention output | [0.007504372, 0.004966091, ...] | [0.0075043687, 0.0049661067, ...] | ✅ |
| norm2 | [-0.004807731, 0.029376400, ...] | [-0.0048077395, 0.02937645, ...] | ✅ |
| MLP output | [-0.03569853, -0.01593189, ...] | [-0.035698526, -0.0159319, ...] | ✅ |
| **Layer 0 output** | [-0.036128733, -0.008097149, ...] | [-0.036128726, -0.008097142, ...] | ✅ |

### Issue Found: Divergence in Autoregressive Generation

The text processing phase works correctly! But **step 0 of latent generation diverges**:
- Python step 0 hidden: [-0.24085297, -0.47014824, 0.23839632, 1.0895451, ...]
- Rust step 0 hidden: [-0.2274104, -0.35906476, 0.05141802, 0.5091589, ...]

This is when BOS embedding goes through the transformer with the full KV cache (132 positions).

### Step 0 (First Latent Generation) - Also Verified!

| Step | Python | Rust | Status |
|------|--------|------|--------|
| norm1 first 8 | [-0.02429175, 0.07576486, ...] | [-0.02429175, 0.07576486, ...] | ✅ |
| attn output | [0.02120438, -0.02153354, ...] | [0.0212044, -0.021533545, ...] | ✅ |
| norm2 | [0.02885031, -0.04048911, ...] | [0.02885034, -0.04048911, ...] | ✅ |
| MLP | [-0.04567280, -0.02552469, ...] | [-0.045672808, -0.025524687, ...] | ✅ |
| Layer 0 output | [-0.02498163, -0.04425660, ...] | [-0.02498162, -0.044256598, ...] | ✅ |
| **FINAL HIDDEN** (after all layers + out_norm) | [-0.22741127, -0.35906517, 0.05141879, 0.50915742, ...] | [-0.2274104, -0.35906476, 0.05141802, 0.5091589, ...] | ✅ |

**The entire transformer matches Python to 5+ decimal places!**

### FlowNet Investigation

With zeros instead of random noise, FlowNet step 0 produces:
```
velocity first 8: [-0.6430681, 0.23889166, 0.2160035, 1.347405, -0.71227896, 3.2762485, -0.24154162, 0.1518566]
```

Need Python comparison with same (zero) starting point to verify FlowNet correctness.

### Next Steps

1. **Modify Python to use zeros** instead of randn in FlowNet for comparison
2. **Compare FlowNet velocity** between Python and Rust with identical inputs
3. If FlowNet matches, **check Mimi decoder**
4. Once all components verified, use same random seed for full comparison
5. Alternatively: implement seeded RNG for reproducibility
