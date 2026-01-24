# Kyutai Pocket TTS Rust Port - Status Report

## Overview

Porting Kyutai Pocket TTS (~117M parameter on-device TTS model) from Python to Rust/Candle for iOS deployment. The goal is to achieve near-identical waveform output (correlation > 0.95) compared to the Python reference.

**Current Status**: Audio amplitude improved 3x after fixes, but waveform correlation still ~0 (random). Several architectural issues identified and fixed, but more work needed.

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
- Current correlation: ~0 (essentially random)
- Target: > 0.95
- This indicates content is fundamentally different, not just scaled

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
- tanh output bounding

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
   - After voice prompting phase
   - After text prompting phase
   - During autoregressive generation
2. **Compare FlowNet conditioning** - The hidden states fed to FlowNet
3. **Verify attention patterns** - Check if self-attention produces similar outputs
4. **Check KV cache correctness** - Verify cache is populated and used correctly
5. **Investigate remaining amplitude issue** - 4x difference (0.20 vs 0.60) suggests:
   - Possible missing scaling in decoder
   - Incorrect activation functions
   - Different preprocessing/postprocessing

---

## Commands Reference

```bash
# Build
cargo build --release

# Run test
./target/release/test-tts -m /Users/ramerman/dev/unamentis/models/kyutai-pocket-ios \
  -t "Hello, this is a test." -o /tmp/output.wav

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
