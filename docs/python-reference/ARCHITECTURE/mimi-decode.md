# Mimi Decoder: Latent to Audio

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/models/mimi.py`

---

## Overview

The Mimi decoder converts latent representations from FlowLM into audio waveforms. It consists of:
1. **Upsample**: 16x temporal upsampling (12.5 Hz → 200 Hz)
2. **Decoder Transformer**: 2-layer transformer for contextual refinement
3. **SEANet Decoder**: Convolutional upsampling to audio (200 Hz → 24 kHz)

---

## Complete Decode Flow

```python
def decode_from_latent(self, latent: torch.Tensor, mimi_state) -> torch.Tensor:
    """Decode latent representation to audio.

    Args:
        latent: [B, T_latent, 32] from FlowLM (or quantizer output)
        mimi_state: Streaming state dictionary

    Returns:
        audio: [B, 1, T_latent * 1920] audio samples
    """
    # Step 1: Upsample from 12.5 Hz to 200 Hz (16x)
    emb = self._to_encoder_framerate(latent, mimi_state)  # [B, T*16, 512]

    # Step 2: Contextual refinement via transformer
    (emb,) = self.decoder_transformer(emb, mimi_state)    # [B, T*16, 512]

    # Step 3: Convert to audio via SEANet
    out = self.decoder(emb, mimi_state)                   # [B, 1, T*1920]

    return out
```

---

## Component Details

### 1. Temporal Upsample (ConvTrUpsample1d)

```python
class ConvTrUpsample1d(nn.Module):
    def __init__(self, stride: int, dimension: int):
        # Depthwise transposed convolution for upsampling
        self.convtr = StreamingConvTranspose1d(
            in_channels=dimension,     # 512
            out_channels=dimension,    # 512
            kernel_size=2 * stride,    # 32
            stride=stride,             # 16
            groups=dimension,          # Depthwise (each channel independent)
            bias=False,
        )

    def forward(self, x, mimi_state):
        return self.convtr(x.transpose(-1, -2), mimi_state).transpose(-1, -2)
```

**Key Details:**
- Input: `[B, T, 512]` (transposed to `[B, 512, T]` for conv)
- Output: `[B, T*16, 512]`
- **Groups = dimension** (depthwise convolution)
- **No bias** (bias=False)
- Overlap state: `partial` of shape `[B, 512, 16]`

### 2. Decoder Transformer (ProjectedTransformer)

```python
class ProjectedTransformer(nn.Module):
    """Transformer with input/output projections."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        causal: bool = False,  # Full self-attention
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            causal=causal,
        )

    def forward(self, x, model_state):
        x = self.input_proj(x)
        x = self.transformer(x, model_state)
        x = self.output_proj(x)
        return (x,)  # Returns tuple for compatibility
```

**Key Details:**
- 2 transformer layers
- **Non-causal** (full self-attention over upsampled sequence)
- Input/output projections maintain dimension
- Uses GELU activation in FFN
- Layer scale for residual connections

### 3. SEANet Decoder

See [SEANet Decoder](seanet-decoder.md) for full details.

Input: `[B, 512, T*16]` (after transpose for convolution)
Output: `[B, 1, T*16*320]` = `[B, 1, T*5120]`

Wait, that's wrong. Let me recalculate:

For the Pocket TTS Mimi decoder:
- Input latent at 12.5 Hz
- Upsample 16x → 200 Hz
- SEANet upsamples by ratios [8, 5, 4, 2] = 320x → 64000 Hz?

Actually, looking at the code more carefully:

```python
encoder_frame_rate = sample_rate / encoder.hop_length
# e.g., 24000 / 320 = 75 Hz

frame_rate = 12.5 Hz  # Mimi output rate

if encoder_frame_rate != frame_rate:
    # Need to downsample from 75 Hz to 12.5 Hz = 6x
    # Or upsample from 12.5 Hz to 75 Hz in decoder
    downsample_stride = encoder_frame_rate / frame_rate  # 75/12.5 = 6
```

Hmm, let me re-read the Mimi configuration. The actual upsampling path is:

1. **Mimi latent**: 12.5 Hz
2. **Upsample**: stride=6 (NOT 16) → 75 Hz (SEANet input rate)
3. **SEANet**: ratios [8, 5, 4, 2] = 320x → 24000 Hz

So the total is 6 * 320 = 1920 samples per latent frame, which matches 24000/12.5 = 1920.

Let me correct the documentation:

---

## Correct Upsampling Math

### Frame Rates

| Stage | Rate | Description |
|-------|------|-------------|
| FlowLM output | 12.5 Hz | 1 latent every 80ms |
| Mimi upsample | 6x | 12.5 × 6 = 75 Hz |
| SEANet input | 75 Hz | = sample_rate / hop_length = 24000 / 320 |
| SEANet output | 24000 Hz | Audio sample rate |

### Samples per Latent Frame

```
1 latent frame @ 12.5 Hz
    ↓ 6x upsample
6 frames @ 75 Hz
    ↓ 320x SEANet (8 × 5 × 4 × 2)
1920 samples @ 24000 Hz
```

**80ms of audio = 1920 samples @ 24kHz**

---

## State Management

### States Created

```python
mimi_state = init_states(mimi, batch_size=1, sequence_length=1000)

# Contains:
{
    # Upsample ConvTranspose (stride=6, kernel=12)
    "upsample.convtr": {"partial": [1, 512, 6]},

    # Decoder transformer (2 layers, each with attention)
    "decoder_transformer.transformer.layers.0.self_attn": {
        "cache": [2, 1, 1000, 8, 64],  # KV cache
        "current_end": [0]  # Position counter
    },
    "decoder_transformer.transformer.layers.1.self_attn": {...},

    # SEANet decoder (many layers)
    "decoder.model.0": {"previous": [...], "first": [...]},
    "decoder.model.2": {"partial": [...]},
    # ... etc
}
```

### State Updates

After each frame:
```python
increment_steps(mimi, mimi_state, increment=6)  # 6 for the upsample
```

This advances the attention position counters.

---

## Per-Frame Processing

For streaming, process one latent frame at a time:

```python
def process_one_frame(mimi, latent_frame, mimi_state):
    """Process exactly one latent frame.

    Args:
        latent_frame: [B, 1, 32] single latent
        mimi_state: Streaming state (maintained across calls)

    Returns:
        audio_chunk: [B, 1, 1920] audio samples
    """
    # Denormalize (FlowLM outputs normalized latents)
    latent = latent_frame * emb_std + emb_mean  # [B, 1, 32]

    # Transpose for convolution: [B, 32, 1]
    latent = latent.transpose(-1, -2)

    # Quantizer (identity in inference, projects 32→512)
    latent = mimi.quantizer(latent)  # [B, 512, 1]

    # Transpose back: [B, 1, 512]
    latent = latent.transpose(-1, -2)

    # Decode
    audio = mimi.decode_from_latent(latent, mimi_state)

    # Increment state positions
    increment_steps(mimi, mimi_state, increment=6)

    return audio  # [B, 1, 1920]
```

---

## Quantizer (DummyQuantizer)

In inference mode, the quantizer is just a linear projection:

```python
class DummyQuantizer(nn.Module):
    def __init__(self, dimension: int = 32, output_dimension: int = 512):
        self.output_proj = nn.Linear(dimension, output_dimension)

    def forward(self, x):
        # x: [B, 32, T]
        x = x.transpose(-1, -2)  # [B, T, 32]
        x = self.output_proj(x)   # [B, T, 512]
        x = x.transpose(-1, -2)  # [B, 512, T]
        return x
```

---

## Critical Implementation Notes

### 1. Frame-by-Frame Processing

The Python code in TTSModel processes frames one at a time:
```python
while True:
    latent = latents_queue.get()  # [B, 1, 32]
    audio_frame = mimi.decode_from_latent(quantized, mimi_state)
    increment_steps(mimi, mimi_state, increment=16)  # (actually 6 for upsample)
```

The Rust port MUST do the same for correct overlap-add behavior.

### 2. State Persistence

States must be maintained across calls. Don't reinitialize per-frame.

### 3. Upsample Stride

The upsample stride depends on configuration:
- `encoder_frame_rate / frame_rate = (sample_rate / hop_length) / frame_rate`
- For 24kHz, hop=320, frame_rate=12.5: stride = (24000/320)/12.5 = 6

### 4. Decoder Transformer is Non-Causal

The decoder transformer uses **full self-attention** (not causal). This means each position can attend to all other positions in the upsampled sequence.

---

## Related Documents

- [SEANet Decoder](seanet-decoder.md) - Detailed decoder architecture
- [State Management](../STREAMING/state-management.md) - StatefulModule pattern
- [Overlap-Add](../STREAMING/conv-transpose-overlap-add.md) - ConvTranspose streaming
