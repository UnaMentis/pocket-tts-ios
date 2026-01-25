# SEANet Decoder Architecture

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/modules/seanet.py`

---

## Overview

SEANet (Streaming Encoder/decoder Audio Network) is the audio codec architecture used in Mimi. The decoder upsamples latent representations to audio waveforms through a series of transposed convolutions and residual blocks.

---

## Complete Layer Structure

```python
# Default configuration from seanet.py
SEANetDecoder(
    channels=1,            # Output audio channels (mono)
    dimension=128,         # Latent dimension from Mimi (after projection)
    n_filters=32,          # Base filter count
    n_residual_layers=3,   # ResBlocks per upsample stage
    ratios=[8, 5, 4, 2],   # Upsample ratios (total: 320x)
    kernel_size=7,         # Initial/final conv kernel
    last_kernel_size=7,    # Final conv kernel
    residual_kernel_size=3, # ResBlock kernel
    dilation_base=2,       # Dilation growth rate
    pad_mode="reflect",    # Padding mode
    compress=2,            # Channel compression in ResBlocks
)
```

---

## Layer-by-Layer Breakdown

### Input
- Shape: `[B, 512, T]` (after Mimi's decoder transformer)
- T = number of timesteps at 200 Hz (after 16x upsample from 12.5 Hz)

### Layer 0: Initial Convolution

```python
StreamingConv1d(
    in_channels=512,       # dimension * 2^4 (for 4 upsample stages)
    out_channels=512,
    kernel_size=7,
    stride=1,
    pad_mode="reflect"
)
```

- Output: `[B, 512, T]`
- State: `previous` buffer of shape `[B, 512, 6]`

### Upsample Stages

For each ratio in `[8, 5, 4, 2]`:

#### Stage 0 (ratio=8)

```
ELU(alpha=1.0)
    ↓
StreamingConvTranspose1d(512, 256, kernel=16, stride=8)
    ↓ [B, 256, T*8]
    ↓
ResBlock(256, dilations=[1, 1])
ResBlock(256, dilations=[2, 1])
ResBlock(256, dilations=[4, 1])
```

State: `partial` buffer of shape `[B, 256, 8]` (kernel - stride = 16 - 8)

#### Stage 1 (ratio=5)

```
ELU(alpha=1.0)
    ↓
StreamingConvTranspose1d(256, 128, kernel=10, stride=5)
    ↓ [B, 128, T*8*5]
    ↓
ResBlock(128, dilations=[1, 1])
ResBlock(128, dilations=[2, 1])
ResBlock(128, dilations=[4, 1])
```

State: `partial` buffer of shape `[B, 128, 5]`

#### Stage 2 (ratio=4)

```
ELU(alpha=1.0)
    ↓
StreamingConvTranspose1d(128, 64, kernel=8, stride=4)
    ↓ [B, 64, T*8*5*4]
    ↓
ResBlock(64, dilations=[1, 1])
ResBlock(64, dilations=[2, 1])
ResBlock(64, dilations=[4, 1])
```

State: `partial` buffer of shape `[B, 64, 4]`

#### Stage 3 (ratio=2)

```
ELU(alpha=1.0)
    ↓
StreamingConvTranspose1d(64, 32, kernel=4, stride=2)
    ↓ [B, 32, T*8*5*4*2]
    ↓
ResBlock(32, dilations=[1, 1])
ResBlock(32, dilations=[2, 1])
ResBlock(32, dilations=[4, 1])
```

State: `partial` buffer of shape `[B, 32, 2]`

### Final Convolution

```python
ELU(alpha=1.0)
    ↓
StreamingConv1d(
    in_channels=32,        # n_filters
    out_channels=1,        # channels (mono audio)
    kernel_size=7,
    stride=1,
    pad_mode="reflect"
)
```

- Output: `[B, 1, T * 320]`
- State: `previous` buffer of shape `[B, 32, 6]`

---

## ResBlock Structure

Each ResBlock has 2 convolutions with a skip connection:

```python
class SEANetResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] = [3, 1],
        dilations: list[int] = [1, 1],  # Second is always 1
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        hidden = dim // compress  # Channel compression

        # Block structure:
        # [ELU, Conv(dim→hidden, k=3, d=dilation), ELU, Conv(hidden→dim, k=1)]
        self.block = [
            nn.ELU(alpha=1.0),
            StreamingConv1d(dim, hidden, kernel_size=3, dilation=dilations[0]),
            nn.ELU(alpha=1.0),
            StreamingConv1d(hidden, dim, kernel_size=1),
        ]

    def forward(self, x, model_state):
        v = x
        for layer in self.block:
            if isinstance(layer, StreamingConv1d):
                v = layer(v, model_state)
            else:
                v = layer(v)  # ELU
        return x + v  # Residual connection
```

### Dilation Pattern

For 3 residual layers per stage:
- ResBlock 0: dilation = `dilation_base^0` = 1
- ResBlock 1: dilation = `dilation_base^1` = 2
- ResBlock 2: dilation = `dilation_base^2` = 4

This expands the receptive field without increasing parameter count.

---

## Complete State Dictionary

For batch_size=1, here are all streaming states:

```python
{
    # Initial conv
    "model.0": {
        "previous": Tensor([1, 512, 6]),
        "first": Tensor([1], dtype=bool)
    },

    # Stage 0 (ratio=8)
    "model.2": {"partial": Tensor([1, 256, 8])},  # ConvTranspose
    "model.3.block.1": {"previous": Tensor([1, 128, 2]), "first": ...},  # ResBlock0 Conv1
    "model.3.block.3": {"previous": Tensor([1, 256, 0]), "first": ...},  # ResBlock0 Conv2 (k=1)
    "model.4.block.1": {"previous": Tensor([1, 128, 4]), "first": ...},  # ResBlock1 Conv1 (d=2)
    "model.4.block.3": {"previous": Tensor([1, 256, 0]), "first": ...},
    "model.5.block.1": {"previous": Tensor([1, 128, 8]), "first": ...},  # ResBlock2 Conv1 (d=4)
    "model.5.block.3": {"previous": Tensor([1, 256, 0]), "first": ...},

    # Stage 1 (ratio=5)
    "model.7": {"partial": Tensor([1, 128, 5])},
    # ... ResBlocks ...

    # Stage 2 (ratio=4)
    "model.12": {"partial": Tensor([1, 64, 4])},
    # ... ResBlocks ...

    # Stage 3 (ratio=2)
    "model.17": {"partial": Tensor([1, 32, 2])},
    # ... ResBlocks ...

    # Final conv
    "model.22": {"previous": Tensor([1, 32, 6]), "first": ...}
}
```

---

## Key Implementation Notes

### 1. Activation Function

SEANet uses **ELU(alpha=1.0)** throughout:
```python
nn.ELU(alpha=1.0)
# ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
```

**NOT tanh** on the output layer. The Rust port initially had an incorrect tanh.

### 2. Kernel Size Formula

For ConvTranspose layers:
```python
kernel_size = ratio * 2  # e.g., ratio=8 → kernel=16
stride = ratio           # e.g., stride=8
overlap = kernel - stride  # e.g., 16-8=8
```

### 3. Channel Progression

Starting from `mult = 2^len(ratios) = 16`:
```
512 → 256 → 128 → 64 → 32 → 1
(mult=16)  (mult=8)  (mult=4)  (mult=2)  (mult=1)  (output)
```

### 4. No Bias in ResBlock Conv2

The second convolution (k=1) in ResBlocks has bias=True by default.

---

## Tensor Shape Flow Example

For input `[1, 512, 16]` (16 frames at 200 Hz):

```
Input:          [1, 512, 16]
Initial Conv:   [1, 512, 16]  (same, k=7, s=1)
Upsample 8x:    [1, 256, 128] (after ConvTranspose + ResBlocks)
Upsample 5x:    [1, 128, 640]
Upsample 4x:    [1, 64, 2560]
Upsample 2x:    [1, 32, 5120]
Final Conv:     [1, 1, 5120]  (5120 samples = 213ms @ 24kHz)
```

---

## Rust Implementation Checklist

1. [ ] ELU activation with alpha=1.0 (not GELU, not SiLU)
2. [ ] ConvTranspose kernel = ratio * 2
3. [ ] Overlap-add state for each ConvTranspose
4. [ ] Causal padding state for each Conv1d
5. [ ] Dilation pattern: 1, 2, 4 for residual layers
6. [ ] Channel compression in ResBlocks (dim // 2)
7. [ ] No tanh on output layer
8. [ ] Correct layer ordering: ELU before each conv/convtr

---

## Related Documents

- [Overlap-Add](../STREAMING/conv-transpose-overlap-add.md) - ConvTranspose streaming
- [Conv1d Streaming](../STREAMING/conv1d-streaming.md) - Causal convolution
- [Mimi Decode](mimi-decode.md) - How SEANet is called
