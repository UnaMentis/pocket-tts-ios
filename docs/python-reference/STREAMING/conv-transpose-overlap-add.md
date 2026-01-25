# StreamingConvTranspose1d: Overlap-Add Mechanism

> **CRITICAL**: This document describes the root cause of the audio amplitude issue in the Rust port.
> The Python implementation uses streaming with state buffers; Rust was using batch processing.

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/modules/conv.py`

---

## Overview

`StreamingConvTranspose1d` implements transposed convolution (upsampling) with streaming support via overlap-add. When upsampling with `stride < kernel_size`, consecutive frames overlap. The overlap-add mechanism accumulates these overlapping regions across frames.

**Key Insight:** Without proper overlap-add, each frame is processed independently, losing the accumulated signal energy in overlap regions. This causes ~5-6x lower amplitude in the output.

---

## Complete Python Implementation

```python
class StreamingConvTranspose1d(StatefulModule):
    """ConvTranspose1d with streaming support via overlap-add.

    State Buffer:
        partial: Tensor of shape (B, out_channels, kernel_size - stride)
                 Stores the trailing samples from the previous frame
                 that overlap with the next frame's output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias
        )

    @property
    def _stride(self) -> int:
        return self.convtr.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.convtr.kernel_size[0]

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        K = self._kernel_size
        S = self._stride
        # Overlap length = kernel - stride
        # This is how many samples from each frame overlap with the next
        return dict(partial=torch.zeros(batch_size, self.convtr.out_channels, K - S))

    def forward(self, x, mimi_state: dict):
        # Get this layer's state from the global state dict
        layer_state = self.get_state(mimi_state)["partial"]

        # Run transposed convolution
        y = self.convtr(x)

        # Overlap length
        PT = layer_state.shape[-1]  # = kernel_size - stride

        if PT > 0:
            # STEP 1: Add the previous frame's tail to current frame's head
            y[..., :PT] += layer_state

            # STEP 2: Extract this frame's tail (will be added to next frame)
            bias = self.convtr.bias
            for_partial = y[..., -PT:]

            # STEP 3: CRITICAL - Remove bias before storing
            # The bias gets added during conv, but we only want it once
            # when the final output is produced, not accumulated across frames
            if bias is not None:
                for_partial -= bias[:, None]

            # STEP 4: Store tail for next frame
            layer_state[:] = for_partial

            # STEP 5: Return non-overlapping portion
            # The tail is incomplete - it needs the next frame's contribution
            y = y[..., :-PT]

        return y
```

---

## Algorithm Visualization

### Example: kernel=8, stride=4, overlap=4

```
Frame N output:     [====|====|====]
                     ↑         ↑
                   head      tail (stored in partial)

Frame N+1 output:       [====|====|====]
                         ↑         ↑
                   (+ partial)   new tail

Final output:       [====|====][====|====]...
                    └─ Frame N ─┘└ Frame N+1 ┘
```

### Step-by-step for Frame N:

1. **ConvTranspose1d output:** `y` has shape `[B, C, T_out]` where `T_out = (T_in - 1) * stride + kernel`
2. **Add previous partial:** `y[:, :, :overlap] += partial`
3. **Extract new partial:** `for_partial = y[:, :, -overlap:]`
4. **Subtract bias:** `for_partial -= bias` (if bias exists)
5. **Store partial:** `partial = for_partial`
6. **Return non-overlap:** `return y[:, :, :-overlap]`

---

## State Buffer Dimensions

For the SEANet decoder with upsampling ratios `[8, 5, 4, 2]`:

| Layer | Type | Kernel | Stride | Overlap (K-S) | Channels | State Shape |
|-------|------|--------|--------|---------------|----------|-------------|
| Mimi Upsample | depthwise | 32 | 16 | 16 | 512 | `[B, 512, 16]` |
| SEANet Stage 0 | standard | 16 | 8 | 8 | 256 | `[B, 256, 8]` |
| SEANet Stage 1 | standard | 10 | 5 | 5 | 128 | `[B, 128, 5]` |
| SEANet Stage 2 | standard | 8 | 4 | 4 | 64 | `[B, 64, 4]` |
| SEANet Stage 3 | standard | 4 | 2 | 2 | 32 | `[B, 32, 2]` |

**Note:** The kernel sizes are `ratio * 2` (e.g., ratio=8 → kernel=16, ratio=5 → kernel=10).

---

## Critical Implementation Details

### 1. Bias Subtraction

The bias is subtracted from `for_partial` before storing:
```python
if bias is not None:
    for_partial -= bias[:, None]
```

**Why:** The ConvTranspose1d adds bias to the entire output. If we store `for_partial` with bias included, then when we add it to the next frame's head, that region gets the bias twice. Subtracting before storage ensures each output sample receives the bias exactly once.

### 2. In-Place State Update

```python
layer_state[:] = for_partial
```

The `[:]` syntax updates the state tensor in-place. This is important because the state dict holds references to these tensors.

### 3. State Initialization

States are initialized to zeros:
```python
return dict(partial=torch.zeros(batch_size, self.convtr.out_channels, K - S))
```

This means the first frame's output head has zeros added to it, which is correct - there's no previous frame to overlap with.

### 4. Groups (Depthwise) Convolution

For the Mimi upsample layer (`groups=dimension`), each channel is upsampled independently:
```python
self.convtr = StreamingConvTranspose1d(
    dimension, dimension,
    kernel_size=2 * stride,  # 32 for stride=16
    stride=stride,           # 16
    groups=dimension,        # 512 (depthwise)
    bias=False,              # No bias for depthwise
)
```

---

## Rust Implementation Considerations

### Current Issue

The Rust implementation in `src/models/mimi.rs` has a `forward_streaming` method with state structures, but:

1. **State may be reinitialized per call** instead of maintained across calls
2. **Bias subtraction** may be missing or incorrect
3. **Frame size mismatch** - Python processes 1 latent frame at a time; Rust may batch

### Required Changes

1. **Maintain state across calls:**
   ```rust
   // State should persist between generate() calls
   struct ConvTranspose1dState {
       partial: Tensor,  // shape: [batch, out_channels, kernel - stride]
   }
   ```

2. **Implement overlap-add correctly:**
   ```rust
   fn forward(&self, x: &Tensor, state: &mut ConvTranspose1dState) -> Tensor {
       let y = self.convtr.forward(x)?;
       let pt = state.partial.dim(-1)?;

       if pt > 0 {
           // Add previous partial to head
           let head = y.narrow(-1, 0, pt)?;
           let head = head.add(&state.partial)?;
           y.slice_assign(-1, 0, pt, &head)?;

           // Extract and store new partial (minus bias)
           let tail = y.narrow(-1, y.dim(-1)? - pt, pt)?;
           if let Some(bias) = &self.bias {
               let tail = tail.sub(&bias.unsqueeze(-1)?)?;
               state.partial = tail;
           } else {
               state.partial = tail;
           }

           // Return non-overlap portion
           y.narrow(-1, 0, y.dim(-1)? - pt)
       } else {
           y
       }
   }
   ```

3. **Process one latent frame at a time** during generation

---

## Verification

### Test Case

To verify the implementation:

1. Generate a single latent frame through Python, capturing intermediate values
2. Generate the same latent through Rust
3. Compare:
   - `y` before overlap-add
   - `layer_state` after update
   - Final output

### Expected Behavior

- First frame: `partial` starts as zeros, output head unchanged
- Subsequent frames: `partial` accumulates, amplitude grows correctly
- Final amplitude should be ~5-6x higher than batch processing

---

## Related Files

- `conv.py:117-161` - StreamingConvTranspose1d implementation
- `seanet.py:116-180` - SEANetDecoder uses StreamingConvTranspose1d
- `mimi.py:81-86` - MimiModel.decode_from_latent orchestrates decoding
- `stateful_module.py` - Base class for state management
