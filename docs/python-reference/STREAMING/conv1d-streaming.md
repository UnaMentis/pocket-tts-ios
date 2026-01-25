# StreamingConv1d: Causal Convolution with Context Buffer

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/modules/conv.py`

---

## Overview

`StreamingConv1d` implements 1D convolution with causal (streaming) support. For causal convolution, the output at time `t` can only depend on inputs at times `≤ t`. This is achieved by prepending context from previous frames.

**Key Concept:** Each frame needs `kernel_size - stride` samples of context from the previous frame to compute its output correctly.

---

## Complete Python Implementation

```python
def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """Calculate extra padding needed to ensure last window is full."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad input so the last convolution window is full.

    Without this, transposed convolution can't reconstruct the same length.
    Extra padding is added at the end (causal).
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


class StreamingConv1d(StatefulModule):
    """Conv1d with causal padding and streaming support.

    State:
        previous: Tensor of shape (B, in_channels, kernel - stride)
                  Context from previous frame
        first: BoolTensor of shape (B,)
               True if this is the first frame (for replicate padding)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        assert pad_mode in ["constant", "replicate"], pad_mode
        self.pad_mode = pad_mode

        # Warn about unusual stride + dilation combination
        if stride > 1 and dilation > 1:
            warnings.warn(
                f"StreamingConv1d initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    @property
    def _stride(self) -> int:
        return self.conv.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.conv.kernel_size[0]

    @property
    def _effective_kernel_size(self) -> int:
        """Kernel size accounting for dilation."""
        dilation = self.conv.dilation[0]
        return (self._kernel_size - 1) * dilation + 1

    def init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
        stride = self._stride
        kernel = self._effective_kernel_size  # Accounts for dilation
        # Context length = effective_kernel - stride
        previous = torch.zeros(batch_size, self.conv.in_channels, kernel - stride)
        first = torch.ones(batch_size, dtype=torch.bool)
        return dict(previous=previous, first=first)

    def forward(self, x, model_state: dict | None):
        B, C, T = x.shape
        S = self._stride

        # Input length must be multiple of stride
        assert T > 0 and T % S == 0, "Steps must be multiple of stride"

        # Get or initialize state
        if model_state is None:
            state = self.init_state(B, 0)
        else:
            state = self.get_state(model_state)

        # Context length from previous frame
        TP = state["previous"].shape[-1]

        # Handle first frame with replicate padding
        if TP and self.pad_mode == "replicate":
            assert T >= TP, "Not enough content to pad streaming."
            init = x[..., :1]  # First sample
            # If first frame, fill context with first sample
            state["previous"][:] = torch.where(
                state["first"].view(-1, 1, 1), init, state["previous"]
            )

        # Prepend context from previous frame
        if TP:
            x = torch.cat([state["previous"], x], dim=-1)

        # Run convolution
        y = self.conv(x)

        # Store context for next frame
        if TP:
            state["previous"][:] = x[..., -TP:]
            if self.pad_mode == "replicate":
                state["first"] = torch.zeros_like(state["first"])

        return y
```

---

## Algorithm Visualization

### Example: kernel=7, stride=1, dilation=1

Context needed: `7 - 1 = 6` samples

```
Previous frame:  [...|123456]
                      └─────┘ stored in 'previous'

Current frame:   [ABCDEFGH...]
                  ↓
After concat:    [123456|ABCDEFGH...]
                  └─────┘ context
                  ↓
Conv output:     [xxxxxxxx...]
                  ↓
Store context:   [...|FGH...]  (last 6 samples)
```

### First Frame with Replicate Padding

```
Input:           [ABCDEFGH...]
                  ↓
If first & replicate mode:
  previous =     [AAAAAA]  (first sample repeated)
                  ↓
After concat:    [AAAAAA|ABCDEFGH...]
```

---

## State Buffer Dimensions

For SEANet layers:

### Initial Convolution
| Layer | Kernel | Stride | Dilation | Context | Channels |
|-------|--------|--------|----------|---------|----------|
| input_conv | 7 | 1 | 1 | 6 | varies |

### Residual Block Convolutions
Each ResBlock has 2 convolutions with increasing dilation:

| ResBlock Index | Conv1 Dilation | Conv1 Context | Conv2 | Conv2 Context |
|----------------|----------------|---------------|-------|---------------|
| 0 | 1 | `(3-1)*1 = 2` | k=1 | 0 |
| 1 | 2 | `(3-1)*2 = 4` | k=1 | 0 |
| 2 | 4 | `(3-1)*4 = 8` | k=1 | 0 |

**Note:** Kernel size 1 convolutions have no context requirement (`1 - 1 = 0`).

### Downsampling (Encoder only)
| Ratio | Kernel | Stride | Context |
|-------|--------|--------|---------|
| 8 | 16 | 8 | 8 |
| 5 | 10 | 5 | 5 |
| 4 | 8 | 4 | 4 |
| 2 | 4 | 2 | 2 |

---

## Key Implementation Details

### 1. Effective Kernel Size with Dilation

Dilation spreads out the kernel:
```python
effective_kernel = (kernel_size - 1) * dilation + 1
```

Example: kernel=3, dilation=4
- Effective kernel = (3-1)*4 + 1 = 9
- Context needed = 9 - stride

### 2. Replicate vs Constant Padding

- **Constant (default):** Context initialized to zeros
- **Replicate:** First sample repeated to fill context

SEANet uses `pad_mode="reflect"` in constructor but the streaming code uses `"replicate"`:
```python
# From seanet.py - SEANet uses reflect for non-streaming, replicate for streaming
StreamingConv1d(..., pad_mode=pad_mode)  # pad_mode is passed through
```

### 3. Stride Requirement

```python
assert T > 0 and T % S == 0, "Steps must be multiple of stride"
```

Input length must be divisible by stride. For stride=1 convolutions, any length works.

### 4. First Frame Detection

The `first` boolean tensor tracks whether this is the first frame:
```python
first = torch.ones(batch_size, dtype=torch.bool)  # Initially True
# After first frame with replicate mode:
state["first"] = torch.zeros_like(state["first"])  # Set to False
```

---

## Rust Implementation Considerations

### State Structure

```rust
struct Conv1dState {
    previous: Tensor,  // shape: [batch, in_channels, effective_kernel - stride]
    first: Tensor,     // shape: [batch], dtype: bool
}
```

### Forward Implementation

```rust
fn forward(&self, x: &Tensor, state: &mut Conv1dState) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    let tp = state.previous.dim(-1)?;

    // Handle first frame replicate padding
    if tp > 0 && self.pad_mode == PadMode::Replicate {
        let init = x.narrow(-1, 0, 1)?;
        // Where first is true, fill with init; else keep previous
        state.previous = state.first
            .unsqueeze(-1)?
            .unsqueeze(-1)?
            .where_cond(&init.broadcast_as(&state.previous)?, &state.previous)?;
    }

    // Concat previous context
    let x_padded = if tp > 0 {
        Tensor::cat(&[&state.previous, x], -1)?
    } else {
        x.clone()
    };

    // Run convolution
    let y = self.conv.forward(&x_padded)?;

    // Store context for next frame
    if tp > 0 {
        state.previous = x_padded.narrow(-1, x_padded.dim(-1)? - tp, tp)?;
        if self.pad_mode == PadMode::Replicate {
            state.first = Tensor::zeros_like(&state.first)?;
        }
    }

    Ok(y)
}
```

---

## Verification

### Test: Context Propagation

1. Process frame 1 with known input
2. Check `state.previous` contains last `TP` samples of concatenated input
3. Process frame 2, verify output uses correct context

### Test: First Frame Replicate

1. Set `pad_mode = "replicate"`
2. Process frame with input `[1, 2, 3, 4, 5, 6, 7, 8]`
3. Verify context was `[1, 1, 1, 1, 1, 1]` (first sample repeated)

---

## Related Files

- `conv.py:36-114` - StreamingConv1d implementation
- `seanet.py:7-41` - SEANetResnetBlock uses StreamingConv1d
- `seanet.py:44-113` - SEANetEncoder structure
- `seanet.py:116-180` - SEANetDecoder structure
