# RoPE: Rotary Position Embeddings

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/modules/rope.py`

---

## Overview

RoPE (Rotary Position Embedding) encodes position information by rotating query and key vectors in a complex plane. This allows the model to understand relative positions through the rotation angle.

---

## Implementation

```python
def apply_rope(
    q: torch.Tensor,  # [B, T, H, D]
    k: torch.Tensor,  # [B, T, H, D]
    offset: int | torch.Tensor = 0,
    max_period: int | float = 10_000,
):
    """Apply rotary position embedding to Q and K.

    Args:
        q: Queries [B, T, H, D]
        k: Keys [B, T, H, D]
        offset: Position offset for streaming
        max_period: Base for frequency computation
    """
    B, T, H, D = q.shape

    # Compute frequencies for each dimension pair
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))

    # Position indices with offset
    ts = torch.arange(T, device=q.device, dtype=torch.float32)
    ts += offset
    ts = ts.view(-1, 1, 1)  # [T, 1, 1]

    # Reshape to pair consecutive dimensions
    q = q.view(B, T, H, D // 2, 2)  # [B, T, H, D/2, 2]
    k = k.view(B, T, H, D // 2, 2)

    # Extract real and imaginary parts
    qr = q[..., 0].float()  # Real
    qi = q[..., 1].float()  # Imaginary

    kr = k[..., 0].float()
    ki = k[..., 1].float()

    # Rotation matrix components
    rotr = torch.cos(freqs * ts)  # [T, D/2, 1]
    roti = torch.sin(freqs * ts)

    # Apply rotation (complex multiplication)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    # Stack back to original format
    qo = torch.stack([qor, qoi], dim=-1)  # [B, T, H, D/2, 2]
    ko = torch.stack([kor, koi], dim=-1)

    return qo.view(B, T, H, D), ko.view(B, T, H, D)
```

---

## Key Details

### 1. Interleaved Pairing (Not Split Halves)

Pocket TTS uses **interleaved** pairing:
```python
# Pairs: (dim0, dim1), (dim2, dim3), ...
q = q.view(B, T, H, D // 2, 2)  # [..., 2] are paired
```

This is different from LLaMA-style split halves:
```python
# LLaMA: (dim0, dim32), (dim1, dim33), ... for D=64
# NOT used here
```

### 2. Frequency Computation

```python
ds = torch.arange(D // 2)  # [0, 1, 2, ..., D/2-1]
freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
# = max_period^(-2*ds/D)
```

Lower dimensions have higher frequencies (change faster with position).

### 3. Offset for Streaming

```python
ts = torch.arange(T) + offset
```

When streaming, `offset` equals the number of cached positions. This ensures new positions get correct rotations relative to cached keys.

### 4. Complex Multiplication

The rotation is a 2D rotation matrix applied as complex multiplication:
```
[cos θ  -sin θ] [real]   [real * cos - imag * sin]
[sin θ   cos θ] [imag] = [real * sin + imag * cos]
```

---

## Rust Implementation Notes

### Interleaved vs Split

```rust
// CORRECT: Interleaved pairing
// For D=64, pairs are: (0,1), (2,3), (4,5), ...
let pairs = q.reshape(&[b, t, h, d / 2, 2])?;

// WRONG: Split halves
// For D=64, pairs would be: (0,32), (1,33), (2,34), ...
let (first_half, second_half) = q.split_at(d / 2)?;
```

### Frequency Formula

```rust
let freqs: Vec<f32> = (0..d/2)
    .map(|i| {
        let exp = (i as f32) * (-f32::ln(max_period) * 2.0 / (d as f32));
        exp.exp()
    })
    .collect();
```

### Position With Offset

```rust
let positions: Vec<f32> = (0..t)
    .map(|i| (i + offset) as f32)
    .collect();
```

---

## Related Documents

- [Transformer](transformer.md) - Where RoPE is applied
- [FlowLM](../ARCHITECTURE/flowlm.md) - Transformer configuration
