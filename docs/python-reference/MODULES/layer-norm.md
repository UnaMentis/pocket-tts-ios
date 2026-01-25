# LayerNorm and RMSNorm

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/modules/mlp.py`

---

## Overview

Pocket TTS uses two normalization variants:
- **LayerNorm**: Standard layer normalization (mean + variance)
- **RMSNorm**: Root mean square normalization (variance only)

---

## LayerNorm Implementation

```python
class LayerNorm(nn.Module):
    """Custom LayerNorm that supports JVP (used in some training scenarios)."""

    def __init__(self, channels, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # Compute mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learned scale and shift if affine
        if hasattr(self, "weight"):
            x = x * self.weight + self.bias

        return x
```

### Usage

| Location | Affine | eps |
|----------|--------|-----|
| FlowLM `out_norm` | Yes | 1e-5 |
| ResBlock `in_ln` | Yes | 1e-6 |
| FinalLayer `norm_final` | **No** | 1e-6 |

---

## RMSNorm Implementation

```python
def _rms_norm(x: torch.Tensor, alpha: torch.Tensor, eps: float):
    """RMS normalization without mean subtraction."""
    x_dtype = x.dtype

    # Variance (RMS squared)
    var = eps + x.var(dim=-1, keepdim=True)

    # Scale and normalize
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)

    return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.full((dim,), 1.0))

    def forward(self, x: torch.Tensor):
        return _rms_norm(x, self.alpha, self.eps)
```

### Key Difference from Standard RMSNorm

This implementation computes variance (not just RMS):
```python
var = eps + x.var(dim=-1, keepdim=True)  # Uses variance
# NOT:
# rms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
```

**Important:** The variance formula:
- Includes mean subtraction internally via `torch.var`
- Uses `unbiased=True` (default) unlike LayerNorm

### Usage

| Location | Description |
|----------|-------------|
| TimestepEmbedder (end of MLP) | Final layer before time embedding output |
| Not used in transformers | Transformers use LayerNorm |

---

## Critical Implementation Notes

### 1. FinalLayer Norm Has No Affine

```python
# In FinalLayer.__init__:
self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
```

This was a common porting bug - having affine parameters when there shouldn't be any.

### 2. Variance Calculation

LayerNorm uses `unbiased=False`:
```python
var = x.var(dim=-1, unbiased=False, keepdim=True)
```

RMSNorm uses default `unbiased=True`:
```python
var = x.var(dim=-1, keepdim=True)  # Default unbiased=True
```

### 3. eps Values

- LayerNorm: `1e-6` (in FlowNet), `1e-5` (in FlowLM out_norm)
- RMSNorm: `1e-5`

### 4. dtype Preservation

Both preserve input dtype:
```python
y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
```

---

## Rust Implementation

### LayerNorm

```rust
fn layer_norm(x: &Tensor, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> Tensor {
    let mean = x.mean_keepdim(-1)?;
    let var = x.var_keepdim(-1, false)?;  // unbiased=false

    let normalized = (x - &mean)? / (var + eps)?.sqrt()?;

    match (weight, bias) {
        (Some(w), Some(b)) => &normalized * w + b,
        _ => normalized,
    }
}
```

### RMSNorm (Pocket TTS Variant)

```rust
fn rms_norm(x: &Tensor, alpha: &Tensor, eps: f32) -> Tensor {
    let dtype = x.dtype();
    let var = x.var_keepdim(-1, true)?;  // unbiased=true

    let scale = (alpha.to_dtype(var.dtype())? * (var + eps)?.rsqrt()?)?;
    (x * &scale)?.to_dtype(dtype)
}
```

---

## Related Documents

- [FlowNet/LSD](../ARCHITECTURE/flownet-lsd.md) - Where these norms are used
- [MLP](mlp.md) - Full MLP implementation
