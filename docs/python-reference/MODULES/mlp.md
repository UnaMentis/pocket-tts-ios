# MLP and Feed-Forward Networks

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/modules/mlp.py`

---

## Components

Pocket TTS uses several MLP variants:

| Component | Activation | Location |
|-----------|------------|----------|
| Transformer FFN | GELU | FlowLM/Mimi transformer layers |
| ResBlock MLP | SiLU | FlowNet residual blocks |
| TimestepEmbedder | SiLU | Time embedding |
| AdaLN modulation | SiLU | Before AdaLN chunk |

---

## Transformer FFN

In transformer layers:
```python
# Two-layer MLP with GELU activation
self.mlp = nn.Sequential(
    nn.Linear(d_model, dim_feedforward),  # 1024 → 4096
    nn.GELU(),
    nn.Linear(dim_feedforward, d_model),  # 4096 → 1024
)
```

---

## FlowNet ResBlock MLP

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),  # 512 → 512
            nn.SiLU(),
            nn.Linear(channels, channels),  # 512 → 512
        )
```

**Note:** SiLU (Swish), not GELU.

---

## TimestepEmbedder MLP

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),  # 256 → 512
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),               # 512 → 512
            RMSNorm(hidden_size),                              # Normalize
        )
```

---

## AdaLN Modulation

Used in ResBlock and FinalLayer to compute adaptive normalization parameters:

```python
# ResBlock: outputs 3 values (shift, scale, gate)
self.adaLN_modulation = nn.Sequential(
    nn.SiLU(),
    nn.Linear(channels, 3 * channels)  # 512 → 1536
)

# FinalLayer: outputs 2 values (shift, scale)
self.adaLN_modulation = nn.Sequential(
    nn.SiLU(),
    nn.Linear(model_channels, 2 * model_channels)  # 512 → 1024
)
```

The modulate function:
```python
def modulate(x, shift, scale):
    return x * (1 + scale) + shift
```

---

## Activation Functions

### SiLU (Swish)
```python
# SiLU(x) = x * sigmoid(x)
nn.SiLU()
```

Used in: FlowNet (ResBlocks, TimestepEmbedder, AdaLN modulation)

### GELU
```python
# GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
nn.GELU()
```

Used in: Transformer FFN

### ELU
```python
# ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
nn.ELU(alpha=1.0)
```

Used in: SEANet decoder only

---

## Key Implementation Notes

### 1. No Bias Variation

Most MLPs have bias. The exception:
```python
# FlowLM input projection
self.input_linear = nn.Linear(ldim, dim, bias=False)
```

### 2. Hidden Dimension Ratios

- Transformer FFN: 4x expansion (1024 → 4096 → 1024)
- FlowNet ResBlock: 1x (no expansion)
- TimestepEmbedder: 2x expansion (256 → 512)

### 3. Output Dimensions

AdaLN modulation outputs:
- ResBlock: 3x (shift, scale, gate)
- FinalLayer: 2x (shift, scale only)

---

## Rust Implementation

```rust
// SiLU activation
fn silu(x: &Tensor) -> Tensor {
    x * sigmoid(x)
}

// GELU activation (approximate)
fn gelu(x: &Tensor) -> Tensor {
    x * 0.5 * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x.sqr())).tanh())
}

// ELU activation
fn elu(x: &Tensor, alpha: f32) -> Tensor {
    x.where_cond(
        &x.gt(0.0),
        &(alpha * (x.exp() - 1.0)),
    )
}
```

---

## Related Documents

- [FlowNet/LSD](../ARCHITECTURE/flownet-lsd.md) - Full FlowNet architecture
- [LayerNorm](layer-norm.md) - Normalization used with MLPs
- [SEANet](../ARCHITECTURE/seanet-decoder.md) - Uses ELU activation
