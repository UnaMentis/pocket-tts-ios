# FlowNet & LSD Decode

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/modules/mlp.py`

---

## Overview

FlowNet (SimpleMLPAdaLN) generates latent audio representations via flow matching. It uses:
- **LSD (Lagrangian Self-Distillation)**: Multi-step denoising from noise to latent
- **AdaLN**: Adaptive layer normalization conditioned on time and context
- **Two-Time Conditioning**: Separate embeddings for start time `s` and target time `t`

---

## LSD Decode Algorithm

```python
def lsd_decode(v_t: FlowNet2, x_0: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
    """Lagrangian Self Distillation decode.

    Reconstructs data sample by integrating the learned flow field.

    Args:
        v_t: Flow network that takes (s, t, x) and returns velocity
        x_0: Starting noise [B, ldim]
        num_steps: Integration steps (default 10 for inference)

    Returns:
        x_1: Reconstructed latent [B, ldim]
    """
    current = x_0  # Start from noise

    for i in range(num_steps):
        s = i / num_steps        # Start time: 0, 0.1, 0.2, ...
        t = (i + 1) / num_steps  # Target time: 0.1, 0.2, 0.3, ...

        # Get velocity at current point
        flow_dir = v_t(
            s * torch.ones_like(x_0[..., :1]),  # [B, 1]
            t * torch.ones_like(x_0[..., :1]),  # [B, 1]
            current                              # [B, ldim]
        )

        # Euler integration step
        current += flow_dir / num_steps

    return current
```

**Key Insight:** Unlike standard diffusion (one time condition), LSD uses TWO times:
- `s`: Current position in the flow (where we are)
- `t`: Target position (where we're going)

---

## FlowNet Architecture (SimpleMLPAdaLN)

```python
class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,        # ldim = 32
        model_channels,     # 512 (internal dimension)
        out_channels,       # ldim = 32
        cond_channels,      # dim = 1024 (from transformer)
        num_res_blocks,     # 6 residual blocks
        num_time_conds=2,   # Two time embeddings (s and t)
    ):
        # Two separate time embedders
        self.time_embed = nn.ModuleList([
            TimestepEmbedder(model_channels) for _ in range(num_time_conds)
        ])

        # Condition embedding
        self.cond_embed = nn.Linear(cond_channels, model_channels)

        # Input projection
        self.input_proj = nn.Linear(in_channels, model_channels)

        # Residual blocks with AdaLN
        self.res_blocks = nn.ModuleList([
            ResBlock(model_channels) for _ in range(num_res_blocks)
        ])

        # Final output layer
        self.final_layer = FinalLayer(model_channels, out_channels)
```

---

## Forward Pass

```python
def forward(
    self, c: torch.Tensor, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        c: Conditioning from transformer [B, 1024]
        s: Start time [B, 1]
        t: Target time [B, 1]
        x: Current noise/latent [B, 32]

    Returns:
        velocity: Flow direction [B, 32]
    """
    # Project input
    x = self.input_proj(x)  # [B, 512]

    # Time embeddings (averaged)
    ts = [s, t]
    t_combined = sum(
        self.time_embed[i](ts[i]) for i in range(self.num_time_conds)
    ) / self.num_time_conds  # [B, 512]

    # Condition embedding
    c = self.cond_embed(c)  # [B, 512]

    # Combined modulation signal
    y = t_combined + c  # [B, 512]

    # Residual blocks with AdaLN
    for block in self.res_blocks:
        x = block(x, y)

    # Final layer
    return self.final_layer(x, y)  # [B, 32]
```

---

## TimestepEmbedder

```python
class TimestepEmbedder(nn.Module):
    """Sinusoidal time embedding with MLP."""

    def __init__(
        self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10000
    ):
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            RMSNorm(hidden_size)
        )

        # Precompute frequencies
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        # t: [B, 1]
        args = t * self.freqs.to(t.dtype)  # [B, 128]

        # IMPORTANT: Order is [cos, sin], NOT [sin, cos]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, 256]

        return self.mlp(embedding)  # [B, 512]
```

**Critical:** The ordering is `[cos, sin]`, not `[sin, cos]`. This was a porting issue.

---

## ResBlock with AdaLN

```python
def modulate(x, shift, scale):
    """Adaptive layer norm modulation."""
    return x * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, channels):
        self.in_ln = LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels)
        )

    def forward(self, x, y):
        # y: combined time + condition embedding

        # AdaLN modulation: [shift, scale, gate]
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)

        # Apply modulation to normalized input
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)

        # MLP
        h = self.mlp(h)

        # Gated residual connection
        return x + gate_mlp * h
```

**Critical:** The chunk order is `[shift, scale, gate]`, not `[scale, shift, gate]`.

---

## FinalLayer

```python
class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        # LayerNorm WITHOUT affine parameters
        self.norm_final = LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)

        self.linear = nn.Linear(model_channels, out_channels)

        # AdaLN outputs 2x channels (shift, scale only, no gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels)
        )

    def forward(self, x, c):
        # Get modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)

        # Normalize (no learned params) then modulate
        x = modulate(self.norm_final(x), shift, scale)

        # Project to output dimension
        return self.linear(x)
```

**Critical:**
- `norm_final` has `elementwise_affine=False` (no learned weight/bias)
- Chunk order is `[shift, scale]` (2 outputs, not 3)

---

## Configuration Values

```python
# From FlowLMConfig
flow_dim = 512           # Internal MLP dimension
flow_depth = 6           # Number of ResBlocks
num_time_conds = 2       # Always 2 (s and t)
frequency_embedding_size = 256
max_period = 10000
```

---

## Key Implementation Notes

### 1. Time Embedding Order

```python
# CORRECT: [cos, sin]
embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

# WRONG: [sin, cos]
embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

### 2. AdaLN Chunk Order

For ResBlock (3 outputs):
```python
shift, scale, gate = modulation.chunk(3, dim=-1)  # CORRECT
scale, shift, gate = modulation.chunk(3, dim=-1)  # WRONG
```

For FinalLayer (2 outputs):
```python
shift, scale = modulation.chunk(2, dim=-1)  # CORRECT
```

### 3. RMSNorm in TimestepEmbedder

The time embedder ends with RMSNorm, not LayerNorm:
```python
self.mlp = nn.Sequential(
    ...,
    RMSNorm(hidden_size)  # NOT LayerNorm
)
```

### 4. FinalLayer Norm Has No Affine

```python
LayerNorm(channels, elementwise_affine=False)  # No learned params
```

---

## Rust Implementation Checklist

- [ ] Two separate TimestepEmbedders
- [ ] Time embeddings averaged (not summed)
- [ ] `[cos, sin]` ordering in sinusoidal embedding
- [ ] ResBlock AdaLN: `[shift, scale, gate]`
- [ ] FinalLayer AdaLN: `[shift, scale]`
- [ ] FinalLayer norm has no affine parameters
- [ ] RMSNorm at end of TimestepEmbedder
- [ ] SiLU activation (not GELU) in ResBlock

---

## Related Documents

- [FlowLM](flowlm.md) - How FlowNet is called
- [MLP Module](../MODULES/mlp.md) - MLP implementation details
- [LayerNorm](../MODULES/layer-norm.md) - Normalization variants
