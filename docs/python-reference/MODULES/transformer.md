# Transformer Module

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/modules/transformer.py`

---

## StreamingMultiheadAttention

```python
class StreamingMultiheadAttention(StatefulModule):
    """Multi-head attention with KV cache for streaming."""

    def __init__(self, embed_dim: int, num_heads: int, rope: RotaryEmbedding):
        self.embed_dim = embed_dim    # 1024
        self.num_heads = num_heads    # 16
        self.rope = rope

        # Combined QKV projection (no bias)
        dim_per_head = embed_dim // num_heads  # 64
        out_dim = embed_dim + 2 * embed_dim    # Q + K + V
        self.in_proj = nn.Linear(embed_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def init_state(self, batch_size: int, sequence_length: int):
        dim_per_head = self.embed_dim // self.num_heads
        return dict(
            current_end=torch.zeros((0,)),  # Position counter
            cache=torch.full(
                (2, batch_size, sequence_length, self.num_heads, dim_per_head),
                float("NaN"),  # Pre-allocate with NaN
            ),
        )

    def forward(self, query: torch.Tensor, model_state: dict):
        state = self.get_state(model_state)

        # Project to Q, K, V
        projected = self.in_proj(query)  # [B, T, 3*embed_dim]
        b, t, _ = projected.shape
        d = self.embed_dim // self.num_heads

        # Reshape: [B, T, 3, H, D]
        packed = projected.view(b, t, 3, self.num_heads, d)
        q, k, v = torch.unbind(packed, dim=2)

        # Apply RoPE to Q and K
        offset = state["current_end"].shape[0]
        q, k = self.rope(q, k, offset=offset)

        # Complete KV from cache
        k, v = self._complete_kv(k, v, state)

        # Build causal mask
        mask_shape = (query.shape[1], query.shape[1] + offset)
        attn_mask = _materialize_causal_mask(mask_shape, shift=offset)

        # Transpose for attention: [B, H, T, D]
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        # Scaled dot-product attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask)

        # Reshape back: [B, T, embed_dim]
        x = x.transpose(1, 2).reshape(b, t, -1)

        return self.out_proj(x)
```

---

## KV Cache Management

```python
def complete_kv(cache, current_end, k, v):
    """Add new K, V to cache and return full sequence."""
    current_end = current_end.shape[0]  # Position counter

    # Store new K, V
    cache[0, :, current_end:current_end + k.shape[1]] = k
    cache[1, :, current_end:current_end + v.shape[1]] = v

    # Return valid portion
    valid = cache[:, :, :current_end + k.shape[1]]
    return valid[0], valid[1]


def increment_step(self, state, increment=1):
    """Increment position counter."""
    new_size = state["current_end"].shape[0] + increment
    state["current_end"] = torch.zeros((new_size,))
```

---

## Causal Mask

```python
def _materialize_causal_mask(shape, shift, device="cpu"):
    """Create causal attention mask.

    Args:
        shape: (num_queries, num_keys)
        shift: Offset for streaming (size of cached keys)
    """
    num_queries, num_keys = shape[-2:]
    shift = num_keys - num_queries

    # Lower triangular matrix
    tensor = torch.full(shape, fill_value=1.0)
    mask = torch.tril(tensor, diagonal=shift)

    # Convert to log space (0 → -inf, 1 → 0)
    return torch.log(mask)
```

---

## Key Implementation Notes

### 1. Combined QKV Projection

Q, K, V are computed in a single linear layer:
```python
self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
```

### 2. RoPE Applied After Projection

```python
q, k = self.rope(q, k, offset=offset)
```

The offset accounts for cached positions.

### 3. Cache Pre-filled with NaN

```python
cache=torch.full(..., float("NaN"))
```

This allows detecting unused positions if needed.

### 4. No Bias in Projections

Both `in_proj` and `out_proj` have `bias=False`.

---

## Related Documents

- [RoPE](rope.md) - Rotary position embeddings
- [FlowLM](../ARCHITECTURE/flowlm.md) - How transformer is used
- [State Management](../STREAMING/state-management.md) - StatefulModule pattern
