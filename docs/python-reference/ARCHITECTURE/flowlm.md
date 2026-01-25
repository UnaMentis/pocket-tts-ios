# FlowLM: Flow Language Model

## Source File
`validation/.venv/lib/python3.11/site-packages/pocket_tts/models/flow_lm.py`

---

## Overview

FlowLM is the autoregressive language model that generates latent audio representations from text. It uses:
1. **Text Conditioner**: SentencePiece tokenization + embedding
2. **Streaming Transformer**: 6-layer transformer with KV cache
3. **FlowNet**: MLP that generates latents via flow matching (LSD)

---

## Architecture

```python
class FlowLMModel(nn.Module):
    def __init__(
        self,
        conditioner: LUTConditioner,  # Text tokenization/embedding
        flow_net: SimpleMLPAdaLN,     # Latent generation MLP
        transformer: StreamingTransformer,
        dim: int = 1024,              # Transformer dimension
        ldim: int = 32,               # Latent dimension
        stats_ema_decay: float = 0.999,
        text_padding_weight: float = 1.0,
        dtype=None,
    ):
        self.conditioner = conditioner
        self.flow_net = flow_net
        self.transformer = transformer

        # Latent normalization statistics (learned during training)
        self.register_buffer("emb_std", torch.ones(ldim, dtype=dtype))
        self.register_buffer("emb_mean", torch.zeros(ldim, dtype=dtype))

        # BOS embedding (NaN signals BOS position)
        self.bos_emb = torch.nn.Parameter(torch.randn(ldim, dtype=dtype))

        # Input projection: latent → transformer dimension
        self.input_linear = nn.Linear(ldim, dim, bias=False)

        # Output heads
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.out_eos = nn.Linear(dim, 1)  # EOS prediction
```

---

## Forward Pass

```python
def forward(
    self,
    sequence: torch.Tensor,      # [B, S, ldim] input latents
    text_embeddings: torch.Tensor,  # [B, T_text, dim] text/voice conditioning
    model_state: dict,
    lsd_decode_steps: int,
    temp: float,
    noise_clamp: float | None,
    eos_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate next latent and EOS prediction.

    Args:
        sequence: Input latents. NaN values signal BOS position.
        text_embeddings: Conditioning from text or voice
        model_state: KV cache and position state
        lsd_decode_steps: Steps for LSD decode (typically 10)
        temp: Sampling temperature
        noise_clamp: Max noise value (or None)
        eos_threshold: Threshold for EOS detection

    Returns:
        (latent, is_eos): Generated latent and EOS flag
    """
    # Replace NaN with BOS embedding
    sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)

    # Project to transformer dimension
    input_ = self.input_linear(sequence)  # [B, S, 1024]

    # Run transformer backbone
    transformer_out = self.backbone(input_, text_embeddings, sequence, model_state)
    transformer_out = transformer_out.to(torch.float32)

    # Take only the last position (autoregressive)
    transformer_out = transformer_out[:, -1]  # [B, 1024]

    # EOS prediction
    out_eos = self.out_eos(transformer_out) > eos_threshold

    # Generate latent via FlowNet
    noise_shape = transformer_out.shape[:-1] + (self.ldim,)
    std = temp**0.5
    noise = torch.empty(noise_shape, ...)
    if noise_clamp is None:
        torch.nn.init.normal_(noise, mean=0.0, std=std)
    else:
        torch.nn.init.trunc_normal_(noise, mean=0.0, std=std, a=-noise_clamp, b=noise_clamp)

    conditioned_flow = partial(self.flow_net, transformer_out)
    return lsd_decode(conditioned_flow, noise, lsd_decode_steps), out_eos
```

---

## Backbone (Transformer)

```python
def backbone(
    self, input_, text_embeddings: torch.Tensor, sequence, model_state: dict
) -> torch.Tensor:
    # Concatenate conditioning with input
    input_ = torch.cat([text_embeddings, input_], dim=1)

    # Run transformer
    transformer_out = self.transformer(input_, model_state)

    # Apply output normalization
    if self.out_norm:
        transformer_out = self.out_norm(transformer_out)

    # Remove conditioning prefix from output
    transformer_out = transformer_out[:, -sequence.shape[1]:]

    return transformer_out
```

---

## Transformer Configuration

```python
StreamingTransformer(
    d_model=1024,
    num_heads=16,
    dim_per_head=64,
    num_layers=6,
    causal=True,         # Causal attention for autoregressive
    use_rope=True,       # Rotary position embeddings
    max_seq_len=1000,
)
```

### Attention Details

- **Causal masking**: Each position can only attend to previous positions
- **RoPE**: Interleaved rotary embeddings (not split halves)
- **KV Cache**: Pre-allocated `[2, B, max_len, H, D]` for streaming

---

## Two-Phase Conditioning

Voice and text are added to the KV cache in sequence:

```
Position:  0    ...   N_voice   N_voice+1  ...  N_voice+N_text
Content:   [Voice Embeddings] [    Text Embeddings    ]
```

During generation, the transformer attends to:
1. Voice embeddings (positions 0 to N_voice-1)
2. Text embeddings (positions N_voice to N_voice+N_text-1)
3. Previously generated latents

---

## BOS Handling

The first latent input is marked with NaN:
```python
backbone_input = torch.full((1, 1, ldim), fill_value=float("NaN"))
```

In forward(), NaN is replaced with learned BOS embedding:
```python
sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
```

---

## Latent Normalization

FlowLM outputs normalized latents. Denormalize before Mimi:
```python
# In TTSModel._decode_audio_worker:
mimi_decoding_input = latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
```

The `emb_mean` and `emb_std` are learned during training and stored in the model weights.

---

## State Management

FlowLM uses StatefulModule for attention:
```python
model_state = init_states(flow_lm, batch_size=1, sequence_length=1000)

# State structure:
{
    "transformer.layers.0.self_attn": {
        "cache": [2, 1, 1000, 16, 64],  # KV cache
        "current_end": [0]              # Position counter
    },
    # ... for each of 6 layers
}
```

After each forward pass:
```python
increment_steps(flow_lm, model_state, increment=num_new_tokens)
```

---

## Key Implementation Notes

### 1. dtype Handling

```python
# Explicit float32 conversion after transformer
transformer_out = transformer_out.to(torch.float32)
```

This ensures FlowNet receives float32 regardless of transformer dtype.

### 2. Input Linear Has No Bias

```python
self.input_linear = nn.Linear(ldim, dim, bias=False)
```

### 3. Output Norm is Standard LayerNorm

```python
self.out_norm = nn.LayerNorm(dim, eps=1e-5)  # With affine params
```

### 4. EOS Threshold

Default is 0.5. The EOS head outputs a scalar, compared against threshold:
```python
out_eos = self.out_eos(transformer_out) > eos_threshold
```

---

## Related Documents

- [FlowNet/LSD](flownet-lsd.md) - Latent generation via flow matching
- [Transformer](../MODULES/transformer.md) - Attention implementation
- [RoPE](../MODULES/rope.md) - Rotary position embeddings
- [Overview](overview.md) - Complete pipeline context
