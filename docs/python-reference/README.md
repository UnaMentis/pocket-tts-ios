# Python Reference Documentation

This directory contains comprehensive documentation extracted from the Pocket TTS Python source code to support the Rust/Candle port.

## Source Location

All documentation is based on the official Kyutai Pocket TTS implementation:
```
validation/.venv/lib/python3.11/site-packages/pocket_tts/
```

## Quick Navigation

### Critical for Fixing Audio Issue

| Document | Description |
|----------|-------------|
| [Overlap-Add](STREAMING/conv-transpose-overlap-add.md) | **ROOT CAUSE** - StreamingConvTranspose1d algorithm |
| [Conv1d Streaming](STREAMING/conv1d-streaming.md) | Causal convolution with context buffer |
| [State Management](STREAMING/state-management.md) | StatefulModule pattern for streaming |

### Architecture Reference

| Document | Description |
|----------|-------------|
| [Overview](ARCHITECTURE/overview.md) | Complete data flow and pipeline |
| [SEANet Decoder](ARCHITECTURE/seanet-decoder.md) | Layer-by-layer decoder structure |
| [Mimi Decode](ARCHITECTURE/mimi-decode.md) | Latent to audio conversion |
| [FlowLM](ARCHITECTURE/flowlm.md) | Transformer language model |
| [FlowNet/LSD](ARCHITECTURE/flownet-lsd.md) | Latent generation via flow matching |

### Module Reference

| Document | Description |
|----------|-------------|
| [Transformer](MODULES/transformer.md) | Attention with KV cache |
| [RoPE](MODULES/rope.md) | Rotary position embeddings |
| [MLP](MODULES/mlp.md) | Feed-forward and AdaLN |
| [LayerNorm](MODULES/layer-norm.md) | Normalization layers |

### Gap Analysis

| Document | Description |
|----------|-------------|
| [Gap Analysis](gap-analysis.md) | Questions for Kyutai outreach |

## Current Port Status

| Component | Status | Notes |
|-----------|--------|-------|
| Tokenizer | ✅ Working | SentencePiece, exact match |
| FlowLM Transformer | ✅ Working | All 6 layers verified |
| FlowNet (MLP) | ✅ Working | LSD decode correct |
| Latent Generation | ✅ Working | Cosine similarity = 1.0 |
| Mimi Decoder | ⚠️ Partial | Streaming not implemented |
| Audio Output | ❌ Broken | 0.013 correlation (target: >0.95) |

**Root Cause:** The Rust implementation processes all latents in batch mode, while Python processes frame-by-frame with streaming state. This causes incorrect overlap-add accumulation.

## Key Insight

The Python implementation maintains state across frames for:

1. **StreamingConv1d**: Previous samples for causal context
2. **StreamingConvTranspose1d**: Partial outputs for overlap-add

Without proper streaming, the Rust output has ~5-6x lower amplitude because the overlap regions don't accumulate correctly.

## How to Use This Documentation

### For Fixing the Audio Issue

1. Read [Overlap-Add](STREAMING/conv-transpose-overlap-add.md) first
2. Understand the state buffer structure
3. Implement frame-by-frame processing in Rust
4. Verify against Python reference outputs

### For Understanding Architecture

1. Start with [Overview](ARCHITECTURE/overview.md)
2. Follow data flow through each component
3. Refer to module docs for implementation details

### For Debugging

1. Use `validation/dump_intermediates.py` to capture Python values
2. Compare against Rust at each stage
3. Check state buffer contents match

## External Resources

| Resource | URL |
|----------|-----|
| Pocket TTS GitHub | https://github.com/kyutai-labs/pocket-tts |
| HuggingFace Model | https://huggingface.co/kyutai/pocket-tts |
| CALM Paper | https://arxiv.org/abs/2509.06926 |
| Moshi/Mimi | https://github.com/kyutai-labs/moshi |

## Document Conventions

- Python code is copied directly from source with annotations
- Tensor shapes are shown as `[B, C, T]` (batch, channels, time)
- State buffers show exact dimensions for batch_size=1
- Implementation notes highlight Rust-specific considerations
