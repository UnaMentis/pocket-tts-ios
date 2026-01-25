# Pocket TTS Architecture Overview

## Source Files
- `validation/.venv/lib/python3.11/site-packages/pocket_tts/models/tts_model.py`
- `validation/.venv/lib/python3.11/site-packages/pocket_tts/models/flow_lm.py`
- `validation/.venv/lib/python3.11/site-packages/pocket_tts/models/mimi.py`

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TTSModel                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌───────────────────────────────────────────────┐      │
│  │    Text      │    │                   FlowLM                       │      │
│  │   Input      │───▶│  ┌─────────────┐  ┌────────────┐  ┌─────────┐ │      │
│  │ "Hello..."   │    │  │ Tokenizer   │──▶│ Transformer│──▶│ FlowNet │ │      │
│  └──────────────┘    │  │(SentencePiece)│  │ (6 layers) │  │  (MLP)  │ │      │
│                      │  └─────────────┘  └────────────┘  └────┬────┘ │      │
│  ┌──────────────┐    │                                        │       │      │
│  │    Voice     │    │  Voice embeddings added to KV cache    │       │      │
│  │   Prompt     │───▶│  at positions 0..N (two-phase)         │       │      │
│  └──────────────┘    └────────────────────────────────────────┼───────┘      │
│                                                                │              │
│                                           Latents [B, T, 32]   │              │
│                                                                ▼              │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                              Mimi Decoder                              │   │
│  │  ┌─────────────┐   ┌────────────────┐   ┌─────────────────────────┐   │   │
│  │  │  Upsample   │──▶│ Decoder        │──▶│      SEANet Decoder     │   │   │
│  │  │  (16x)      │   │ Transformer    │   │  (ConvTranspose layers) │   │   │
│  │  └─────────────┘   │ (2 layers)     │   └────────────┬────────────┘   │   │
│  │                    └────────────────┘                 │                │   │
│  └───────────────────────────────────────────────────────┼────────────────┘   │
│                                                          │                    │
│                                     Audio [B, 1, samples]│                    │
│                                                          ▼                    │
│                                               ┌──────────────────┐            │
│                                               │   WAV Output     │            │
│                                               │   24kHz mono     │            │
│                                               └──────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### TTSModel (Orchestrator)

| Attribute | Value |
|-----------|-------|
| Sample Rate | 24,000 Hz |
| Frame Rate | 12.5 Hz (80ms per frame) |
| Latent Dimension | 32 |
| Default Temperature | 0.7 |
| LSD Decode Steps | 10 |
| EOS Threshold | 0.5 |

### FlowLM (Flow Language Model)

| Component | Specification |
|-----------|---------------|
| Embedding Dim | 1024 |
| Latent Dim | 32 |
| Transformer Layers | 6 |
| Attention Heads | 16 |
| Dim per Head | 64 |
| Max Sequence Length | 1000 |
| Tokenizer | SentencePiece |
| Vocab Size | ~32,000 |

### Mimi Decoder

| Component | Specification |
|-----------|---------------|
| Input Latent Dim | 32 |
| Internal Dim | 512 |
| Upsample Ratio | 16x (12.5 Hz → 200 Hz) |
| SEANet Ratios | [8, 5, 4, 2] → 320x (200 Hz → 24 kHz) |
| Total Upsample | 16 × 320 = 5120x (1 latent → 1920 samples) |
| Decoder Transformer | 2 layers |

---

## Generation Pipeline

### Phase 1: Voice Conditioning (Once per voice)

```python
# 1. Load or encode voice prompt
if predefined_voice:
    prompt = load_predefined_voice("alba")  # Pre-computed embeddings
else:
    audio = audio_read("voice.wav")
    audio = convert_audio(audio, orig_sr, 24000, channels=1)
    prompt = mimi.encode_to_latent(audio)  # [B, T_voice, 512]
    prompt = linear(prompt, speaker_proj_weight)  # [B, T_voice, 1024]

# 2. Initialize FlowLM state with voice embeddings
model_state = init_states(flow_lm, batch_size=1, sequence_length=1000)

# 3. Run FlowLM with voice conditioning (populates KV cache at positions 0..T_voice)
flow_lm._sample_next_latent(
    sequence=empty,  # No latents yet
    text_embeddings=prompt,  # Voice goes here
    model_state=model_state,
)
increment_steps(flow_lm, model_state, increment=T_voice)
```

### Phase 2: Text Prompting (Once per text)

```python
# 1. Tokenize text
tokens = conditioner.tokenize(text)  # e.g., "Hello world" → [1, 5, 7, 23, ...]

# 2. Embed and run through FlowLM (populates KV cache at positions T_voice..T_voice+T_text)
text_embeddings = conditioner(tokens)  # [B, T_text, 1024]
flow_lm._sample_next_latent(
    sequence=empty,
    text_embeddings=text_embeddings,
    model_state=model_state,
)
increment_steps(flow_lm, model_state, increment=T_text)
```

### Phase 3: Autoregressive Generation (Loop)

```python
# Initialize with BOS (NaN signals BOS position)
backbone_input = torch.full((1, 1, 32), fill_value=float("NaN"))

for step in range(max_gen_len):
    # 3a. Generate next latent via FlowLM
    next_latent, is_eos = flow_lm._sample_next_latent(
        sequence=backbone_input,  # [B, 1, 32]
        text_embeddings=empty,    # Already in KV cache
        model_state=model_state,
    )
    increment_steps(flow_lm, model_state, increment=1)

    # 3b. Check for end-of-sequence
    if is_eos and step >= eos_step + frames_after_eos:
        break

    # 3c. Queue latent for decoding
    latents_queue.put(next_latent)

    # 3d. Use this latent as input for next step
    backbone_input = next_latent
```

### Phase 4: Audio Decoding (Parallel Thread)

```python
mimi_state = init_states(mimi, batch_size=1, sequence_length=1000)

while True:
    latent = latents_queue.get()  # [B, 1, 32]
    if latent is None:
        break

    # Denormalize latent
    latent = latent * emb_std + emb_mean

    # Decode to audio
    audio_frame = mimi.decode_from_latent(latent, mimi_state)  # [B, 1, 1920]
    increment_steps(mimi, mimi_state, increment=16)  # 16 for upsample

    yield audio_frame
```

---

## Threading Model

```
Main Thread                    Decoder Thread
─────────────                  ──────────────
generate_audio_stream()
    │
    ├─► Start decoder thread ─────────────────►  _decode_audio_worker()
    │                                                   │
    ├─► _autoregressive_generation()                    │
    │       │                                           │
    │       ├─► Generate latent 1                       │
    │       │       │                                   │
    │       │       └─► latents_queue.put()  ──────►   get() → decode
    │       │                                           │
    │       ├─► Generate latent 2                       │
    │       │       │                                   │
    │       │       └─► latents_queue.put()  ──────►   get() → decode
    │       │                                           │
    │       │                           ◄────────────  result_queue.put(chunk)
    │       │                                           │
    │       └─► latents_queue.put(None)  ──────►       exit loop
    │                                                   │
    └─► result_queue.get() ◄───────────────────────────┘
            │
            └─► yield audio_chunk
```

---

## Frame Timing

| Event | Time | Description |
|-------|------|-------------|
| Voice prompt | Variable | Typically 3-10 seconds of audio → 38-125 frames |
| Text tokens | Instant | ~10ms to tokenize and embed |
| First latent | ~50ms | First FlowLM autoregressive step |
| Each latent | ~50ms | Subsequent FlowLM steps |
| First audio | ~200ms | Mimi decode + streaming startup |
| Total latency | ~200ms | Time to first audio chunk |

### Audio Output Timing

- 1 latent = 80ms of audio (1920 samples @ 24kHz)
- FlowLM generates at ~20 latents/second
- Mimi decodes at ~6x realtime
- Overall: ~3-4x realtime on M4 Mac

---

## State Management Summary

| Component | State Contents | Purpose |
|-----------|----------------|---------|
| FlowLM Transformer | KV cache `[2, B, T, H, D]` | Avoid recomputing attention |
| FlowLM Transformer | `current_end` position | Track sequence position |
| Mimi Upsample | `partial` buffer | Overlap-add for ConvTranspose |
| Mimi SEANet | `previous` buffers | Causal context for Conv1d |
| Mimi SEANet | `partial` buffers | Overlap-add for ConvTranspose |

All states are initialized once and updated in-place during generation.

---

## Related Documents

- [SEANet Decoder](seanet-decoder.md) - Detailed SEANet architecture
- [Mimi Decode](mimi-decode.md) - Mimi decoder flow
- [FlowLM](flowlm.md) - FlowLM transformer
- [FlowNet/LSD](flownet-lsd.md) - LSD decode algorithm
- [State Management](../STREAMING/state-management.md) - StatefulModule pattern
- [Overlap-Add](../STREAMING/conv-transpose-overlap-add.md) - StreamingConvTranspose1d
