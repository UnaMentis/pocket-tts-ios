# Pocket TTS Rust Port — Knowledge Index

Distilled from PORTING_STATUS.md (1,491 lines of session logs) into a compact reference.
This is the "institutional memory" — what works, what doesn't, and why.

Last distilled: 2026-03-10

## Current State

**Status**: Production ready. All components functional. 91 unit tests passing.

| Component | Status | Key Detail |
|-----------|--------|------------|
| Tokenizer | Match | SentencePiece, matches Python exactly |
| FlowLM (6-layer transformer) | Match | ~70M params, correct architecture |
| FlowNet (consistency sampling) | Working | Random noise enabled (production mode) |
| Mimi Decoder | Working | Full streaming with replicate padding |
| SEANet | Working | Streaming Conv1d + ConvTranspose1d |
| EOS Detection | Working | Threshold -4.0, natural detection |
| Audio Output | Working | Int16 PCM WAV, 24kHz, healthy amplitudes |

## Critical Lessons Learned

### 1. FlowNet Noise Is the Dominant Source of Divergence
- **Discovery**: Rust used zeros (DEBUG mode) while Python uses `normal_(mean=0, std=sqrt(temp))`
- **Impact**: This single difference accounts for most latent divergence between Rust and Python
- **Resolution**: Enabled random noise in Rust. Latents now naturally differ between implementations — this is expected and correct
- **Implication**: Exact latent matching is impossible with random noise. Quality must be judged by audio intelligibility, not waveform correlation

### 2. Batch vs Streaming Produces Fundamentally Different Audio
- **Discovery**: Python streaming correlation with Python batch is ~-0.04 (essentially uncorrelated)
- **Impact**: Don't expect high Rust-vs-Python waveform correlation
- **Resolution**: Compare quality metrics (WER, MCD, SNR) not raw waveform correlation

### 3. SEANet Requires Full Streaming for ALL Layers
- **Bug**: Forward method used streaming for ConvTranspose1d but batch mode for Conv1d
- **Fix**: All convolutions (input_conv, residual blocks, output_conv) must use streaming
- **Impact**: Correlation jumped from 0.13 to 0.69

### 4. Decoder Transformer Uses Non-Causal (Full) Attention
- **Bug**: Rust applied causal mask; Python uses full self-attention in decoder
- **Fix**: Removed causal mask from decoder transformer streaming
- **Impact**: Part of the 0.13 → 0.69 correlation improvement

### 5. First-Frame Replicate Padding Matters
- **Bug**: Rust used zero padding for initial Conv1d context; Python uses `pad_mode="replicate"`
- **Fix**: Added `PadMode::Replicate` — fills initial context with first sample repeated
- **Impact**: Correlation improved from 0.69 to 0.81, frame 0 correlation from 0.13 to 0.49

### 6. iOS Requires Int16 PCM WAV
- **Bug**: Float32 WAV files produce noise/spikes in AVAudioPlayer
- **Fix**: Output Int16 PCM (16-bit signed), `bits_per_sample: 16`, `sample_format: SampleFormat::Int`
- **Location**: `src/audio.rs`

### 7. EOS Detection Diverges on Long Sequences
- **Observation**: For medium+ phrases, Rust detects EOS earlier than Python (e.g., step 145 vs 166)
- **Root cause**: Numerical precision accumulation over many transformer steps + different random noise
- **Impact**: Minimal — audio is still complete and intelligible
- **Workaround**: Could adjust -4.0 threshold, but not necessary for production quality

### 8. Max Audio Length ~40 Seconds
- **Constraint**: Mimi decoder max sequence length = 8192 frames
- **Calculation**: 8192 ÷ 16 (upsample) = 512 latents × 80ms = ~40 seconds
- **Workaround**: Chunk at sentence boundaries for longer content

## Architecture Decisions (Why Things Are the Way They Are)

### Streaming Implementation
Based on Kyutai's official Moshi Rust implementation:
- `StreamableConv1d`: Left-padding on first frame only, context buffer concatenation for subsequent frames
- `StreamableConvTranspose1d`: Overlap-add with partial buffer persistence
- KV cache for decoder transformer (non-causal attention, accumulates across frames)

### Audio Pipeline
```
Text → SentencePiece → FlowLM (transformer, 6 layers)
     → FlowNet (consistency sampling, temp-scaled noise)
     → Mimi Decoder:
         → output_proj → upsample (ConvTranspose1d, 16x)
         → decoder_transformer (non-causal, KV cache)
         → SEANet (streaming Conv1d, 4 decoder blocks, ConvTranspose1d upsampling)
     → Int16 PCM WAV @ 24kHz
```

### Key Parameters
| Parameter | Default | Notes |
|-----------|---------|-------|
| temperature | 0.7 | Controls FlowNet noise std (sqrt(temp)) |
| top_p | 0.9 | Nucleus sampling for token selection |
| consistency_steps | 2 | FlowNet denoising iterations |
| speed | 1.0 | Playback rate |
| EOS threshold | -4.0 | Logit threshold for end-of-speech detection |
| frames_after_eos | min(5, ceil(num_text_tokens/4)) | Post-EOS padding for natural endings |

## Things That Were Tried and Failed

### Amplitude Scaling (Removed)
- Added 5x amplitude scaling to compensate for low output
- **Root cause**: Batch mode SEANet was the real problem, not amplitude
- **Lesson**: Fix the underlying bug, don't paper over it with scaling

### Crossfade at Chunk Boundaries (Removed)
- Applied crossfade between streaming audio chunks
- **Problem**: Blended DIFFERENT audio content together (end of chunk A + start of chunk B)
- **Lesson**: Mimi's streaming ConvTranspose1d state already handles continuity

### min_gen_steps = 40 (Removed)
- Forced minimum generation length, bypassing natural EOS
- **Problem**: Prevented short phrases from ending naturally
- **Fix**: Changed to min_gen_steps = 0

### Deterministic Comparison with Zeros
- Used `Tensor::zeros` instead of `Tensor::randn` in FlowNet for reproducible comparison
- **Problem**: Doesn't match production behavior, masks real quality issues
- **Lesson**: Use production noise, judge by audio quality not latent correlation

## File Reference

| File | What It Does | When to Modify |
|------|-------------|----------------|
| `src/models/flowlm.rs` | 6-layer transformer, EOS detection | EOS tuning, generation behavior |
| `src/modules/flownet.rs` | Consistency sampling, noise schedule | Temperature/noise behavior |
| `src/models/mimi.rs` | Mimi decoder, streaming state | Decoder quality, streaming bugs |
| `src/models/seanet.rs` | SEANet upsampling blocks | Audio quality, upsampling |
| `src/modules/conv.rs` | Streaming Conv1d/ConvTranspose1d | Padding, causality |
| `src/modules/attention.rs` | Multi-head attention, KV cache | Attention quality |
| `src/audio.rs` | WAV encoding | Format issues |
| `src/engine.rs` | UniFFI interface | API changes |

## Validation Infrastructure

| Tool | Purpose |
|------|---------|
| `validation/quality_metrics.py` | WER, MCD, SNR, THD measurement |
| `validation/baseline_tracker.py` | Baseline comparison |
| `autotuning/scorer.py` | Composite quality score |
| `autotuning/memory.py` | Structured experiment memory |
| `autotuning/autotune.py` | Automated tuning loop |
| `autotuning/program.md` | AI agent instructions for autonomous tuning |
