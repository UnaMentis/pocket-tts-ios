# Kyutai Moshi Streaming Implementation Analysis

This document analyzes Kyutai's official Rust Mimi implementation from the [moshi repository](https://github.com/kyutai-labs/moshi) to understand the correct streaming patterns.

## Overview

The Moshi repo contains a production-quality Rust implementation of Mimi that properly handles streaming audio synthesis. The key insight is that **all layers use streaming step() methods** with persistent state, rather than batch processing followed by streaming.

## Core Streaming Abstractions

### StreamTensor (streaming.rs)

A wrapper around `Option<Tensor>` that elegantly handles empty states:

```rust
pub struct StreamTensor(Option<Tensor>);

impl StreamTensor {
    pub fn empty() -> Self { Self(None) }
    pub fn from_tensor(tensor: Tensor) -> Self { Self(Some(tensor)) }

    // Concatenate two StreamTensors along a dimension
    pub fn cat2<D: Dim>(&self, rhs: &Self, dim: D) -> Result<Self>

    // Split into (first n elements, remainder)
    pub fn split<D: Dim>(&self, dim: D, lhs_len: usize) -> Result<(Self, Self)>

    // Apply a Module to the inner tensor
    pub fn apply<M: candle::Module>(&self, m: &M) -> Result<Self>
}
```

### StreamingModule Trait (streaming.rs)

```rust
pub trait StreamingModule {
    fn step(&mut self, xs: &StreamTensor, mask: &StreamMask) -> Result<StreamTensor>;
    fn reset_state(&mut self);
}
```

Every streaming layer implements this trait, allowing uniform composition.

## Streaming Convolution Implementations

### StreamableConv1d::step() (conv.rs:312-370)

Key implementation details:

1. **First-frame left padding**: Uses a `left_pad_applied` flag to add causal padding only on the first frame
2. **Buffer concatenation**: Concatenates previous buffer with current input
3. **Frame calculation**: Computes how many full convolution outputs can be produced
4. **State storage**: Stores remaining samples for next frame

```rust
fn step(&mut self, xs: &StreamTensor, mask: &StreamMask) -> Result<StreamTensor> {
    let xs = match xs.as_option() {
        None => return Ok(().into()),
        Some(xs) => xs.clone(),
    };

    // Apply left pad only on first frame
    let xs = if self.left_pad_applied {
        xs
    } else {
        self.left_pad_applied = true;
        let padding_total = kernel - stride;
        pad1d(&xs, padding_total, 0, self.pad_mode)?
    };

    // Concatenate with previous buffer
    let xs = StreamTensor::cat2(&self.state_prev_xs, &xs.into(), D::Minus1)?;
    let seq_len = xs.seq_len(D::Minus1)?;

    // Calculate how many frames we can produce
    let num_frames = (seq_len + stride).saturating_sub(kernel) / stride;

    if num_frames > 0 {
        let offset = num_frames * stride;
        // Store remaining for next call
        let state_prev_xs = xs.narrow(D::Minus1, offset, seq_len - offset)?;
        // Process current frames
        let in_l = (num_frames - 1) * stride + kernel;
        let xs = xs.narrow(D::Minus1, 0, in_l)?;
        let ys = xs.apply(&self.conv.conv)?;  // Direct conv, no padding

        self.state_prev_xs = state_prev_xs;
        (state_prev_xs, ys)
    } else {
        self.state_prev_xs = xs;
        (xs, StreamTensor::empty())
    }
}
```

### StreamableConvTranspose1d::step() (conv.rs:448-501)

The overlap-add implementation with correct bias handling:

```rust
fn step(&mut self, xs: &StreamTensor, mask: &StreamMask) -> Result<StreamTensor> {
    let xs = match xs.as_option() {
        Some(xs) => xs,
        None => return Ok(StreamTensor::empty()),
    };

    // Apply convtr directly (no padding)
    let ys = self.convtr.forward(xs)?;
    let ot = ys.dim(D::Minus1)?;

    // Add previous state to head of output
    let ys = match self.state_prev_ys.as_option() {
        None => ys,
        Some(prev_ys) => {
            let pt = prev_ys.dim(D::Minus1)?;

            // CRITICAL: Subtract bias before adding to avoid double-counting
            let prev_ys = match &self.convtr.bs {
                None => prev_ys.clone(),
                Some(bias) => {
                    let bias = bias.reshape((1, (), 1))?;
                    prev_ys.broadcast_sub(&bias)?
                }
            };

            // Overlap-add: add prev_ys to head of current output
            let ys1 = (ys.narrow(D::Minus1, 0, pt)? + prev_ys)?;
            let ys2 = ys.narrow(D::Minus1, pt, ot - pt)?;
            Tensor::cat(&[ys1, ys2], D::Minus1)?
        }
    };

    // Split into valid output and new state
    let invalid_steps = self.kernel_size - stride;  // overlap length
    let (ys, prev_ys) = StreamTensor::from(ys).split(D::Minus1, ot - invalid_steps)?;

    self.state_prev_ys = prev_ys;
    Ok(ys)
}
```

## SEANet Decoder Streaming (seanet.rs)

The SEANetDecoder implements StreamingModule:

```rust
impl StreamingModule for SeaNetDecoder {
    fn step(&mut self, xs: &StreamTensor, m: &StreamMask) -> Result<StreamTensor> {
        // Initial conv (streaming)
        let mut xs = self.init_conv1d.step(xs, m)?;

        // Each upsample stage
        for layer in self.layers.iter_mut() {
            // ELU activation then streaming ConvTranspose
            xs = layer.upsample.step(&xs.apply(&self.activation)?, m)?;

            // ResBlocks (all streaming)
            for residual in layer.residuals.iter_mut() {
                xs = residual.step(&xs, m)?;
            }
        }

        // Final conv (streaming)
        let xs = self.final_conv1d.step(&xs.apply(&self.activation)?, m)?;

        // Optional final activation
        match self.final_activation.as_ref() {
            None => xs,
            Some(act) => xs.apply(act)?,
        }
    }
}
```

## Mimi Decoder Streaming (mimi.rs)

The complete decode_step shows the correct pattern:

```rust
pub fn decode_step(&mut self, codes: &StreamTensor, m: &StreamMask) -> Result<StreamTensor> {
    // Step 1: Decode codes (stateless)
    let emb = match codes.as_option() {
        Some(codes) => StreamTensor::from_tensor(self.quantizer.decode(codes)?),
        None => StreamTensor::empty(),
    };

    // Step 2: Upsample (streaming ConvTranspose)
    let emb = self.upsample.step(&emb, m)?;

    // Step 3: Decoder transformer (streaming with KV cache)
    let out = self.decoder_transformer.step(&emb, m)?;

    // Step 4: SEANet decoder (streaming)
    self.decoder.step(&out, m)
}
```

## Key Differences from Current Implementation

| Aspect | Current (Wrong) | Kyutai (Correct) |
|--------|-----------------|------------------|
| Processing | Batch all latents, then partial streaming | Step-by-step for each latent |
| State | Re-initialized per batch | Persists across all steps |
| Conv1d | Uses symmetric padding | Uses causal left-pad on first frame only |
| ConvTranspose1d | Has overlap-add but wrong calling pattern | Correct overlap-add with bias subtraction |
| Composition | Layers called with full tensors | Layers called with step() and StreamTensor |

## Required Changes to pocket-tts

1. **Introduce StreamTensor abstraction** - Wrap Option<Tensor> for cleaner handling

2. **Implement StreamingModule trait** - Standardize streaming interface

3. **Refactor MimiDecoder::forward_streaming()** to be a loop:
   ```rust
   pub fn forward_streaming(&mut self, latents: &Tensor) -> Result<Tensor> {
       let seq_len = latents.dim(1)?;
       let mut audio_chunks = Vec::new();

       for i in 0..seq_len {
           let latent = latents.narrow(1, i, 1)?;  // Single frame
           let codes = StreamTensor::from_tensor(self.output_proj.forward(&latent)?);
           let upsampled = self.upsample.step(&codes)?;
           let transformed = self.decoder_transformer.step(&upsampled)?;
           let audio = self.seanet.step(&transformed)?;

           if let Some(chunk) = audio.as_option() {
               audio_chunks.push(chunk.clone());
           }
       }

       Tensor::cat(&audio_chunks, D::Minus1)
   }
   ```

4. **Fix SEANet to use step() throughout** - All Conv1d and ResBlocks must use streaming

5. **Fix Conv1d first-frame handling** - Left-pad only on first frame, not all frames

## Verification Strategy

1. Process one latent through Python streaming, capture all intermediate values
2. Process same latent through Rust streaming
3. Compare at each step:
   - After output_proj
   - After upsample
   - After decoder_transformer
   - After each SEANet stage

Target: >0.99 correlation at each step should yield >0.95 final waveform correlation.
