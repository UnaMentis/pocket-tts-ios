//! Mimi VAE Decoder
//!
//! Neural audio codec decoder that converts quantized latents
//! to high-quality 24kHz audio.
//!
//! Portions of this file derived from:
//! <https://github.com/babybirdprd/pocket-tts>
//! Licensed under MIT

// Allow dead code - streaming methods will be used in future implementation
#![allow(dead_code)]

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::modules::layer_norm::LayerNorm;
use crate::modules::streaming::{StreamTensor, StreamingModule};

/// State for streaming Conv1d (causal context buffer)
#[derive(Debug, Clone)]
pub struct StreamingConv1dState {
    /// Previous input samples: [batch, in_channels, kernel - stride]
    /// Keeps causal context from previous frames
    pub previous: Tensor,
    /// Whether this is the first frame (for replicate padding)
    pub is_first: bool,
}

impl StreamingConv1dState {
    /// Create new state with zero buffer
    pub fn new(batch_size: usize, in_channels: usize, context_len: usize, device: &Device) -> Result<Self> {
        let previous = Tensor::zeros((batch_size, in_channels, context_len), DType::F32, device)?;
        Ok(Self {
            previous,
            is_first: true,
        })
    }
}

/// State for streaming ConvTranspose1d (overlap-add buffer)
#[derive(Debug, Clone)]
pub struct StreamingConvTr1dState {
    /// Partial output buffer: [batch, out_channels, kernel - stride]
    /// Accumulates overlapping output contributions between frames
    pub partial: Tensor,
}

impl StreamingConvTr1dState {
    /// Create new state with zero buffer
    pub fn new(batch_size: usize, out_channels: usize, overlap: usize, device: &Device) -> Result<Self> {
        let partial = Tensor::zeros((batch_size, out_channels, overlap), DType::F32, device)?;
        Ok(Self { partial })
    }
}

/// State for SEANet ResidualBlock streaming
#[derive(Debug)]
pub struct StreamingResBlockState {
    /// Conv1 (k=3) context: 2 samples
    pub conv1_state: StreamingConv1dState,
    // Conv2 (k=1) doesn't need streaming state
}

/// State for the full SEANet decoder streaming
#[derive(Debug)]
pub struct StreamingSEANetState {
    /// Input conv (k=7) context: 6 samples
    pub input_conv_state: StreamingConv1dState,
    /// ConvTranspose states
    pub convtr_states: [StreamingConvTr1dState; 3],
    /// ResBlock states (one per upsample block)
    pub resblock_states: [StreamingResBlockState; 3],
    /// Output conv (k=3) context: 2 samples
    pub output_conv_state: StreamingConv1dState,
}

/// State for the full Mimi decoder streaming
#[derive(Debug)]
pub struct StreamingMimiState {
    /// State for depthwise 16x upsampler: overlap = 32 - 16 = 16
    pub upsample_state: StreamingConvTr1dState,
    /// State for SEANet decoder
    pub seanet_state: StreamingSEANetState,
}

/// Mimi decoder configuration
#[derive(Debug, Clone)]
pub struct MimiConfig {
    pub latent_dim: usize,
    pub mimi_dim: usize,
    pub sample_rate: usize,
    pub frame_rate: f32,
    pub num_transformer_layers: usize,
}

impl Default for MimiConfig {
    fn default() -> Self {
        Self {
            latent_dim: 32,
            mimi_dim: 512,
            sample_rate: 24000,
            frame_rate: 12.5,
            num_transformer_layers: 2,
        }
    }
}

/// Conv1d layer for the decoder
#[derive(Debug)]
struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl Conv1d {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
        let bias = vb.get(out_channels, "bias").ok();
        Ok(Self {
            weight,
            bias,
            kernel_size,
            stride: 1,
            padding: (kernel_size - 1) / 2,
        })
    }

    fn new_no_bias(in_channels: usize, out_channels: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
        Ok(Self {
            weight,
            bias: None,
            kernel_size,
            stride: 1,
            padding: (kernel_size - 1) / 2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv1d(&self.weight, self.padding, self.stride, 1, 1)?;
        if let Some(bias) = &self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            x.broadcast_add(&bias)
        } else {
            Ok(x)
        }
    }

    /// Streaming forward with causal context
    ///
    /// Replicates Python's StreamingConv1d:
    /// 1. On first frame with replicate mode, fill previous buffer with first sample
    /// 2. Concatenate previous buffer with current input
    /// 3. Run the standard conv with no padding
    /// 4. Store trailing (kernel - stride) samples as new previous buffer
    fn forward_streaming(&self, x: &Tensor, state: &mut StreamingConv1dState) -> Result<Tensor> {
        // Context length = kernel - stride (for stride=1, this is kernel - 1)
        let context_len = self.kernel_size - self.stride;

        // On first frame, use REPLICATE padding (first sample repeated)
        // This matches Python's StreamingConv1d with pad_mode="replicate"
        if state.is_first && context_len > 0 {
            // Get first sample: [batch, channels, 1]
            let first_sample = x.narrow(2, 0, 1)?;
            // Repeat to fill context: [batch, channels, context_len]
            state.previous = first_sample.repeat(&[1, 1, context_len])?;
            state.is_first = false;
        }

        // Concatenate previous context with current input
        let x_with_context = Tensor::cat(&[&state.previous, x], 2)?;

        // Run conv with NO padding (context provides the causal padding)
        let y = x_with_context.conv1d(&self.weight, 0, self.stride, 1, 1)?;

        // Add bias if present
        let y = if let Some(bias) = &self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            y.broadcast_add(&bias)?
        } else {
            y
        };

        // Store trailing samples as new previous buffer
        let in_len = x_with_context.dim(2)?;
        if in_len >= context_len {
            state.previous = x_with_context.narrow(2, in_len - context_len, context_len)?;
        }

        Ok(y)
    }

    /// Get context length for streaming (kernel - stride)
    fn context_len(&self) -> usize {
        self.kernel_size - self.stride
    }

    /// Get input channels
    fn in_channels(&self) -> Result<usize> {
        Ok(self.weight.dim(1)?)
    }
}

/// ConvTranspose1d for upsampling
#[derive(Debug)]
struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: usize,
    stride: usize,
    groups: usize,
}

impl ConvTranspose1d {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        // ConvTranspose weight shape is [in_channels, out_channels, kernel]
        let weight = vb.get((in_channels, out_channels, kernel_size), "weight")?;
        let bias = vb.get(out_channels, "bias").ok();

        Ok(Self {
            weight,
            bias,
            kernel_size,
            stride,
            groups: 1,
        })
    }

    /// Create depthwise ConvTranspose1d (groups = channels)
    /// Used for temporal upsampling where each channel is processed independently
    /// Weight shape: [channels, 1, kernel_size]
    fn new_depthwise(channels: usize, kernel_size: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        // Depthwise: weight shape is [channels, 1, kernel_size]
        let weight = vb.get((channels, 1, kernel_size), "weight")?;
        // No bias for depthwise upsample in this model

        Ok(Self {
            weight,
            bias: None,
            kernel_size,
            stride,
            groups: channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // For batch processing, use padding that maintains output length close to input * stride
        // Standard approach: padding = (kernel_size - stride) / 2 to center the kernel
        let padding = (self.kernel_size - self.stride) / 2;
        let output_padding = (self.kernel_size - self.stride) % 2;

        let y = x.conv_transpose1d(
            &self.weight,
            padding,
            output_padding,
            self.stride,
            1, // dilation
            self.groups,
        )?;

        if let Some(bias) = &self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            y.broadcast_add(&bias)
        } else {
            Ok(y)
        }
    }

    /// Forward pass with no padding - matches Python's non-streaming behavior
    ///
    /// This produces the same output as Python's ConvTranspose1d with padding=0.
    /// Output length = (input_length - 1) * stride + kernel_size
    /// The caller should trim the last `kernel_size - stride` samples if needed
    /// to match Python's streaming output shape.
    fn forward_no_padding(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose1d(
            &self.weight,
            0, // no padding
            0, // no output_padding
            self.stride,
            1, // dilation
            self.groups,
        )?;

        if let Some(bias) = &self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            y.broadcast_add(&bias)
        } else {
            Ok(y)
        }
    }

    /// Streaming forward with overlap-add state accumulation
    ///
    /// This replicates Python's StreamingConvTranspose1d behavior:
    /// 1. Run conv_transpose with NO padding (get full overlap output)
    /// 2. Add previous partial buffer to left edge of output
    /// 3. Store right edge as new partial buffer
    /// 4. Return all but rightmost (kernel - stride) samples
    fn forward_streaming(&self, x: &Tensor, state: &mut StreamingConvTr1dState) -> Result<Tensor> {
        let overlap = self.kernel_size - self.stride;
        let in_len = x.dim(2)?;

        // Run conv_transpose with NO padding to get full overlapping output
        // Output length = (input_len - 1) * stride + kernel_size
        let y = x.conv_transpose1d(
            &self.weight,
            0, // no padding
            0, // no output_padding
            self.stride,
            1, // dilation
            self.groups,
        )?;

        // Add bias if present
        let y = if let Some(bias) = &self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            y.broadcast_add(&bias)?
        } else {
            y
        };

        let out_len = y.dim(2)?;

        // Verify output length: should be (in_len - 1) * stride + kernel
        let expected_len = (in_len - 1) * self.stride + self.kernel_size;
        debug_assert_eq!(
            out_len, expected_len,
            "ConvTranspose1d output length mismatch: got {}, expected {}",
            out_len, expected_len
        );

        // Add previous partial buffer to left edge of output
        // y[..., :overlap] += state.partial
        let left_edge = y.narrow(2, 0, overlap)?;
        let left_edge = left_edge.add(&state.partial)?;

        // Get the middle and right portions
        let middle_len = out_len.saturating_sub(2 * overlap);
        let output = if middle_len > 0 {
            let middle = y.narrow(2, overlap, middle_len)?;
            Tensor::cat(&[&left_edge, &middle], 2)?
        } else {
            // Very short output - just use left edge (minus the part going to partial)
            left_edge.narrow(2, 0, out_len.saturating_sub(overlap))?
        };

        // Store right edge (minus bias) as new partial buffer
        // for_partial = y[..., -overlap:]
        // if bias: for_partial -= bias
        let right_edge = y.narrow(2, out_len - overlap, overlap)?;
        let new_partial = if let Some(bias) = &self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            right_edge.broadcast_sub(&bias)?
        } else {
            right_edge
        };
        state.partial = new_partial;

        Ok(output)
    }

    /// Get overlap size (kernel - stride)
    fn overlap(&self) -> usize {
        self.kernel_size - self.stride
    }

    /// Get output channels (for initializing state)
    fn out_channels(&self) -> Result<usize> {
        // Weight shape: [in_channels, out_channels/groups, kernel] for groups=1
        // Weight shape: [channels, 1, kernel] for depthwise
        if self.groups == 1 {
            Ok(self.weight.dim(1)?)
        } else {
            // Depthwise: out_channels = groups
            Ok(self.groups)
        }
    }
}

/// Residual block in the decoder
#[derive(Debug)]
struct ResidualBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ResidualBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        // block.1.conv: narrow then block.3.conv: expand back
        let hidden = channels / 2;
        let conv1 = Conv1d::new(channels, hidden, 3, vb.pp("1.conv"))?;
        let conv2 = Conv1d::new(hidden, channels, 1, vb.pp("3.conv"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Python SEANet ResBlock: ELU → Conv1 → ELU → Conv2
        // (ELU is applied BEFORE each conv, not after)
        let h = x.elu(1.0)?;
        let h = self.conv1.forward(&h)?;
        let h = h.elu(1.0)?;
        let h = self.conv2.forward(&h)?;
        x + h
    }

    fn forward_streaming(&self, x: &Tensor, state: &mut StreamingResBlockState) -> Result<Tensor> {
        // Same as forward but uses streaming conv1
        let h = x.elu(1.0)?;
        let h = self.conv1.forward_streaming(&h, &mut state.conv1_state)?;
        let h = h.elu(1.0)?;
        // Conv2 has k=1, no streaming needed
        let h = self.conv2.forward(&h)?;
        x + h
    }
}

/// Decoder transformer layer with layer scales and RoPE
#[derive(Debug)]
struct DecoderTransformerLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    in_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    layer_scale_1: Tensor,
    layer_scale_2: Tensor,
    num_heads: usize,
    head_dim: usize,
    // Streaming state (KV cache)
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl DecoderTransformerLayer {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;

        let norm1 = LayerNorm::new(dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = LayerNorm::new(dim, 1e-5, vb.pp("norm2"))?;

        // Self-attention projections (no bias in this model)
        let in_proj = candle_nn::linear_no_bias(dim, dim * 3, vb.pp("self_attn.in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(dim, dim, vb.pp("self_attn.out_proj"))?;

        // FFN (no bias)
        let linear1 = candle_nn::linear_no_bias(dim, dim * 4, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear_no_bias(dim * 4, dim, vb.pp("linear2"))?;

        // Layer scales
        let layer_scale_1 = vb.get(dim, "layer_scale_1.scale")?;
        let layer_scale_2 = vb.get(dim, "layer_scale_2.scale")?;

        Ok(Self {
            norm1,
            norm2,
            in_proj,
            out_proj,
            linear1,
            linear2,
            layer_scale_1,
            layer_scale_2,
            num_heads,
            head_dim,
            k_cache: None,
            v_cache: None,
        })
    }

    fn forward(&self, x: &Tensor, rope: &crate::modules::rotary::RotaryEmbedding) -> Result<Tensor> {
        let (batch, seq, dim) = x.dims3()?;

        // Self-attention
        let h = self.norm1.forward(x)?;
        let qkv = self.in_proj.forward(&h)?;
        let qkv = qkv.reshape((batch, seq, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // [3, batch, heads, seq, head_dim]

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        // Apply RoPE to Q and K
        // Q, K are [batch, heads, seq, head_dim]
        // RoPE expects [batch, seq, heads, head_dim]
        let q = q.permute((0, 2, 1, 3))?; // [batch, seq, heads, head_dim]
        let k = k.permute((0, 2, 1, 3))?; // [batch, seq, heads, head_dim]
        let (q, k) = rope.forward(&q, &k, 0)?;
        // Permute back to [batch, heads, seq, head_dim]
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;

        // Scaled dot-product attention with causal mask
        // Python's MimiStreamingMultiheadAttention uses causal attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = (attn / scale)?;

        // Create causal mask: positions can only attend to earlier or same positions
        // Shape: [seq, seq] -> [1, 1, seq, seq] for broadcasting
        let device = attn.device();
        let causal_mask = Tensor::tril2(seq, candle_core::DType::F32, device)?;
        // Convert to attention mask: 1 (allowed) -> 0, 0 (masked) -> -inf
        // First invert: 1 -> 0, 0 -> 1
        let ones = Tensor::ones_like(&causal_mask)?;
        let inverted = (ones - &causal_mask)?;
        // Then multiply by large negative value
        let attn_mask = (inverted * (-1e9))?;
        let attn_mask = attn_mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, seq]

        let attn = attn.broadcast_add(&attn_mask)?;
        let attn = candle_nn::ops::softmax(&attn, 3)?;
        let attn_out = attn.matmul(&v)?;

        // Reshape back
        let attn_out = attn_out.permute((0, 2, 1, 3))?; // [batch, seq, heads, head_dim]
        let attn_out = attn_out.reshape((batch, seq, dim))?;
        let attn_out = self.out_proj.forward(&attn_out)?;

        // Apply layer scale and residual
        let attn_out = attn_out.broadcast_mul(&self.layer_scale_1)?;
        let x = (x + attn_out)?;

        // FFN
        let h = self.norm2.forward(&x)?;
        let h = self.linear1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.linear2.forward(&h)?;

        // Apply layer scale and residual
        let h = h.broadcast_mul(&self.layer_scale_2)?;
        x + h
    }

    /// Streaming forward with KV cache
    ///
    /// Processes a chunk of input, caching K/V for future chunks.
    /// This maintains causal attention context across streaming frames.
    fn forward_streaming(&mut self, x: &Tensor, rope: &crate::modules::rotary::RotaryEmbedding) -> Result<Tensor> {
        let (batch, seq, dim) = x.dims3()?;

        // Get current cache length (for RoPE offset)
        let offset = self.k_cache.as_ref().map_or(0, |k| k.dim(2).unwrap_or(0));

        // Self-attention
        let h = self.norm1.forward(x)?;
        let qkv = self.in_proj.forward(&h)?;
        let qkv = qkv.reshape((batch, seq, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // [3, batch, heads, seq, head_dim]

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        // Apply RoPE with correct offset
        // Q, K are [batch, heads, seq, head_dim]
        // RoPE expects [batch, seq, heads, head_dim]
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;
        let (q, k) = rope.forward(&q, &k, offset)?;
        // Permute back to [batch, heads, seq, head_dim]
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;

        // Update KV cache
        let (k_full, v_full) = match (&self.k_cache, &self.v_cache) {
            (Some(k_cache), Some(v_cache)) => {
                let k_new = Tensor::cat(&[k_cache, &k], 2)?;
                let v_new = Tensor::cat(&[v_cache, &v], 2)?;
                (k_new, v_new)
            },
            _ => (k, v.clone()),
        };
        self.k_cache = Some(k_full.clone());
        self.v_cache = Some(v_full.clone());

        // Compute attention: Q attends to full K/V cache
        // NOTE: Decoder transformer uses NON-CAUSAL (full) self-attention
        // Each position can attend to ALL positions in the sequence
        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k_full.transpose(2, 3)?)?;
        let attn = (attn / scale)?;

        // No causal mask - full attention (decoder transformer is non-causal)
        let attn = candle_nn::ops::softmax(&attn, 3)?;
        let attn_out = attn.matmul(&v_full)?;

        // Reshape back
        let attn_out = attn_out.permute((0, 2, 1, 3))?;
        let attn_out = attn_out.reshape((batch, seq, dim))?;
        let attn_out = self.out_proj.forward(&attn_out)?;

        // Apply layer scale and residual
        let attn_out = attn_out.broadcast_mul(&self.layer_scale_1)?;
        let x = (x + attn_out)?;

        // FFN (same as batch)
        let h = self.norm2.forward(&x)?;
        let h = self.linear1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.linear2.forward(&h)?;

        let h = h.broadcast_mul(&self.layer_scale_2)?;
        x + h
    }

    /// Reset the KV cache
    fn reset_cache(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }
}

/// Decoder transformer with RoPE
#[derive(Debug)]
struct DecoderTransformer {
    layers: Vec<DecoderTransformerLayer>,
    rope: crate::modules::rotary::RotaryEmbedding,
}

impl DecoderTransformer {
    fn new(dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let num_heads = 8; // 512 / 64 = 8 heads
        let head_dim = dim / num_heads; // 64
        let device = vb.device();

        // Create RoPE with same parameters as Python's MimiStreamingMultiheadAttention
        // Python uses max_period=10000.0 (default)
        let rope = crate::modules::rotary::RotaryEmbedding::new(
            head_dim, 4096,    // max_seq_len (should be enough for any audio)
            10000.0, // base (max_period in Python)
            device,
        )?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(DecoderTransformerLayer::new(
                dim,
                num_heads,
                vb.pp(format!("transformer.layers.{}", i)),
            )?);
        }
        Ok(Self { layers, rope })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope)?;
        }
        Ok(x)
    }

    /// Streaming forward with KV cache across all layers
    fn forward_streaming(&mut self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &mut self.layers {
            x = layer.forward_streaming(&x, &self.rope)?;
        }
        Ok(x)
    }

    /// Reset all layer KV caches
    fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.reset_cache();
        }
    }
}

/// SEANet-style decoder
#[derive(Debug)]
struct SEANetDecoder {
    input_conv: Conv1d,
    upsample_blocks: Vec<(ConvTranspose1d, Option<ResidualBlock>)>,
    output_conv: Conv1d,
}

impl SEANetDecoder {
    fn new(vb: VarBuilder) -> Result<Self> {
        // model.0.conv: 512 -> 512, k=7
        let input_conv = Conv1d::new(512, 512, 7, vb.pp("model.0.conv"))?;

        // Upsample blocks with residuals
        // Strides are derived from kernel sizes and expected upsampling
        let mut upsample_blocks = Vec::new();

        // model.2.convtr: 512 -> 256, k=12, stride=6
        let convtr2 = ConvTranspose1d::new(512, 256, 12, 6, vb.pp("model.2.convtr"))?;
        let block3 = ResidualBlock::new(256, vb.pp("model.3.block"))?;
        upsample_blocks.push((convtr2, Some(block3)));

        // model.5.convtr: 256 -> 128, k=10, stride=5
        let convtr5 = ConvTranspose1d::new(256, 128, 10, 5, vb.pp("model.5.convtr"))?;
        let block6 = ResidualBlock::new(128, vb.pp("model.6.block"))?;
        upsample_blocks.push((convtr5, Some(block6)));

        // model.8.convtr: 128 -> 64, k=8, stride=4
        let convtr8 = ConvTranspose1d::new(128, 64, 8, 4, vb.pp("model.8.convtr"))?;
        let block9 = ResidualBlock::new(64, vb.pp("model.9.block"))?;
        upsample_blocks.push((convtr8, Some(block9)));

        // model.11.conv: 64 -> 1, k=3
        let output_conv = Conv1d::new(64, 1, 3, vb.pp("model.11.conv"))?;

        Ok(Self {
            input_conv,
            upsample_blocks,
            output_conv,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Input: [batch, channels, seq]
        // Python order: Conv → ELU → ConvTranspose → ResBlock → ELU → ...
        // Note: Batch processing produces lower amplitude than Python's streaming
        // due to lack of inter-frame state accumulation. See PORTING_STATUS.md.
        let mut x = self.input_conv.forward(x)?;
        x = x.elu(1.0)?;

        // Upsample through blocks
        // Python: ConvTranspose → ResBlock → ELU (before next stage)
        for (convtr, block) in &self.upsample_blocks {
            x = convtr.forward(&x)?;
            if let Some(res_block) = block {
                x = res_block.forward(&x)?;
            }
            x = x.elu(1.0)?; // ELU after ResBlock, before next ConvTranspose
        }

        // Output projection (ELU already applied after last ResBlock)
        // Note: Python SEANet does NOT apply tanh - output is raw from final conv
        self.output_conv.forward(&x)
    }

    /// Streaming forward with full streaming support for all layers
    ///
    /// Uses streaming mode for ALL convolution layers:
    /// - Conv1d: causal context buffer
    /// - ConvTranspose1d: overlap-add state
    /// - ResBlocks: streaming conv1
    fn forward_streaming(&self, x: &Tensor, state: &mut StreamingSEANetState) -> Result<Tensor> {
        // Input: [batch, 512, seq] (typically seq=16 from upsampler)
        // Use streaming mode for input conv (causal context)
        let mut x = self.input_conv.forward_streaming(x, &mut state.input_conv_state)?;
        x = x.elu(1.0)?;

        // Upsample through blocks - streaming for ALL layers
        for (i, (convtr, block)) in self.upsample_blocks.iter().enumerate() {
            x = convtr.forward_streaming(&x, &mut state.convtr_states[i])?;
            if let Some(res_block) = block {
                // Use streaming mode for ResBlock conv1 (conv2 is k=1, no context needed)
                x = res_block.forward_streaming(&x, &mut state.resblock_states[i])?;
            }
            x = x.elu(1.0)?;
        }

        // Use streaming mode for output conv (causal context)
        self.output_conv.forward_streaming(&x, &mut state.output_conv_state)
    }
}

/// Mimi VAE Decoder
///
/// Converts low-dimensional latents from FlowLM to audio waveforms.
#[derive(Debug)]
pub struct MimiDecoder {
    config: MimiConfig,
    output_proj: Conv1d, // quantizer.output_proj: projects 32 -> 512
    decoder_transformer: DecoderTransformer,
    upsample_convtr: ConvTranspose1d, // 16x temporal upsampling before SEANet
    seanet: SEANetDecoder,
}

impl MimiDecoder {
    pub fn new(config: MimiConfig, vb: VarBuilder) -> Result<Self> {
        // Output projection from latent (32) to mimi dim (512)
        // This is stored as quantizer.output_proj in the model
        let output_proj = Conv1d::new_no_bias(config.latent_dim, config.mimi_dim, 1, vb.pp("quantizer.output_proj"))?;

        // Decoder transformer (2 layers)
        let decoder_transformer =
            DecoderTransformer::new(config.mimi_dim, config.num_transformer_layers, vb.pp("decoder_transformer"))?;

        // Depthwise 16x temporal upsampling
        // Weight path: upsample.convtr.convtr
        // Shape: [512, 1, 32] = depthwise with groups=512
        let upsample_convtr = ConvTranspose1d::new_depthwise(
            config.mimi_dim, // 512 channels
            32,              // kernel_size
            16,              // stride (16x upsampling)
            vb.pp("upsample.convtr.convtr"),
        )?;

        // SEANet decoder for waveform generation
        let seanet = SEANetDecoder::new(vb.pp("decoder"))?;

        Ok(Self {
            config,
            output_proj,
            decoder_transformer,
            upsample_convtr,
            seanet,
        })
    }

    /// Create initial streaming state for frame-by-frame processing
    pub fn init_streaming_state(&self, batch_size: usize, device: &Device) -> Result<StreamingMimiState> {
        // Depthwise upsampler: 512 channels, k=32, s=16 → overlap = 16
        let upsample_state = StreamingConvTr1dState::new(
            batch_size,
            self.config.mimi_dim,           // 512
            self.upsample_convtr.overlap(), // 16
            device,
        )?;

        // SEANet state
        // Input conv: 512 channels, k=7, s=1 → context = 6
        let input_conv_state = StreamingConv1dState::new(batch_size, 512, 6, device)?;

        // ConvTranspose states
        let convtr_states = [
            StreamingConvTr1dState::new(batch_size, 256, 6, device)?, // k=12, s=6
            StreamingConvTr1dState::new(batch_size, 128, 5, device)?, // k=10, s=5
            StreamingConvTr1dState::new(batch_size, 64, 4, device)?,  // k=8, s=4
        ];

        // ResBlock states (conv1 k=3, context=2)
        let resblock_states = [
            StreamingResBlockState {
                conv1_state: StreamingConv1dState::new(batch_size, 256, 2, device)?,
            },
            StreamingResBlockState {
                conv1_state: StreamingConv1dState::new(batch_size, 128, 2, device)?,
            },
            StreamingResBlockState {
                conv1_state: StreamingConv1dState::new(batch_size, 64, 2, device)?,
            },
        ];

        // Output conv: 64 channels, k=3, s=1 → context = 2
        let output_conv_state = StreamingConv1dState::new(batch_size, 64, 2, device)?;

        Ok(StreamingMimiState {
            upsample_state,
            seanet_state: StreamingSEANetState {
                input_conv_state,
                convtr_states,
                resblock_states,
                output_conv_state,
            },
        })
    }

    /// Decode latents to audio waveform using streaming processing
    ///
    /// This method processes latents with streaming ConvTranspose1d for the
    /// upsample and SEANet layers. Based on studying Kyutai's official Moshi
    /// Rust implementation.
    ///
    /// Strategy:
    /// 1. Batch: output_proj, upsample (with overlap-add), transformer
    /// 2. Streaming: SEANet ConvTranspose1d layers (overlap-add per frame)
    ///
    /// Input: [batch, seq, latent_dim] latent representations
    /// Output: [batch, samples] audio waveform
    pub fn forward_streaming(&self, latents: &Tensor) -> Result<Tensor> {
        let (batch, seq, _latent_dim) = latents.dims3()?;
        let device = latents.device();
        eprintln!("[Mimi-Stream] Processing {} latent frames", seq);

        // Step 1: Transpose to [batch, latent_dim, seq] for conv
        let x = latents.transpose(1, 2)?;

        // Step 2: Project from latent (32) to mimi dim (512)
        let x = self.output_proj.forward(&x)?;
        eprintln!("[Mimi-Stream] After output_proj: {:?}", x.dims());

        // Step 3: 16x temporal upsampling
        // Use streaming ConvTranspose1d to properly accumulate overlap-add state
        let mut upsample_state = StreamingConvTr1dState::new(
            batch,
            self.config.mimi_dim,           // 512
            self.upsample_convtr.overlap(), // 16
            device,
        )?;

        // Process frame by frame through upsampler for proper overlap-add
        let mut upsampled_chunks: Vec<Tensor> = Vec::with_capacity(seq);
        for frame_idx in 0..seq {
            let frame = x.narrow(2, frame_idx, 1)?;
            let upsampled = self.upsample_convtr.forward_streaming(&frame, &mut upsample_state)?;
            if upsampled.dim(2)? > 0 {
                upsampled_chunks.push(upsampled);
            }
        }

        let x = if upsampled_chunks.is_empty() {
            return Err(candle_core::Error::Msg("No upsampled frames produced".to_string()));
        } else {
            Tensor::cat(&upsampled_chunks, 2)?
        };
        eprintln!("[Mimi-Stream] After streaming upsample: {:?}", x.dims());

        // Step 4: Transpose for transformer: [batch, seq*16, dim]
        let x = x.transpose(1, 2)?;

        // Step 5: Decoder transformer (batch mode - needs full causal context)
        let x = self.decoder_transformer.forward(&x)?;
        eprintln!("[Mimi-Stream] After decoder_transformer: {:?}", x.dims());

        // Step 6: Transpose for SEANet: [batch, dim, seq*16]
        let x = x.transpose(1, 2)?;

        // Step 7: SEANet decoder with frame-by-frame streaming
        // Process in chunks of 16 samples (one upsampled latent frame)
        // to properly accumulate overlap-add state between chunks
        let mut seanet_state = self.init_seanet_state(batch, device)?;

        let total_len = x.dim(2)?;
        let chunk_size = 16; // upsampled frames per latent
        let num_chunks = (total_len + chunk_size - 1) / chunk_size;

        let mut audio_chunks: Vec<Tensor> = Vec::with_capacity(num_chunks);
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let len = usize::min(chunk_size, total_len - start);
            let chunk = x.narrow(2, start, len)?;

            let audio_chunk = self.seanet.forward_streaming(&chunk, &mut seanet_state)?;
            if audio_chunk.dim(2)? > 0 {
                audio_chunks.push(audio_chunk);
            }
        }

        let audio = if audio_chunks.is_empty() {
            Tensor::zeros((batch, 1, 0), DType::F32, device)?
        } else {
            Tensor::cat(&audio_chunks, 2)?
        };

        // Log final audio stats
        let audio_stats: Vec<f32> = audio.flatten_all()?.to_vec1()?;
        let max_amp = audio_stats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[Mimi-Stream] Final audio shape {:?}, max={:.4}", audio.dims(), max_amp);

        // Squeeze channel dim: [batch, 1, samples] -> [batch, samples]
        audio.squeeze(1)
    }

    /// Initialize SEANet streaming state
    fn init_seanet_state(&self, batch_size: usize, device: &Device) -> Result<StreamingSEANetState> {
        // Input conv: 512 channels, k=7, s=1 → context = 6
        let input_conv_state = StreamingConv1dState::new(batch_size, 512, 6, device)?;

        // ConvTranspose states: k=12,s=6; k=10,s=5; k=8,s=4
        let convtr_states = [
            StreamingConvTr1dState::new(batch_size, 256, 6, device)?, // k=12, s=6 -> overlap=6
            StreamingConvTr1dState::new(batch_size, 128, 5, device)?, // k=10, s=5 -> overlap=5
            StreamingConvTr1dState::new(batch_size, 64, 4, device)?,  // k=8, s=4 -> overlap=4
        ];

        // ResBlock states (conv1 k=3, context=2)
        // Streaming state stores INPUT to conv1, which is the ResBlock input channels
        // ResBlock 0: 256->128->256, conv1 INPUT = 256 (after ELU)
        // ResBlock 1: 128->64->128, conv1 INPUT = 128 (after ELU)
        // ResBlock 2: 64->32->64, conv1 INPUT = 64 (after ELU)
        let resblock_states = [
            StreamingResBlockState {
                conv1_state: StreamingConv1dState::new(batch_size, 256, 2, device)?,
            },
            StreamingResBlockState {
                conv1_state: StreamingConv1dState::new(batch_size, 128, 2, device)?,
            },
            StreamingResBlockState {
                conv1_state: StreamingConv1dState::new(batch_size, 64, 2, device)?,
            },
        ];

        // Output conv: 64 channels, k=3, s=1 → context = 2
        let output_conv_state = StreamingConv1dState::new(batch_size, 64, 2, device)?;

        Ok(StreamingSEANetState {
            input_conv_state,
            convtr_states,
            resblock_states,
            output_conv_state,
        })
    }

    /// Fully streaming forward with KV cache and streaming convolutions
    ///
    /// This method processes latents through the entire pipeline frame-by-frame
    /// with proper streaming state at each layer:
    /// 1. output_proj (k=1, stateless)
    /// 2. upsample (streaming ConvTranspose1d with overlap-add)
    /// 3. decoder_transformer (streaming with KV cache)
    /// 4. SEANet (streaming convolutions)
    ///
    /// Input: [batch, seq, latent_dim] latent representations
    /// Output: [batch, samples] audio waveform
    pub fn forward_true_streaming(&mut self, latents: &Tensor) -> Result<Tensor> {
        use crate::modules::conv::StreamableConvTranspose1d;

        let (batch, seq, _latent_dim) = latents.dims3()?;
        let device = latents.device();
        eprintln!("[Mimi-TrueStream] Processing {} latent frames", seq);

        // Step 1: Transpose to [batch, latent_dim, seq] for conv
        let x = latents.transpose(1, 2)?;

        // Step 2: Project from latent (32) to mimi dim (512)
        // output_proj has k=1, so it's stateless
        let x = self.output_proj.forward(&x)?;
        eprintln!("[Mimi-TrueStream] After output_proj: {:?}", x.dims());

        // Step 3: Create streaming upsample convtr
        let mut upsample_streaming = StreamableConvTranspose1d::from_weights(
            self.upsample_convtr.weight.clone(),
            self.upsample_convtr.bias.clone(),
            self.upsample_convtr.kernel_size,
            self.upsample_convtr.stride,
            self.upsample_convtr.groups,
        );

        // Step 4: Reset transformer KV cache for fresh inference
        self.decoder_transformer.reset_cache();

        // Step 5: Create SEANet streaming state (must persist across ALL frames)
        let mut seanet_state = self.init_seanet_state(batch, device)?;

        // Step 6: Process frame by frame
        let mut audio_chunks: Vec<Tensor> = Vec::with_capacity(seq);

        for frame_idx in 0..seq {
            // Extract single latent frame: [batch, 512, 1]
            let frame = x.narrow(2, frame_idx, 1)?;

            // 3a. Streaming upsample: [batch, 512, 1] -> [batch, 512, 16]
            let upsampled = upsample_streaming.step(&StreamTensor::from_tensor(frame))?;
            if upsampled.is_empty() {
                continue;
            }
            let upsampled = upsampled.unwrap();

            // 3b. Transpose for transformer: [batch, 16, 512]
            let x = upsampled.transpose(1, 2)?;

            // 3c. Streaming transformer with KV cache
            let x = self.decoder_transformer.forward_streaming(&x)?;

            // 3d. Transpose for SEANet: [batch, 512, 16]
            let x = x.transpose(1, 2)?;

            // 3e. SEANet decoder with streaming convolutions
            //     State persists across ALL frames for proper overlap-add accumulation
            let audio = self.seanet.forward_streaming(&x, &mut seanet_state)?;

            if audio.dim(2)? > 0 {
                audio_chunks.push(audio);
            }
        }

        let audio = if audio_chunks.is_empty() {
            Tensor::zeros((batch, 1, 0), DType::F32, device)?
        } else {
            Tensor::cat(&audio_chunks, 2)?
        };

        // Log final audio stats
        let audio_stats: Vec<f32> = audio.flatten_all()?.to_vec1()?;
        let max_amp = audio_stats.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[Mimi-TrueStream] Final audio shape {:?}, max={:.4}", audio.dims(), max_amp);

        // Squeeze channel dim: [batch, 1, samples] -> [batch, samples]
        audio.squeeze(1)
    }

    /// Decode latents to audio waveform (batch mode - produces lower amplitude)
    ///
    /// IMPORTANT: The correct order is:
    /// 1. output_proj: [B, 32, seq] -> [B, 512, seq]
    /// 2. upsample (16x): [B, 512, seq] -> [B, 512, seq*16]
    /// 3. decoder_transformer: [B, 512, seq*16] -> [B, 512, seq*16]
    /// 4. SEANet: [B, 512, seq*16] -> [B, 1, audio_samples]
    ///
    /// Input: [batch, seq, latent_dim] latent representations
    /// Output: [batch, samples] audio waveform
    ///
    /// Note: This batch mode produces ~5-6x lower amplitude than Python's
    /// streaming implementation. Use `forward_streaming` for correct output.
    pub fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        // Step 1: Transpose to [batch, latent_dim, seq] for conv
        let x = latents.transpose(1, 2)?;
        eprintln!("[Mimi] after input transpose: {:?}", x.dims());

        // Step 2: Project from latent (32) to mimi dim (512)
        let x = self.output_proj.forward(&x)?;
        Self::log_tensor_stats("output_proj", &x)?;

        // Step 3: 16x temporal upsampling (BEFORE transformer!)
        // This brings frame rate from 12.5 Hz to 200 Hz
        let x = self.upsample_convtr.forward(&x)?;
        Self::log_tensor_stats("upsample", &x)?;
        eprintln!("[Mimi] post-upsample shape: {:?}", x.dims());

        // Step 4: Transpose for transformer: [batch, seq*16, dim]
        let x = x.transpose(1, 2)?;

        // Step 5: Decoder transformer
        let x = self.decoder_transformer.forward(&x)?;
        Self::log_tensor_stats("decoder_transformer", &x)?;

        // Step 6: Transpose for convolutions: [batch, dim, seq*16]
        let x = x.transpose(1, 2)?;
        eprintln!("[Mimi] pre-seanet shape: {:?}", x.dims());

        // Step 7: SEANet decoder to waveform (120x upsampling: 200 Hz -> 24kHz)
        let audio = self.seanet.forward(&x)?;
        Self::log_tensor_stats("seanet_output", &audio)?;

        // Squeeze channel dim: [batch, 1, samples] -> [batch, samples]
        audio.squeeze(1)
    }

    /// Log tensor statistics for debugging
    fn log_tensor_stats(name: &str, tensor: &Tensor) -> Result<()> {
        let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let mean = flat.iter().sum::<f32>() / flat.len() as f32;
        let max_val = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let std = (flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat.len() as f32).sqrt();
        eprintln!(
            "[Mimi] {}: mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
            name, mean, std, min_val, max_val
        );
        Ok(())
    }

    /// Decode with overlap-add for streaming
    pub fn decode_streaming(
        &self,
        latents: &Tensor,
        overlap_samples: usize,
        previous_tail: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // Decode full chunk
        let audio = self.forward(latents)?;
        let total_samples = audio.dim(1)?;

        if let Some(prev) = previous_tail {
            let prev_len = prev.dim(0)?;
            let fade_len = overlap_samples.min(prev_len).min(total_samples);

            if fade_len > 0 {
                let fade_out: Vec<f32> = (0..fade_len).map(|i| 1.0 - (i as f32 / fade_len as f32)).collect();
                let fade_in: Vec<f32> = (0..fade_len).map(|i| i as f32 / fade_len as f32).collect();

                let fade_out = Tensor::from_vec(fade_out, (fade_len,), audio.device())?;
                let fade_in = Tensor::from_vec(fade_in, (fade_len,), audio.device())?;

                let prev_overlap = prev.narrow(0, prev_len - fade_len, fade_len)?;
                let curr_overlap = audio.narrow(1, 0, fade_len)?.squeeze(0)?;

                let blended = (prev_overlap.broadcast_mul(&fade_out)? + curr_overlap.broadcast_mul(&fade_in)?)?;

                let rest = audio.narrow(1, fade_len, total_samples - fade_len)?;
                let output = Tensor::cat(&[&blended.unsqueeze(0)?, &rest], 1)?;

                let tail_start = total_samples.saturating_sub(overlap_samples);
                let tail = audio.narrow(1, tail_start, total_samples - tail_start)?.squeeze(0)?;

                Ok((output, tail))
            } else {
                let tail = audio.narrow(1, total_samples - overlap_samples, overlap_samples)?.squeeze(0)?;
                Ok((audio, tail))
            }
        } else {
            let tail_start = total_samples.saturating_sub(overlap_samples);
            let tail = audio.narrow(1, tail_start, total_samples - tail_start)?.squeeze(0)?;
            Ok((audio, tail))
        }
    }

    /// Get samples per latent frame
    pub fn samples_per_frame(&self) -> usize {
        (self.config.sample_rate as f32 / self.config.frame_rate) as usize
    }

    pub fn config(&self) -> &MimiConfig {
        &self.config
    }

    pub fn sample_rate(&self) -> usize {
        self.config.sample_rate
    }
}
