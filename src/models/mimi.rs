//! Mimi VAE Decoder
//!
//! Neural audio codec decoder that converts quantized latents
//! to high-quality 24kHz audio.
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::modules::layer_norm::LayerNorm;

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
}

/// ConvTranspose1d for upsampling
#[derive(Debug)]
struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
}

impl ConvTranspose1d {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        // ConvTranspose weight shape is [in_channels, out_channels, kernel]
        let weight = vb.get((in_channels, out_channels, kernel_size), "weight")?;
        let bias = vb.get(out_channels, "bias").ok();

        // Padding to maintain proper output size
        let padding = (kernel_size - stride) / 2;
        let output_padding = (kernel_size - stride) % 2;

        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
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

        // Padding to maintain proper output size
        let padding = (kernel_size - stride) / 2;
        let output_padding = (kernel_size - stride) % 2;

        Ok(Self {
            weight,
            bias: None,
            stride,
            padding,
            output_padding,
            groups: channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv_transpose1d(
            &self.weight,
            self.padding,
            self.output_padding,
            self.stride,
            1, // dilation
            self.groups,
        )?;
        if let Some(bias) = &self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            x.broadcast_add(&bias)
        } else {
            Ok(x)
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
        let h = self.conv1.forward(x)?;
        let h = h.elu(1.0)?;  // Python SEANet ResBlock uses ELU, not GELU
        let h = self.conv2.forward(&h)?;
        x + h
    }
}

/// Decoder transformer layer with layer scales
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
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, dim) = x.dims3()?;

        // Self-attention
        let h = self.norm1.forward(x)?;
        let qkv = self.in_proj.forward(&h)?;
        let qkv = qkv.reshape((batch, seq, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // [3, batch, heads, seq, head_dim]

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = (attn / scale)?;
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
}

/// Decoder transformer
#[derive(Debug)]
struct DecoderTransformer {
    layers: Vec<DecoderTransformerLayer>,
}

impl DecoderTransformer {
    fn new(dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let num_heads = 8; // 512 / 64 = 8 heads
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(DecoderTransformerLayer::new(
                dim,
                num_heads,
                vb.pp(format!("transformer.layers.{}", i)),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
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
        let mut x = self.input_conv.forward(x)?;
        x = x.elu(1.0)?;  // Python SEANet uses ELU(alpha=1.0), not GELU

        // Upsample through blocks
        for (convtr, block) in &self.upsample_blocks {
            x = convtr.forward(&x)?;
            x = x.elu(1.0)?;  // Python SEANet uses ELU(alpha=1.0), not GELU
            if let Some(res_block) = block {
                x = res_block.forward(&x)?;
            }
        }

        // Output projection
        let x = self.output_conv.forward(&x)?;

        // Tanh to bound to [-1, 1]
        x.tanh()
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
        let output_proj = Conv1d::new_no_bias(
            config.latent_dim,
            config.mimi_dim,
            1,
            vb.pp("quantizer.output_proj"),
        )?;

        // Decoder transformer (2 layers)
        let decoder_transformer = DecoderTransformer::new(
            config.mimi_dim,
            config.num_transformer_layers,
            vb.pp("decoder_transformer"),
        )?;

        // Depthwise 16x temporal upsampling
        // Weight path: upsample.convtr.convtr
        // Shape: [512, 1, 32] = depthwise with groups=512
        let upsample_convtr = ConvTranspose1d::new_depthwise(
            config.mimi_dim,  // 512 channels
            32,               // kernel_size
            16,               // stride (16x upsampling)
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

    /// Decode latents to audio waveform
    ///
    /// Input: [batch, seq, latent_dim] latent representations
    /// Output: [batch, samples] audio waveform
    pub fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        // Transpose to [batch, latent_dim, seq] for conv
        let x = latents.transpose(1, 2)?;
        eprintln!("[Mimi] after input transpose: {:?}", x.dims());

        // Project from latent (32) to mimi dim (512)
        let x = self.output_proj.forward(&x)?;
        Self::log_tensor_stats("output_proj", &x)?;

        // Transpose back for transformer: [batch, seq, dim]
        let x = x.transpose(1, 2)?;

        // Decoder transformer
        let x = self.decoder_transformer.forward(&x)?;
        Self::log_tensor_stats("decoder_transformer", &x)?;

        // Transpose for convolutions: [batch, dim, seq]
        let x = x.transpose(1, 2)?;
        eprintln!("[Mimi] pre-upsample shape: {:?}", x.dims());

        // 16x temporal upsampling (depthwise ConvTranspose1d)
        // This brings frame rate from 12.5 Hz to 200 Hz
        let x = self.upsample_convtr.forward(&x)?;
        Self::log_tensor_stats("upsample", &x)?;
        eprintln!("[Mimi] post-upsample shape: {:?}", x.dims());

        // SEANet decoder to waveform (120x upsampling: 200 Hz -> 24kHz)
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
        eprintln!("[Mimi] {}: mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]", name, mean, std, min_val, max_val);
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
                let fade_out: Vec<f32> = (0..fade_len)
                    .map(|i| 1.0 - (i as f32 / fade_len as f32))
                    .collect();
                let fade_in: Vec<f32> = (0..fade_len)
                    .map(|i| i as f32 / fade_len as f32)
                    .collect();

                let fade_out = Tensor::from_vec(fade_out, (fade_len,), audio.device())?;
                let fade_in = Tensor::from_vec(fade_in, (fade_len,), audio.device())?;

                let prev_overlap = prev.narrow(0, prev_len - fade_len, fade_len)?;
                let curr_overlap = audio.narrow(1, 0, fade_len)?.squeeze(0)?;

                let blended = (prev_overlap.broadcast_mul(&fade_out)?
                    + curr_overlap.broadcast_mul(&fade_in)?)?;

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
