//! SEANet Decoder for waveform generation
//!
//! Converts latent representations to audio waveforms using
//! a series of upsampling convolution blocks.
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::modules::conv::{CausalConv1d, SEANetDecoderBlock};

/// SEANet decoder configuration
#[derive(Debug, Clone)]
pub struct SEANetConfig {
    pub latent_dim: usize,
    pub channels: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub strides: Vec<usize>,
    pub sample_rate: usize,
}

impl Default for SEANetConfig {
    fn default() -> Self {
        Self {
            latent_dim: 512,
            channels: vec![512, 256, 128, 64, 32],
            kernel_sizes: vec![7, 7, 7, 7, 7],
            strides: vec![8, 5, 4, 2, 2],
            sample_rate: 24000,
        }
    }
}

/// SEANet Decoder
#[derive(Debug)]
pub struct SEANetDecoder {
    config: SEANetConfig,
    input_conv: CausalConv1d,
    blocks: Vec<SEANetDecoderBlock>,
    output_conv: CausalConv1d,
}

impl SEANetDecoder {
    pub fn new(config: SEANetConfig, vb: VarBuilder) -> Result<Self> {
        // Input projection
        let input_conv = CausalConv1d::new(
            config.latent_dim,
            config.channels[0],
            7,
            1,
            1,
            vb.pp("input_conv"),
        )?;

        // Upsampling blocks
        let mut blocks = Vec::with_capacity(config.channels.len() - 1);
        for i in 0..(config.channels.len() - 1) {
            let block = SEANetDecoderBlock::new(
                config.channels[i],
                config.channels[i + 1],
                config.kernel_sizes[i],
                config.strides[i],
                vb.pp(format!("blocks.{}", i)),
            )?;
            blocks.push(block);
        }

        // Output projection to waveform (1 channel)
        let output_conv = CausalConv1d::new(
            *config.channels.last().unwrap_or(&32),
            1, // Mono output
            7,
            1,
            1,
            vb.pp("output_conv"),
        )?;

        Ok(Self {
            config,
            input_conv,
            blocks,
            output_conv,
        })
    }

    /// Decode latents to waveform
    pub fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        // Input: [batch, seq, latent_dim]
        // Transpose to [batch, latent_dim, seq] for conv
        let x = latents.transpose(1, 2)?;

        // Input projection
        let mut x = self.input_conv.forward(&x)?;
        x = x.elu(1.0)?;  // Python SEANet uses ELU(alpha=1.0), not GELU

        // Upsample through blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Output projection
        let x = self.output_conv.forward(&x)?;

        // Apply tanh to bound output to [-1, 1]
        let x = x.tanh()?;

        // Squeeze channel dimension: [batch, 1, samples] -> [batch, samples]
        x.squeeze(1)
    }

    /// Get upsampling ratio (latent frames to audio samples)
    pub fn upsampling_ratio(&self) -> usize {
        self.config.strides.iter().product()
    }

    pub fn config(&self) -> &SEANetConfig {
        &self.config
    }
}
