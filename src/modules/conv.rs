//! Convolution modules for audio processing
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{Result, Tensor};
use candle_nn::{Conv1d as CandleConv1d, Conv1dConfig, ConvTranspose1d as CandleConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

/// Apply ELU activation function with alpha=1.0
/// ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
/// Python SEANet uses nn.ELU(alpha=1.0), NOT GELU
fn elu(x: &Tensor) -> Result<Tensor> {
    // Use candle's elu method if available, otherwise implement manually
    // Candle's elu takes (tensor, alpha) - we use alpha=1.0 to match Python
    x.elu(1.0)
}

/// 1D Convolution wrapper
#[derive(Debug)]
pub struct Conv1d {
    conv: CandleConv1d,
    #[allow(dead_code)]
    kernel_size: usize,
    #[allow(dead_code)]
    padding: usize,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding,
            stride,
            dilation: 1,
            groups: 1,
        };

        let conv = candle_nn::conv1d(in_channels, out_channels, kernel_size, config, vb)?;

        Ok(Self {
            conv,
            kernel_size,
            padding,
        })
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// Causal 1D Convolution (for autoregressive models)
#[derive(Debug)]
pub struct CausalConv1d {
    conv: CandleConv1d,
    #[allow(dead_code)]
    kernel_size: usize,
}

impl CausalConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Causal padding: only pad on the left
        let causal_padding = (kernel_size - 1) * dilation;

        let config = Conv1dConfig {
            padding: causal_padding,
            stride,
            dilation,
            groups: 1,
        };

        let conv = candle_nn::conv1d(in_channels, out_channels, kernel_size, config, vb)?;

        Ok(Self { conv, kernel_size })
    }
}

impl Module for CausalConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        // Remove future samples (causal)
        let seq_len = x.dim(2)?;
        y.narrow(2, 0, seq_len)
    }
}

/// Transposed 1D Convolution (upsampling)
#[derive(Debug)]
pub struct ConvTranspose1d {
    conv: CandleConvTranspose1d,
    #[allow(dead_code)]
    kernel_size: usize,
    #[allow(dead_code)]
    stride: usize,
}

impl ConvTranspose1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            padding,
            stride,
            dilation: 1,
            output_padding: 0,
            groups: 1,
        };

        let conv = candle_nn::conv_transpose1d(in_channels, out_channels, kernel_size, config, vb)?;

        Ok(Self {
            conv,
            kernel_size,
            stride,
        })
    }
}

impl Module for ConvTranspose1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// SEANet encoder block
#[derive(Debug)]
pub struct SEANetEncoderBlock {
    conv1: CausalConv1d,
    conv2: CausalConv1d,
    downsample: Conv1d,
}

impl SEANetEncoderBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = CausalConv1d::new(in_channels, out_channels, kernel_size, 1, 1, vb.pp("conv1"))?;
        let conv2 = CausalConv1d::new(out_channels, out_channels, kernel_size, 1, 1, vb.pp("conv2"))?;
        let downsample = Conv1d::new(out_channels, out_channels, stride * 2, stride, stride / 2, vb.pp("downsample"))?;

        Ok(Self { conv1, conv2, downsample })
    }
}

impl Module for SEANetEncoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(x)?;
        let h = h.gelu_erf()?;
        let h = self.conv2.forward(&h)?;
        let h = h.gelu_erf()?;
        self.downsample.forward(&h)
    }
}

/// SEANet decoder block
#[derive(Debug)]
pub struct SEANetDecoderBlock {
    upsample: ConvTranspose1d,
    conv1: CausalConv1d,
    conv2: CausalConv1d,
}

impl SEANetDecoderBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let upsample = ConvTranspose1d::new(in_channels, out_channels, stride * 2, stride, stride / 2, vb.pp("upsample"))?;
        let conv1 = CausalConv1d::new(out_channels, out_channels, kernel_size, 1, 1, vb.pp("conv1"))?;
        let conv2 = CausalConv1d::new(out_channels, out_channels, kernel_size, 1, 1, vb.pp("conv2"))?;

        Ok(Self { upsample, conv1, conv2 })
    }
}

impl Module for SEANetDecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.upsample.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = elu(&h)?;  // Python SEANet uses ELU, not GELU
        let h = self.conv2.forward(&h)?;
        elu(&h)  // Python SEANet uses ELU, not GELU
    }
}
