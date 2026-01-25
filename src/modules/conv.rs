//! Convolution modules for audio processing
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{Result, Tensor, D};
use candle_nn::{
    Conv1d as CandleConv1d, Conv1dConfig, ConvTranspose1d as CandleConvTranspose1d, ConvTranspose1dConfig, Module,
    VarBuilder,
};

use super::streaming::{StreamTensor, StreamingModule, TensorPadding};

/// Padding mode for streaming convolutions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PadMode {
    /// Fill initial context with zeros (default)
    #[default]
    Constant,
    /// Fill initial context with first sample repeated (SEANet style)
    Replicate,
}

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

        Ok(Self {
            conv1,
            conv2,
            downsample,
        })
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
        let upsample =
            ConvTranspose1d::new(in_channels, out_channels, stride * 2, stride, stride / 2, vb.pp("upsample"))?;
        let conv1 = CausalConv1d::new(out_channels, out_channels, kernel_size, 1, 1, vb.pp("conv1"))?;
        let conv2 = CausalConv1d::new(out_channels, out_channels, kernel_size, 1, 1, vb.pp("conv2"))?;

        Ok(Self { upsample, conv1, conv2 })
    }
}

impl Module for SEANetDecoderBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.upsample.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = elu(&h)?; // Python SEANet uses ELU, not GELU
        let h = self.conv2.forward(&h)?;
        elu(&h) // Python SEANet uses ELU, not GELU
    }
}

// ============================================================================
// STREAMING CONVOLUTIONS
// ============================================================================

/// Streamable 1D Convolution with causal (left-only) padding
///
/// This implements frame-by-frame streaming convolution where:
/// - LEFT padding is applied only on the FIRST frame (causal)
/// - Previous samples are buffered for context
/// - Output is produced as soon as enough samples are available
///
/// Pattern from Kyutai Moshi: rust/moshi-core/src/conv.rs
#[derive(Debug)]
pub struct StreamableConv1d {
    /// Convolution weights [out_channels, in_channels/groups, kernel_size]
    weight: Tensor,
    /// Optional bias [out_channels]
    bias: Option<Tensor>,
    /// Kernel size
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Dilation
    dilation: usize,
    /// Groups for grouped convolution
    groups: usize,
    /// Padding mode for first frame
    pad_mode: PadMode,

    // Streaming state
    /// Buffer of previous input samples
    state_prev_xs: Option<Tensor>,
    /// Whether this is the first frame (for replicate padding)
    is_first_frame: bool,
}

impl StreamableConv1d {
    /// Create a new StreamableConv1d from weight tensors
    pub fn from_weights(
        weight: Tensor,
        bias: Option<Tensor>,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        pad_mode: PadMode,
    ) -> Self {
        Self {
            weight,
            bias,
            kernel_size,
            stride,
            dilation,
            groups,
            pad_mode,
            state_prev_xs: None,
            is_first_frame: true,
        }
    }

    /// Create from a VarBuilder (for loading from safetensors)
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        pad_mode: PadMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((out_channels, in_channels / groups, kernel_size), "weight")?;
        let bias = vb.get(out_channels, "bias").ok();

        Ok(Self::from_weights(
            weight,
            bias,
            kernel_size,
            stride,
            dilation,
            groups,
            pad_mode,
        ))
    }

    /// Batch forward (non-streaming, applies symmetric padding)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Calculate causal padding (left only, then trim right)
        let effective_kernel = (self.kernel_size - 1) * self.dilation + 1;
        let padding_total = effective_kernel - self.stride;

        // Apply symmetric padding for batch mode
        let half_pad = padding_total / 2;
        let out = x.conv1d(&self.weight, half_pad, self.stride, self.dilation, self.groups)?;

        if let Some(bias) = &self.bias {
            let bias = bias.reshape((1, (), 1))?;
            out.broadcast_add(&bias)
        } else {
            Ok(out)
        }
    }

    /// Causal batch forward (for validation - applies left-only padding)
    pub fn forward_causal(&self, x: &Tensor) -> Result<Tensor> {
        let effective_kernel = (self.kernel_size - 1) * self.dilation + 1;
        let padding_total = effective_kernel - self.stride;

        // Apply LEFT padding only (causal)
        let x = x.pad_zeros_left(D::Minus1, padding_total)?;

        // Run conv with no padding (we've manually padded)
        let out = x.conv1d(&self.weight, 0, self.stride, self.dilation, self.groups)?;

        if let Some(bias) = &self.bias {
            let bias = bias.reshape((1, (), 1))?;
            out.broadcast_add(&bias)
        } else {
            Ok(out)
        }
    }
}

impl StreamingModule for StreamableConv1d {
    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        let xs = match xs.as_option() {
            None => return Ok(StreamTensor::empty()),
            Some(xs) => xs.clone(),
        };

        // Effective kernel size with dilation
        let k_eff = (self.kernel_size - 1) * self.dilation + 1;
        // Context we need to keep from previous frame
        let context_len = k_eff - self.stride;

        // Handle first frame padding
        let xs = if self.is_first_frame && context_len > 0 {
            self.is_first_frame = false;

            match self.pad_mode {
                PadMode::Constant => {
                    // Pad with zeros on left
                    xs.pad_zeros_left(D::Minus1, context_len)?
                },
                PadMode::Replicate => {
                    // Replicate first sample to fill context
                    // Get first sample: [B, C, 1]
                    let first_sample = xs.narrow(D::Minus1, 0, 1)?;
                    // Repeat it context_len times: [B, C, context_len]
                    let context = first_sample.repeat(&[1, 1, context_len])?;
                    // Concatenate: [context | xs]
                    Tensor::cat(&[&context, &xs], D::Minus1)?
                },
            }
        } else if self.is_first_frame {
            // First frame but no context needed (context_len == 0)
            self.is_first_frame = false;
            xs
        } else {
            xs
        };

        // Concatenate with previous context buffer
        let xs = match &self.state_prev_xs {
            None => xs,
            Some(prev) => Tensor::cat(&[prev, &xs], D::Minus1)?,
        };
        let seq_len = xs.dim(D::Minus1)?;

        // Calculate how many output frames we can produce
        // With stride=1, each input sample produces one output (after initial padding)
        let num_frames = if seq_len >= k_eff {
            (seq_len - k_eff) / self.stride + 1
        } else {
            0
        };

        if num_frames > 0 {
            // Keep the last (k_eff - stride) samples as context for next frame
            // This is the sliding window approach
            let keep_from = seq_len.saturating_sub(context_len);
            if context_len > 0 && keep_from < seq_len {
                self.state_prev_xs = Some(xs.narrow(D::Minus1, keep_from, context_len)?);
            } else {
                self.state_prev_xs = None;
            }

            // Run conv with NO padding (we've handled it manually)
            let out = xs.conv1d(&self.weight, 0, self.stride, self.dilation, self.groups)?;

            if let Some(bias) = &self.bias {
                let bias = bias.reshape((1, (), 1))?;
                Ok(StreamTensor::from_tensor(out.broadcast_add(&bias)?))
            } else {
                Ok(StreamTensor::from_tensor(out))
            }
        } else {
            // Not enough samples yet - buffer for later
            self.state_prev_xs = Some(xs);
            Ok(StreamTensor::empty())
        }
    }

    fn reset_state(&mut self) {
        self.state_prev_xs = None;
        self.is_first_frame = true;
    }
}

/// Streamable Transposed 1D Convolution (upsampling) with overlap-add
///
/// This implements frame-by-frame streaming for ConvTranspose1d where:
/// - Each output frame overlaps with neighbors due to kernel > stride
/// - Overlap regions are accumulated (overlap-add algorithm)
/// - Bias is subtracted before storing partial, added after combining
///
/// Pattern from Kyutai Moshi: rust/moshi-core/src/conv.rs
#[derive(Debug)]
pub struct StreamableConvTranspose1d {
    /// Convolution weights [in_channels, out_channels/groups, kernel_size]
    weight: Tensor,
    /// Optional bias [out_channels]
    bias: Option<Tensor>,
    /// Kernel size
    kernel_size: usize,
    /// Stride (upsampling factor)
    stride: usize,
    /// Groups for grouped convolution
    groups: usize,

    // Streaming state
    /// Partial output buffer (overlap region from previous frame)
    state_partial: Option<Tensor>,
}

impl StreamableConvTranspose1d {
    /// Create from weight tensors
    pub fn from_weights(
        weight: Tensor,
        bias: Option<Tensor>,
        kernel_size: usize,
        stride: usize,
        groups: usize,
    ) -> Self {
        Self {
            weight,
            bias,
            kernel_size,
            stride,
            groups,
            state_partial: None,
        }
    }

    /// Create from a VarBuilder
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((in_channels, out_channels / groups, kernel_size), "weight")?;
        let bias = vb.get(out_channels, "bias").ok();

        Ok(Self::from_weights(weight, bias, kernel_size, stride, groups))
    }

    /// Batch forward (non-streaming)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Calculate output padding to match causal behavior
        // For kernel_size=32, stride=16: overlap = 32 - 16 = 16
        // Python uses padding=stride//2 for "same" behavior
        let padding = self.stride / 2;
        let output_padding = 0;

        let out = x.conv_transpose1d(
            &self.weight,
            padding,
            output_padding,
            self.stride,
            1, // dilation
            self.groups,
        )?;

        if let Some(bias) = &self.bias {
            let bias = bias.reshape((1, (), 1))?;
            out.broadcast_add(&bias)
        } else {
            Ok(out)
        }
    }
}

impl StreamingModule for StreamableConvTranspose1d {
    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        let xs = match xs.as_option() {
            None => return Ok(StreamTensor::empty()),
            Some(xs) => xs,
        };

        // Run conv_transpose with NO padding (we handle overlap manually)
        let ys = xs.conv_transpose1d(
            &self.weight,
            0, // padding
            0, // output_padding
            self.stride,
            1, // dilation
            self.groups,
        )?;

        let out_len = ys.dim(D::Minus1)?;
        let overlap = self.kernel_size - self.stride;

        // Add bias before overlap-add logic
        let ys = if let Some(bias) = &self.bias {
            let bias = bias.reshape((1, (), 1))?;
            ys.broadcast_add(&bias)?
        } else {
            ys
        };

        // Handle overlap with previous frame
        let ys = match &self.state_partial {
            None => ys,
            Some(prev_partial) => {
                // Add previous partial to left edge of current output
                let left = ys.narrow(D::Minus1, 0, overlap)?;
                let left = left.add(prev_partial)?;
                let right = ys.narrow(D::Minus1, overlap, out_len - overlap)?;
                Tensor::cat(&[&left, &right], D::Minus1)?
            },
        };

        if overlap > 0 && out_len > overlap {
            // Store right edge (without bias) as partial for next frame
            // The key insight from Kyutai: subtract bias before storing,
            // so when we add it back next frame, the bias is applied correctly once
            let partial = ys.narrow(D::Minus1, out_len - overlap, overlap)?;
            let partial = if let Some(bias) = &self.bias {
                let bias = bias.reshape((1, (), 1))?;
                partial.broadcast_sub(&bias)?
            } else {
                partial
            };
            self.state_partial = Some(partial);

            // Return all but the overlap region
            let out = ys.narrow(D::Minus1, 0, out_len - overlap)?;
            Ok(StreamTensor::from_tensor(out))
        } else {
            // No overlap case (stride >= kernel)
            self.state_partial = None;
            Ok(StreamTensor::from_tensor(ys))
        }
    }

    fn reset_state(&mut self) {
        self.state_partial = None;
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_streamable_conv1d_causal_matches_batch() -> Result<()> {
        let device = Device::Cpu;

        // Create random weights
        let in_ch = 4;
        let out_ch = 8;
        let kernel = 3;
        let stride = 1;
        let dilation = 1;

        let weight = Tensor::randn(0f32, 1., (out_ch, in_ch, kernel), &device)?;
        let bias = Tensor::randn(0f32, 1., out_ch, &device)?;

        let conv = StreamableConv1d::from_weights(
            weight.clone(),
            Some(bias.clone()),
            kernel,
            stride,
            dilation,
            1,
            PadMode::Constant,
        );

        // Create input
        let input = Tensor::randn(0f32, 1., (1, in_ch, 10), &device)?;

        // Batch causal forward (reference)
        let batch_out = conv.forward_causal(&input)?;

        // Streaming frame-by-frame (with Constant mode to match zero-padded batch)
        let mut conv_streaming =
            StreamableConv1d::from_weights(weight, Some(bias), kernel, stride, dilation, 1, PadMode::Constant);

        let mut outputs = vec![];
        for i in 0..10 {
            let frame = input.narrow(2, i, 1)?;
            let st = StreamTensor::from_tensor(frame);
            if let Some(out) = conv_streaming.step(&st)?.as_option() {
                outputs.push(out.clone());
            }
        }
        let streaming_out = Tensor::cat(&outputs, 2)?;

        // Compare
        let diff = (&batch_out - &streaming_out)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "Streaming doesn't match causal batch: diff={}", diff);

        Ok(())
    }

    #[test]
    fn test_streamable_conv_transpose1d_overlap_add() -> Result<()> {
        let device = Device::Cpu;

        // Create simple weights for predictable behavior
        let in_ch = 2;
        let out_ch = 4;
        let kernel = 4;
        let stride = 2;

        let weight = Tensor::ones((in_ch, out_ch, kernel), DType::F32, &device)?;

        let mut conv = StreamableConvTranspose1d::from_weights(weight, None, kernel, stride, 1);

        // Process two frames
        let frame1 = Tensor::ones((1, in_ch, 1), DType::F32, &device)?;
        let frame2 = Tensor::ones((1, in_ch, 1), DType::F32, &device)?;

        let out1 = conv.step(&StreamTensor::from_tensor(frame1))?;
        let out2 = conv.step(&StreamTensor::from_tensor(frame2))?;

        // First frame should produce stride outputs
        assert_eq!(out1.seq_len(D::Minus1)?, stride);
        // Second frame should also produce stride outputs
        assert_eq!(out2.seq_len(D::Minus1)?, stride);

        Ok(())
    }
}
