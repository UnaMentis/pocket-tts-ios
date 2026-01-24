//! Layer normalization modules
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// RMS Layer Normalization (used in modern transformers)
#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((hidden_size,), "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn load(_hidden_size: usize, eps: f64, weight: Tensor) -> Self {
        Self { weight, eps }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;

        // Calculate RMS
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        // Scale by weight
        let result = x_normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;

        result.to_dtype(dtype)
    }
}

/// Standard Layer Normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    /// Create a LayerNorm with bias (Kyutai Pocket style)
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((hidden_size,), "weight")?;
        let bias = Some(vb.get((hidden_size,), "bias")?);
        Ok(Self { weight, bias, eps })
    }

    /// Create a LayerNorm with configurable bias
    pub fn new_with_bias(hidden_size: usize, eps: f64, has_bias: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((hidden_size,), "weight")?;
        let bias = if has_bias {
            Some(vb.get((hidden_size,), "bias")?)
        } else {
            None
        };
        Ok(Self { weight, bias, eps })
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        let mut result = x_normed.broadcast_mul(&self.weight)?;

        if let Some(ref bias) = self.bias {
            result = result.broadcast_add(bias)?;
        }

        Ok(result)
    }
}
