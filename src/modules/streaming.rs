//! Streaming infrastructure for frame-by-frame audio processing
//!
//! This module provides core types and traits for streaming neural network layers.
//! Inspired by Kyutai's Moshi Rust implementation.

use candle_core::{shape::Dim, Result, Tensor, D};

/// Wrapper for Option<Tensor> to handle empty states in streaming pipelines.
///
/// StreamTensor represents data flowing through a streaming pipeline where
/// some steps may not produce output on every call (e.g., when buffering).
#[derive(Debug, Clone)]
pub struct StreamTensor(Option<Tensor>);

impl StreamTensor {
    /// Create an empty StreamTensor (no data)
    pub fn empty() -> Self {
        Self(None)
    }

    /// Create a StreamTensor from a Tensor
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self(Some(tensor))
    }

    /// Get the inner tensor as an Option reference
    pub fn as_option(&self) -> Option<&Tensor> {
        self.0.as_ref()
    }

    /// Check if this StreamTensor is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// Unwrap the inner tensor, panicking if empty
    pub fn unwrap(self) -> Tensor {
        self.0.expect("StreamTensor is empty")
    }

    /// Unwrap the inner tensor or return a default
    pub fn unwrap_or(self, default: Tensor) -> Tensor {
        self.0.unwrap_or(default)
    }

    /// Concatenate two StreamTensors along a dimension
    ///
    /// If either is empty, returns the other (or empty if both empty)
    pub fn cat2(&self, rhs: &Self, dim: D) -> Result<Self> {
        match (&self.0, &rhs.0) {
            (None, None) => Ok(Self::empty()),
            (Some(t), None) => Ok(Self::from_tensor(t.clone())),
            (None, Some(t)) => Ok(Self::from_tensor(t.clone())),
            (Some(lhs), Some(rhs)) => {
                let cat = Tensor::cat(&[lhs, rhs], dim)?;
                Ok(Self::from_tensor(cat))
            },
        }
    }

    /// Split at position, returns (left, right)
    pub fn split(&self, dim: D, lhs_len: usize) -> Result<(Self, Self)> {
        match &self.0 {
            None => Ok((Self::empty(), Self::empty())),
            Some(t) => {
                let total = t.dim(dim)?;
                if lhs_len >= total {
                    Ok((Self::from_tensor(t.clone()), Self::empty()))
                } else {
                    let left = t.narrow(dim, 0, lhs_len)?;
                    let right = t.narrow(dim, lhs_len, total - lhs_len)?;
                    Ok((Self::from_tensor(left), Self::from_tensor(right)))
                }
            },
        }
    }

    /// Narrow slice along a dimension
    pub fn narrow(&self, dim: D, offset: usize, len: usize) -> Result<Self> {
        match &self.0 {
            None => Ok(Self::empty()),
            Some(t) => {
                let narrowed = t.narrow(dim, offset, len)?;
                Ok(Self::from_tensor(narrowed))
            },
        }
    }

    /// Get sequence length along a dimension
    pub fn seq_len(&self, dim: D) -> Result<usize> {
        match &self.0 {
            None => Ok(0),
            Some(t) => t.dim(dim),
        }
    }

    /// Apply a function to the inner tensor if present
    pub fn map<F>(&self, f: F) -> Result<Self>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        match &self.0 {
            None => Ok(Self::empty()),
            Some(t) => {
                let result = f(t)?;
                Ok(Self::from_tensor(result))
            },
        }
    }

    /// Transpose dimensions if tensor is present
    pub fn transpose(&self, d1: D, d2: D) -> Result<Self> {
        self.map(|t| t.transpose(d1, d2))
    }
}

impl From<()> for StreamTensor {
    fn from(_: ()) -> Self {
        Self::empty()
    }
}

impl From<Tensor> for StreamTensor {
    fn from(t: Tensor) -> Self {
        Self::from_tensor(t)
    }
}

impl From<Option<Tensor>> for StreamTensor {
    fn from(opt: Option<Tensor>) -> Self {
        Self(opt)
    }
}

/// Standard interface for streaming neural network layers.
///
/// Streaming modules process data frame-by-frame, maintaining internal state
/// between calls to handle dependencies across frames (e.g., convolution buffers,
/// KV caches, overlap-add buffers).
pub trait StreamingModule {
    /// Process a single frame/chunk of input
    ///
    /// May return empty if buffering (e.g., waiting for more input to produce output)
    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor>;

    /// Reset all internal streaming state
    ///
    /// Call this between utterances or when starting a new sequence
    fn reset_state(&mut self);
}

/// Helper trait for padding tensors
pub trait TensorPadding {
    /// Pad with zeros on the left (for causal convolutions)
    fn pad_zeros_left(&self, dim: D, amount: usize) -> Result<Tensor>;

    /// Pad with zeros on the right
    fn pad_zeros_right(&self, dim: D, amount: usize) -> Result<Tensor>;
}

impl TensorPadding for Tensor {
    fn pad_zeros_left(&self, dim: D, amount: usize) -> Result<Tensor> {
        if amount == 0 {
            return Ok(self.clone());
        }

        // Create padding shape: same as input but with `amount` in the target dim
        let mut pad_shape: Vec<usize> = self.dims().to_vec();
        let dim_idx = dim.to_index(self.shape(), "pad_zeros_left")?;
        pad_shape[dim_idx] = amount;

        let zeros = Tensor::zeros(pad_shape.as_slice(), self.dtype(), self.device())?;
        Tensor::cat(&[&zeros, self], dim)
    }

    fn pad_zeros_right(&self, dim: D, amount: usize) -> Result<Tensor> {
        if amount == 0 {
            return Ok(self.clone());
        }

        // Create padding shape: same as input but with `amount` in the target dim
        let mut pad_shape: Vec<usize> = self.dims().to_vec();
        let dim_idx = dim.to_index(self.shape(), "pad_zeros_right")?;
        pad_shape[dim_idx] = amount;

        let zeros = Tensor::zeros(pad_shape.as_slice(), self.dtype(), self.device())?;
        Tensor::cat(&[self, &zeros], dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_stream_tensor_empty() {
        let st = StreamTensor::empty();
        assert!(st.is_empty());
        assert!(st.as_option().is_none());
    }

    #[test]
    fn test_stream_tensor_from_tensor() -> Result<()> {
        let t = Tensor::zeros(&[1, 2, 3], candle_core::DType::F32, &Device::Cpu)?;
        let st = StreamTensor::from_tensor(t.clone());
        assert!(!st.is_empty());
        assert!(st.as_option().is_some());
        Ok(())
    }

    #[test]
    fn test_stream_tensor_cat2() -> Result<()> {
        let t1 = Tensor::ones(&[1, 2, 3], candle_core::DType::F32, &Device::Cpu)?;
        let t2 = Tensor::ones(&[1, 2, 4], candle_core::DType::F32, &Device::Cpu)?;

        let st1 = StreamTensor::from_tensor(t1);
        let st2 = StreamTensor::from_tensor(t2);

        let cat = st1.cat2(&st2, D::Minus1)?;
        assert_eq!(cat.seq_len(D::Minus1)?, 7);

        // Test with empty
        let empty = StreamTensor::empty();
        let cat_with_empty = st1.cat2(&empty, D::Minus1)?;
        assert_eq!(cat_with_empty.seq_len(D::Minus1)?, 3);

        Ok(())
    }

    #[test]
    fn test_stream_tensor_split() -> Result<()> {
        let t = Tensor::ones(&[1, 2, 10], candle_core::DType::F32, &Device::Cpu)?;
        let st = StreamTensor::from_tensor(t);

        let (left, right) = st.split(D::Minus1, 3)?;
        assert_eq!(left.seq_len(D::Minus1)?, 3);
        assert_eq!(right.seq_len(D::Minus1)?, 7);

        Ok(())
    }

    #[test]
    fn test_pad_zeros_left() -> Result<()> {
        let t = Tensor::ones(&[1, 2, 5], candle_core::DType::F32, &Device::Cpu)?;
        let padded = t.pad_zeros_left(D::Minus1, 3)?;

        assert_eq!(padded.dims(), &[1, 2, 8]);

        // Check that first 3 elements are zeros
        let first_3 = padded.narrow(2, 0, 3)?;
        let sum: f32 = first_3.sum_all()?.to_scalar()?;
        assert_eq!(sum, 0.0);

        // Check that last 5 elements are ones
        let last_5 = padded.narrow(2, 3, 5)?;
        let sum: f32 = last_5.sum_all()?.to_scalar()?;
        assert_eq!(sum, 10.0); // 1 * 2 * 5 = 10

        Ok(())
    }
}
