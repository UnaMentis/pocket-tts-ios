//! Rotary Position Embeddings (RoPE)
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{Device, Result, Tensor};

/// Rotary Position Embedding
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
    dim: usize,
    max_seq_len: usize,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32, device: &Device) -> Result<Self> {
        let inv_freq = Self::compute_inv_freq(dim, base, device)?;
        let (cos_cache, sin_cache) = Self::compute_cache(&inv_freq, max_seq_len)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            dim,
            max_seq_len,
        })
    }

    fn compute_inv_freq(dim: usize, base: f32, device: &Device) -> Result<Tensor> {
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
            .collect();

        Tensor::from_vec(inv_freq, (half_dim,), device)
    }

    fn compute_cache(inv_freq: &Tensor, max_seq_len: usize) -> Result<(Tensor, Tensor)> {
        let device = inv_freq.device();
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;

        // Outer product: positions @ inv_freq.T -> [max_seq_len, half_dim]
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;

        // cos/sin have shape [seq, half_dim] - NOT doubled
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok((cos_cache, sin_cache))
    }

    /// Apply rotary embeddings to query and key tensors
    /// Input shape: [batch, seq, num_heads, head_dim]
    pub fn forward(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let end = offset + seq_len;

        if end > self.max_seq_len {
            return Err(candle_core::Error::Msg(format!(
                "Sequence length {} exceeds max {}",
                end, self.max_seq_len
            )));
        }

        // cos/sin have shape [seq, half_dim]
        let cos = self.cos_cache.narrow(0, offset, seq_len)?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?;

        let q_rotated = self.apply_rotary(q, &cos, &sin)?;
        let k_rotated = self.apply_rotary(k, &cos, &sin)?;

        Ok((q_rotated, k_rotated))
    }

    fn apply_rotary(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // x has shape [batch, seq, heads, head_dim]
        let (batch, seq, heads, head_dim) = x.dims4()?;
        let half_dim = head_dim / 2;

        // Kyutai Pocket uses INTERLEAVED real/imaginary pairs:
        // [x0, x1, x2, x3, ...] -> [(x0,x1), (x2,x3), ...] where first is real, second is imaginary
        // Reshape to [batch, seq, heads, half_dim, 2] to access pairs
        let x = x.reshape((batch, seq, heads, half_dim, 2))?;

        // Extract real (even indices) and imaginary (odd indices) components
        let xr = x.narrow(4, 0, 1)?.squeeze(4)?;  // [batch, seq, heads, half_dim]
        let xi = x.narrow(4, 1, 1)?.squeeze(4)?;  // [batch, seq, heads, half_dim]

        // cos/sin have shape [seq, half_dim]
        // Reshape to [1, seq, 1, half_dim] for broadcasting with [batch, seq, heads, half_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        // Complex rotation: (xr + i*xi) * (cos + i*sin) = (xr*cos - xi*sin) + i*(xr*sin + xi*cos)
        let rotated_r = (xr.broadcast_mul(&cos)? - xi.broadcast_mul(&sin)?)?;
        let rotated_i = (xr.broadcast_mul(&sin)? + xi.broadcast_mul(&cos)?)?;

        // Stack back to interleaved format: [(r0,i0), (r1,i1), ...]
        let rotated_r = rotated_r.unsqueeze(4)?;  // [batch, seq, heads, half_dim, 1]
        let rotated_i = rotated_i.unsqueeze(4)?;  // [batch, seq, heads, half_dim, 1]
        let stacked = Tensor::cat(&[&rotated_r, &rotated_i], 4)?;  // [batch, seq, heads, half_dim, 2]

        // Reshape back to [batch, seq, heads, head_dim]
        stacked.reshape((batch, seq, heads, head_dim))
    }
}
