//! Rotary Position Embeddings (RoPE)
//!
//! Portions of this file derived from:
//! <https://github.com/babybirdprd/pocket-tts>
//! Licensed under MIT

use candle_core::{Device, Result, Tensor};

/// Rotary Position Embedding
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
    #[allow(dead_code)] // Stored for future use in cache invalidation
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
        let inv_freq: Vec<f32> = (0..half_dim).map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32)).collect();

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
    /// Uses candle_nn::rotary_emb::rope_i() for interleaved RoPE (matches Kyutai's Moshi Rust)
    pub fn forward(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let end = offset + seq_len;

        if end > self.max_seq_len {
            return Err(candle_core::Error::Msg(format!(
                "Sequence length {} exceeds max {}",
                end, self.max_seq_len
            )));
        }

        // cos/sin have shape [seq, half_dim] — matches rope_i's expected (seq_len, n_embd/2)
        let cos = self.cos_cache.narrow(0, offset, seq_len)?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?;

        // rope_i expects (B, H, T, D), our input is (B, T, H, D)
        let q_bhtd = q.transpose(1, 2)?.contiguous()?;
        let k_bhtd = k.transpose(1, 2)?.contiguous()?;

        let q_rotated = candle_nn::rotary_emb::rope_i(&q_bhtd, &cos, &sin)?;
        let k_rotated = candle_nn::rotary_emb::rope_i(&k_bhtd, &cos, &sin)?;

        // Transpose back to (B, T, H, D)
        let q_rotated = q_rotated.transpose(1, 2)?;
        let k_rotated = k_rotated.transpose(1, 2)?;

        Ok((q_rotated, k_rotated))
    }
}
