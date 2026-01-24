//! Multi-head attention modules
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::rotary::RotaryEmbedding;

/// Multi-Head Attention with optional KV caching
#[derive(Debug)]
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MultiHeadAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        let q_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rotary: Option<&RotaryEmbedding>,
        kv_cache: Option<&mut KVCache>,
        causal_mask: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Apply rotary embeddings BEFORE transpose (RoPE expects [batch, seq, heads, dim])
        let (q, k) = if let Some(rope) = rotary {
            let offset = kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
            rope.forward(&q, &k, offset)?
        } else {
            (q, k)
        };

        // Transpose to [batch, num_heads, seq, head_dim] for attention
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Update KV cache if provided
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(k, v)?
        } else {
            (k, v)
        };

        // Attention scores: Q @ K^T
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * self.scale as f64)?;

        // Apply causal mask if needed
        let attn_weights = if causal_mask {
            let mask = self.create_causal_mask(seq_len, k.dim(2)?, x.device())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

        // Weighted sum of values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }

    fn create_causal_mask(&self, q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j <= i + (kv_len - q_len) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
    }
}

/// KV Cache for efficient autoregressive generation
#[derive(Debug)]
pub struct KVCache {
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            k_cache: None,
            v_cache: None,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.k_cache.as_ref().map(|k| k.dim(2).unwrap_or(0)).unwrap_or(0)
    }

    pub fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let (k_out, v_out) = match (&self.k_cache, &self.v_cache) {
            (Some(k_cache), Some(v_cache)) => {
                let k_new = Tensor::cat(&[k_cache, &k], 2)?;
                let v_new = Tensor::cat(&[v_cache, &v], 2)?;
                (k_new, v_new)
            }
            _ => (k, v),
        };

        self.k_cache = Some(k_out.clone());
        self.v_cache = Some(v_out.clone());

        Ok((k_out, v_out))
    }

    pub fn clear(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Causal Self-Attention (convenience wrapper)
pub type CausalSelfAttention = MultiHeadAttention;

/// Fused Multi-Head Attention (Kyutai Pocket style)
/// Uses combined in_proj for Q/K/V instead of separate projections
#[derive(Debug)]
pub struct FusedMultiHeadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl FusedMultiHeadAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        // Kyutai Pocket uses in_proj (combined QKV) and out_proj - NO BIAS
        let in_proj = candle_nn::linear_no_bias(hidden_size, hidden_size * 3, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear_no_bias(hidden_size, hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        rotary: Option<&RotaryEmbedding>,
        kv_cache: Option<&mut KVCache>,
        causal_mask: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = x.dims3()?;

        // Combined projection to get Q, K, V together
        let qkv = self.in_proj.forward(x)?;

        // Split into Q, K, V (each has hidden_size dimension)
        let q = qkv.narrow(2, 0, hidden_size)?;
        let k = qkv.narrow(2, hidden_size, hidden_size)?;
        let v = qkv.narrow(2, hidden_size * 2, hidden_size)?;

        // Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // DEBUG: Log Q, K, V before RoPE
        // Counter: 0-5 = voice layers, 6-11 = text layers, 12-17 = step 0 layers
        static DEBUG_ATTN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let attn_call = DEBUG_ATTN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let is_text_l0 = attn_call == 6;
        let is_step0_l0 = attn_call == 12;

        if is_text_l0 {
            if let Ok(vals) = q.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] Q head0 first 8 before RoPE: {:?}", &vals[..8.min(vals.len())]);
            }
            if let Ok(vals) = k.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] K head0 first 8 before RoPE: {:?}", &vals[..8.min(vals.len())]);
            }
            if let Ok(vals) = v.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] V head0 first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }
        if is_step0_l0 {
            eprintln!("[Attn-L0-Step0] seq_len={}, hidden_size={}", seq_len, hidden_size);
            if let Ok(vals) = q.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0-Step0] Q head0 first 8 before RoPE: {:?}", &vals[..8.min(vals.len())]);
            }
            if let Ok(vals) = k.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0-Step0] K head0 first 8 before RoPE: {:?}", &vals[..8.min(vals.len())]);
            }
            if let Ok(vals) = v.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0-Step0] V head0 first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }

        // Apply rotary embeddings BEFORE transpose (RoPE expects [batch, seq, heads, dim])
        // This matches Python which applies RoPE before transposing for attention
        let (q, k) = if let Some(rope) = rotary {
            let offset = kv_cache.as_ref().map(|c| c.seq_len()).unwrap_or(0);
            if is_text_l0 {
                eprintln!("[Attn-L0] RoPE offset: {}", offset);
            }
            if is_step0_l0 {
                eprintln!("[Attn-L0-Step0] RoPE offset: {} (should be 132 = 125 voice + 7 text)", offset);
            }
            rope.forward(&q, &k, offset)?
        } else {
            (q, k)
        };

        // DEBUG: Log Q after RoPE
        if is_text_l0 {
            if let Ok(vals) = q.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] Q head0 first 8 after RoPE: {:?}", &vals[..8.min(vals.len())]);
            }
        }
        if is_step0_l0 {
            if let Ok(vals) = q.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0-Step0] Q head0 first 8 after RoPE: {:?}", &vals[..8.min(vals.len())]);
            }
        }

        // Transpose to [batch, num_heads, seq, head_dim] for attention computation
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Update KV cache if provided
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(k, v)?
        } else {
            (k, v)
        };

        // DEBUG: Log K cache state and attention computation
        if is_step0_l0 {
            eprintln!("[Attn-L0-Step0] K shape after cache: {:?} (should be [1,16,133,64])", k.dims());
            eprintln!("[Attn-L0-Step0] Q shape: {:?} (should be [1,16,1,64])", q.dims());
        }
        if is_text_l0 {
            eprintln!("[Attn-L0] K shape after cache: {:?}", k.dims());
            eprintln!("[Attn-L0] Q shape: {:?}", q.dims());
            // K should have shape [1, 16, 132, 64] (125 voice + 7 text = 132 total in cache)
            // Q should have shape [1, 16, 7, 64] (7 text tokens)

            // Log K values from position 0 (first voice position in cache)
            if let Ok(k_vals) = k.narrow(2, 0, 1)?.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] K[pos=0] head0 first 8 (voice): {:?}", &k_vals[..8.min(k_vals.len())]);
            }

            // Log K values from position 125 (first text position in cache)
            if let Ok(k_vals) = k.narrow(2, 125, 1)?.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] K[pos=125] head0 first 8 (text): {:?}", &k_vals[..8.min(k_vals.len())]);
            }

            // Log V values too for completeness
            if let Ok(v_vals) = v.narrow(2, 0, 1)?.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] V[pos=0] head0 first 8 (voice): {:?}", &v_vals[..8.min(v_vals.len())]);
            }
        }

        // Attention scores: Q @ K^T
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;

        // DEBUG: Log raw attention scores before scaling
        if attn_call == 6 {
            // attn_weights shape: [1, 16, 17, 142]
            // Log Q[0] attending to K positions (head 0, first query position)
            if let Ok(scores) = attn_weights.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] Raw attn scores head0 q0 first 8: {:?}", &scores[..8.min(scores.len())]);
                // Also log scores for positions 125-132 (text attending to text)
                if scores.len() > 132 {
                    eprintln!("[Attn-L0] Raw attn scores head0 q0 pos125-132: {:?}", &scores[125..133]);
                }
            }
        }

        let attn_weights = (attn_weights * self.scale as f64)?;

        // Apply causal mask if needed
        let attn_weights = if causal_mask {
            let mask = self.create_causal_mask(seq_len, k.dim(2)?, x.device())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

        // DEBUG: Log softmax attention weights
        if attn_call == 6 {
            if let Ok(probs) = attn_weights.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] Softmax attn probs head0 q0 first 8: {:?}", &probs[..8.min(probs.len())]);
                if probs.len() > 132 {
                    eprintln!("[Attn-L0] Softmax attn probs head0 q0 pos125-132: {:?}", &probs[125..133]);
                }
            }
        }

        // Weighted sum of values
        let attn_output = attn_weights.matmul(&v)?;

        // DEBUG: Log attention output
        if is_text_l0 {
            if let Ok(out) = attn_output.narrow(1, 0, 1)?.narrow(2, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] Attn output head0 q0 first 8: {:?}", &out[..8.min(out.len())]);
            }
        }

        // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        let final_output = self.out_proj.forward(&attn_output)?;

        // DEBUG: Log final attention output (after out_proj)
        if is_text_l0 {
            if let Ok(out) = final_output.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0] FINAL output (after out_proj) first 8: {:?}", &out[..8.min(out.len())]);
            }
        }
        if is_step0_l0 {
            if let Ok(out) = final_output.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Attn-L0-Step0] FINAL output (after out_proj) first 8: {:?}", &out[..8.min(out.len())]);
            }
        }

        Ok(final_output)
    }

    fn create_causal_mask(&self, q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j <= i + (kv_len - q_len) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
    }
}
