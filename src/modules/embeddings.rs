//! Embedding modules for text and voice
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

/// Text token embeddings
#[derive(Debug)]
pub struct TextEmbedding {
    embedding: Embedding,
    hidden_size: usize,
}

impl TextEmbedding {
    pub fn new(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, hidden_size, vb)?;
        Ok(Self { embedding, hidden_size })
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let result = self.embedding.forward(token_ids)?;

        // Debug: print first few embedding values
        if let Ok(flat) = result.flatten_all() {
            if let Ok(vals) = flat.to_vec1::<f32>() {
                let first8: Vec<f32> = vals.iter().take(8).cloned().collect();
                let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
                let std: f32 = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
                eprintln!("[TextEmbed] first 8: {:?}", first8);
                eprintln!("[TextEmbed] mean: {:.6}, std: {:.4}", mean, std);
            }
        }

        Ok(result)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// A precomputed per-layer transformer KV-cache voice state (Pocket TTS v2 voice format).
///
/// v2 voice files ship the already-computed self-attention KV cache (with `bos_before_voice`
/// and the speaker projection baked in offline) instead of a raw embedding sequence. Each
/// layer's `(k, v)` is stored in our cache layout `[batch, num_heads, seq, head_dim]`.
#[derive(Debug, Clone)]
pub struct VoiceKvState {
    /// Per-layer `(k, v)` tensors in `[batch, num_heads, seq, head_dim]` layout.
    pub layers: Vec<(Tensor, Tensor)>,
    /// Number of preloaded sequence positions (e.g. 126 = 1 BOS + 125 voice frames).
    pub seq_len: usize,
}

/// Voice embedding (speaker identity)
#[derive(Debug, Clone)]
pub struct VoiceEmbedding {
    embedding: Tensor,
    voice_dim: usize,
    /// v2 voice state (precomputed KV cache). When set, the FlowLM uses this directly
    /// instead of running `embedding` through the transformer.
    kv_state: Option<VoiceKvState>,
}

impl VoiceEmbedding {
    /// Load voice embedding from safetensors file
    pub fn from_file(path: &std::path::Path, device: &Device) -> Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data, device)
    }

    /// Load voice embedding from bytes
    pub fn from_bytes(data: &[u8], device: &Device) -> Result<Self> {
        let tensors = safetensors::SafeTensors::deserialize(data)?;

        // v1 format: a raw voice embedding sequence (Kyutai uses "audio_prompt").
        if let Ok(embedding_data) = tensors
            .tensor("audio_prompt")
            .or_else(|_| tensors.tensor("embedding"))
            .or_else(|_| tensors.tensor("voice"))
            .or_else(|_| tensors.tensor("speaker"))
        {
            let shape = embedding_data.shape();
            // Kyutai voice embeddings are [1, seq_len, dim] where dim is typically 1024
            let voice_dim = shape.last().copied().unwrap_or(1024);

            let candle_dtype = convert_safetensors_dtype(embedding_data.dtype())?;
            let embedding = Tensor::from_raw_buffer(embedding_data.data(), candle_dtype, shape, device)?;

            // Squeeze out batch dimension if present: [1, seq, dim] -> [seq, dim]
            let embedding = if shape.len() == 3 && shape[0] == 1 {
                embedding.squeeze(0)?
            } else {
                embedding
            };

            return Ok(Self {
                embedding,
                voice_dim,
                kv_state: None,
            });
        }

        // v2 format: a precomputed per-layer KV-cache state
        // ("transformer.layers.{i}.self_attn/cache" of shape [2, batch, seq, heads, head_dim]).
        if tensors.tensor("transformer.layers.0.self_attn/cache").is_ok() {
            let mut layers: Vec<(Tensor, Tensor)> = Vec::new();
            let mut idx = 0usize;
            loop {
                let key = format!("transformer.layers.{}.self_attn/cache", idx);
                let Ok(cache_t) = tensors.tensor(&key) else {
                    break;
                };
                let shape = cache_t.shape(); // [2, batch, seq, heads, head_dim]
                let dtype = convert_safetensors_dtype(cache_t.dtype())?;
                let cache = Tensor::from_raw_buffer(cache_t.data(), dtype, shape, device)?;

                // Split K (index 0) and V (index 1) and drop that leading dim.
                let k = cache.narrow(0, 0, 1)?.squeeze(0)?; // [batch, seq, heads, head_dim]
                let v = cache.narrow(0, 1, 1)?.squeeze(0)?;
                // Move heads ahead of seq to match our cache layout [batch, heads, seq, head_dim].
                let k = k.transpose(1, 2)?.contiguous()?;
                let v = v.transpose(1, 2)?.contiguous()?;
                layers.push((k, v));
                idx += 1;
            }

            if layers.is_empty() {
                return Err(candle_core::Error::Msg("v2 voice state contained no transformer layers".into()));
            }

            let seq_len = layers[0].0.dim(2)?;
            let num_heads = layers[0].0.dim(1)?;
            let head_dim = layers[0].0.dim(3)?;
            let voice_dim = num_heads * head_dim;

            // Placeholder embedding; unused on the v2 (KV-state) path.
            let embedding = Tensor::zeros((1, voice_dim), candle_core::DType::F32, device)?;

            return Ok(Self {
                embedding,
                voice_dim,
                kv_state: Some(VoiceKvState { layers, seq_len }),
            });
        }

        Err(candle_core::Error::Msg(
            "Voice file contained neither a v1 embedding (audio_prompt) nor a v2 KV state".into(),
        ))
    }

    /// The precomputed v2 KV-cache voice state, if this is a v2 voice file.
    pub fn kv_state(&self) -> Option<&VoiceKvState> {
        self.kv_state.as_ref()
    }

    /// Create voice embedding from raw tensor
    /// Expects shape [seq, dim] or [dim]
    pub fn from_tensor(embedding: Tensor) -> Result<Self> {
        let voice_dim = embedding.dim(candle_core::D::Minus1)?;
        // Ensure embedding is at least 2D: [seq, dim]
        let embedding = if embedding.dims().len() == 1 {
            embedding.unsqueeze(0)? // [dim] -> [1, dim]
        } else {
            embedding
        };
        Ok(Self {
            embedding,
            voice_dim,
            kv_state: None,
        })
    }

    /// Get the embedding tensor
    pub fn embedding(&self) -> &Tensor {
        &self.embedding
    }

    /// Get voice dimension
    pub fn voice_dim(&self) -> usize {
        self.voice_dim
    }

    /// Expand embedding to match batch size and text sequence length
    /// The voice embedding is [prompt_seq, dim], we mean-pool it and expand to [batch, text_seq, dim]
    /// This allows the voice embedding to condition all positions in the text sequence
    pub fn expand_to_seq(&self, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        // Mean-pool across the prompt sequence dimension: [prompt_seq, dim] -> [dim]
        let pooled = self.embedding.mean(0)?;

        // Add batch and sequence dimensions: [dim] -> [1, 1, dim]
        let expanded = pooled.unsqueeze(0)?.unsqueeze(0)?;

        // Expand to [batch, seq_len, dim] to match text embeddings
        expanded.expand(&[batch_size, seq_len, self.voice_dim])
    }

    /// Get the prompt sequence length (number of audio prompt frames)
    pub fn prompt_seq_len(&self) -> Result<usize> {
        self.embedding.dim(0)
    }
}

/// Voice embedding bank (all 8 built-in voices)
#[derive(Debug)]
pub struct VoiceBank {
    voices: Vec<VoiceEmbedding>,
    voice_dim: usize,
}

impl VoiceBank {
    pub fn new(voice_dim: usize) -> Self {
        Self {
            voices: Vec::with_capacity(8),
            voice_dim,
        }
    }

    /// Load all voices from a directory
    pub fn load_from_dir(dir: &std::path::Path, device: &Device) -> Result<Self> {
        let voice_names = [
            "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
        ];

        let mut voices = Vec::with_capacity(8);
        let mut voice_dim = 512; // Default

        for name in &voice_names {
            let path = dir.join(format!("{}.safetensors", name));
            if path.exists() {
                let voice = VoiceEmbedding::from_file(&path, device)?;
                voice_dim = voice.voice_dim();
                voices.push(voice);
            }
        }

        Ok(Self { voices, voice_dim })
    }

    /// Get voice by index
    pub fn get(&self, index: usize) -> Option<&VoiceEmbedding> {
        self.voices.get(index)
    }

    /// Number of loaded voices
    pub fn len(&self) -> usize {
        self.voices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }

    pub fn voice_dim(&self) -> usize {
        self.voice_dim
    }
}

/// Convert safetensors dtype to candle dtype
fn convert_safetensors_dtype(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::I64 => Ok(DType::I64),
        safetensors::Dtype::U32 => Ok(DType::U32),
        safetensors::Dtype::U8 => Ok(DType::U8),
        _ => Err(candle_core::Error::Msg(format!("Unsupported dtype: {:?}", dtype))),
    }
}
