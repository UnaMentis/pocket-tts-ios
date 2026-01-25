//! FlowLM Transformer for Pocket TTS
//!
//! 6-layer transformer backbone that generates latent representations
//! from text tokens and voice embeddings. Includes FlowNet for flow
//! matching based latent generation.
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::modules::{
    attention::{FusedMultiHeadAttention, KVCache},
    embeddings::{TextEmbedding, VoiceEmbedding},
    flownet::{FlowNet, FlowNetConfig},
    layer_norm::LayerNorm,
    mlp::SimpleMLP,
    rotary::RotaryEmbedding,
};

/// FlowLM configuration
#[derive(Debug, Clone)]
pub struct FlowLMConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub rms_norm_eps: f64,
    pub latent_dim: usize,
}

impl Default for FlowLMConfig {
    fn default() -> Self {
        Self {
            vocab_size: 4001, // Kyutai Pocket TTS vocabulary size
            hidden_size: 1024,
            intermediate_size: 4096,
            num_layers: 6,
            num_heads: 16,
            max_seq_len: 2048,
            rope_base: 10000.0,
            rms_norm_eps: 1e-5, // Match Python nn.LayerNorm default
            latent_dim: 32,
        }
    }
}

/// Single transformer layer
#[derive(Debug)]
struct TransformerLayer {
    attn: FusedMultiHeadAttention,
    mlp: SimpleMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerLayer {
    fn new(config: &FlowLMConfig, vb: VarBuilder) -> Result<Self> {
        // Kyutai Pocket uses fused in_proj/out_proj attention
        let attn = FusedMultiHeadAttention::new(config.hidden_size, config.num_heads, vb.pp("self_attn"))?;

        // Kyutai Pocket uses simple 2-layer MLP (linear1/linear2)
        let mlp = SimpleMLP::new(
            config.hidden_size,
            config.intermediate_size,
            vb.clone(), // MLP tensors are at layer level, not in "mlp" submodule
        )?;

        // Kyutai Pocket uses norm1/norm2 naming
        let norm1 = LayerNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm1"))?;

        let norm2 = LayerNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm2"))?;

        Ok(Self {
            attn,
            mlp,
            norm1,
            norm2,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &RotaryEmbedding, kv_cache: Option<&mut KVCache>) -> Result<Tensor> {
        // Pre-norm attention (Kyutai Pocket architecture)
        let residual = x;
        let normed = self.norm1.forward(x)?;

        // DEBUG: Log norm1 output for text and step 0
        // Counter: 0-5 = voice layers, 6-11 = text layers, 12-17 = step 0 layers
        static DEBUG_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call_num = DEBUG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Layer 0 text processing is call 6
        let is_text_layer0 = call_num == 6;
        // Layer 0 step 0 (first latent generation) is call 12
        let is_step0_layer0 = call_num == 12;

        if is_text_layer0 {
            if let Ok(n) = normed.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Text] norm1 first 8: {:?}", &n[..8.min(n.len())]);
                let mean: f32 = n.iter().sum::<f32>() / n.len() as f32;
                let std = (n.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n.len() as f32).sqrt();
                eprintln!("[Layer0-Text] norm1 mean: {:.6}, std: {:.6}", mean, std);
            }
        }
        if is_step0_layer0 {
            if let Ok(n) = normed.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Step0] norm1 first 8: {:?}", &n[..8.min(n.len())]);
                let mean: f32 = n.iter().sum::<f32>() / n.len() as f32;
                let std = (n.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n.len() as f32).sqrt();
                eprintln!("[Layer0-Step0] norm1 mean: {:.6}, std: {:.6}", mean, std);
            }
        }

        let attn_out = self.attn.forward(&normed, Some(rotary), kv_cache, true)?;
        let x = (residual + &attn_out)?;

        // DEBUG: Log after first residual add
        if is_text_layer0 {
            if let Ok(vals) = x.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Text] after attn+residual first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }
        if is_step0_layer0 {
            if let Ok(vals) = x.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Step0] after attn+residual first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }

        // Pre-norm MLP
        let residual = &x;
        let x = self.norm2.forward(&x)?;

        // DEBUG: Log after norm2
        if is_text_layer0 {
            if let Ok(vals) = x.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Text] after norm2 first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }
        if is_step0_layer0 {
            if let Ok(vals) = x.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Step0] after norm2 first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }

        let x = self.mlp.forward(&x)?;

        // DEBUG: Log after MLP
        if is_text_layer0 {
            if let Ok(vals) = x.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Text] after MLP first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }
        if is_step0_layer0 {
            if let Ok(vals) = x.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Step0] after MLP first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }

        let output = (residual + x)?;

        // DEBUG: Log final layer output
        if is_text_layer0 {
            if let Ok(vals) = output.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Text] LAYER OUTPUT first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }
        if is_step0_layer0 {
            if let Ok(vals) = output.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>() {
                eprintln!("[Layer0-Step0] LAYER OUTPUT first 8: {:?}", &vals[..8.min(vals.len())]);
            }
        }

        Ok(output)
    }
}

/// FlowLM Transformer with FlowNet
///
/// The Kyutai Pocket architecture uses AUTOREGRESSIVE latent generation:
/// 1. Text tokens are used as prefix/conditioning
/// 2. Starting from BOS embedding, generate latents one at a time
/// 3. Each generated latent is fed back as input to generate the next
/// 4. Continue until EOS is predicted or max length reached
#[derive(Debug)]
pub struct FlowLM {
    config: FlowLMConfig,
    text_embedding: TextEmbedding,
    layers: Vec<TransformerLayer>,
    final_norm: LayerNorm, // Kyutai Pocket uses LayerNorm with bias (not RMSNorm)
    flow_net: FlowNet,
    input_linear: candle_nn::Linear, // Projects latent (32) → hidden (1024)
    out_eos: candle_nn::Linear,      // Predicts EOS from hidden (1024 → 1)
    rotary: RotaryEmbedding,
    kv_caches: Vec<KVCache>,
    device: Device,
    // Latent normalization parameters
    emb_mean: Tensor,
    emb_std: Tensor,
    bos_emb: Tensor,
}

impl FlowLM {
    pub fn new(config: FlowLMConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        // Kyutai Pocket uses conditioner.embed for text embeddings
        let text_embedding = TextEmbedding::new(config.vocab_size, config.hidden_size, vb.pp("conditioner.embed"))?;

        // Kyutai Pocket uses transformer.layers.{i} path
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerLayer::new(&config, vb.pp(format!("transformer.layers.{}", i)))?);
        }

        // Kyutai Pocket uses LayerNorm (with bias) for final normalization
        let final_norm = LayerNorm::new(
            config.hidden_size,
            1e-5, // Python nn.LayerNorm uses eps=1e-5 by default
            vb.pp("out_norm"),
        )?;

        // FlowNet for latent generation via flow matching
        let flownet_config = FlowNetConfig {
            hidden_dim: 512,
            cond_dim: config.hidden_size,
            latent_dim: config.latent_dim,
            num_res_blocks: 6,
            time_embed_dim: 256,
        };
        let flow_net = FlowNet::new(flownet_config, vb.pp("flow_net"))?;

        // Kyutai Pocket uses input_linear to project latent (32) → hidden (1024)
        // This is used to condition on previous latent tokens
        let input_linear = candle_nn::linear_no_bias(config.latent_dim, config.hidden_size, vb.pp("input_linear"))?;

        // EOS prediction layer: hidden (1024) → 1
        let out_eos = candle_nn::linear(config.hidden_size, 1, vb.pp("out_eos"))?;

        let head_dim = config.hidden_size / config.num_heads;
        let rotary = RotaryEmbedding::new(head_dim, config.max_seq_len, config.rope_base, device)?;

        let kv_caches = (0..config.num_layers).map(|_| KVCache::new()).collect();

        // Load latent normalization parameters
        // These are used to denormalize the FlowNet output
        let emb_mean = vb.get((config.latent_dim,), "emb_mean")?;
        let emb_std = vb.get((config.latent_dim,), "emb_std")?;
        let bos_emb = vb.get((config.latent_dim,), "bos_emb")?;

        // Debug: print loaded weights for verification
        if let Ok(vals) = emb_mean.to_vec1::<f32>() {
            eprintln!("[FlowLM] emb_mean first 8: {:?}", &vals[..8.min(vals.len())]);
        }
        if let Ok(vals) = emb_std.to_vec1::<f32>() {
            eprintln!("[FlowLM] emb_std first 8: {:?}", &vals[..8.min(vals.len())]);
        }
        if let Ok(vals) = bos_emb.to_vec1::<f32>() {
            eprintln!("[FlowLM] bos_emb first 8: {:?}", &vals[..8.min(vals.len())]);
            let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
            eprintln!("[FlowLM] bos_emb mean: {:.6}", mean);
        }

        Ok(Self {
            config,
            text_embedding,
            layers,
            final_norm,
            flow_net,
            input_linear,
            out_eos,
            rotary,
            kv_caches,
            device: device.clone(),
            emb_mean,
            emb_std,
            bos_emb,
        })
    }

    /// Forward pass with optional voice conditioning
    /// Returns hidden states (1024-dim) from transformer
    pub fn forward(
        &mut self,
        token_ids: &Tensor,
        voice_embedding: Option<&VoiceEmbedding>,
        use_cache: bool,
    ) -> Result<Tensor> {
        // Get text embeddings
        let mut hidden = self.text_embedding.forward(token_ids)?;

        // Add voice conditioning if provided
        if let Some(voice) = voice_embedding {
            let (batch_size, seq_len, _) = hidden.dims3()?;
            let voice_expanded = voice.expand_to_seq(batch_size, seq_len)?;
            hidden = (hidden + voice_expanded)?;
        }

        // Pass through transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = if use_cache { Some(&mut self.kv_caches[i]) } else { None };
            hidden = layer.forward(&hidden, &self.rotary, cache)?;
        }

        // Final norm - return hidden states for FlowNet to generate latents
        self.final_norm.forward(&hidden)
    }

    /// Generate latents autoregressively from text tokens
    ///
    /// This matches the Python reference generation flow:
    /// 1. FIRST: Process voice embeddings alone (populates KV cache with voice context)
    /// 2. THEN: Process text embeddings (appends to KV cache, sees voice context)
    /// 3. FINALLY: Generate latents autoregressively (each sees voice + text + previous latents)
    ///
    /// The KV cache ordering is critical:
    /// - Positions 0-124: Voice conditioning
    /// - Positions 125-141: Text conditioning
    /// - Positions 142+: Generated latents
    pub fn generate_latents(
        &mut self,
        token_ids: &Tensor,
        voice_embedding: Option<&VoiceEmbedding>,
        num_flow_steps: usize,
        temperature: f32,
    ) -> Result<Tensor> {
        // Reset caches before generation
        self.reset_cache();

        let text_embeddings = self.text_embedding.forward(token_ids)?;
        let (batch_size, _seq_len, _hidden_dim) = text_embeddings.dims3()?;

        // Phase 1: Process voice embeddings FIRST (if provided)
        // This matches Python's get_state_for_audio_prompt() which runs voice through
        // transformer BEFORE text to populate KV cache with voice context
        if let Some(voice) = voice_embedding {
            let voice_emb = voice.embedding().unsqueeze(0)?;
            let voice_emb = voice_emb.broadcast_as((batch_size, voice_emb.dim(1)?, voice_emb.dim(2)?))?;

            eprintln!("[FlowLM] Phase 1: Processing voice embeddings");
            eprintln!("[FlowLM] voice embedding shape: {:?}", voice_emb.dims());

            // Diagnostic: Check voice embedding stats
            let v_flat: Vec<f32> = voice_emb.flatten_all()?.to_vec1()?;
            let v_mean = v_flat.iter().sum::<f32>() / v_flat.len() as f32;
            let v_std = (v_flat.iter().map(|x| (x - v_mean).powi(2)).sum::<f32>() / v_flat.len() as f32).sqrt();
            eprintln!("[FlowLM] voice emb: mean={:.6}, std={:.4}", v_mean, v_std);

            // Run voice through transformer (populates KV cache positions 0-124)
            let mut hidden = voice_emb;
            for (i, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward(&hidden, &self.rotary, Some(&mut self.kv_caches[i]))?;
                // Diagnostic: Check hidden stats after each layer
                if i == 0 || i == 5 {
                    let h_flat: Vec<f32> = hidden.flatten_all()?.to_vec1()?;
                    let h_mean = h_flat.iter().sum::<f32>() / h_flat.len() as f32;
                    let h_std = (h_flat.iter().map(|x| (x - h_mean).powi(2)).sum::<f32>() / h_flat.len() as f32).sqrt();
                    eprintln!("[FlowLM] after layer {}: mean={:.6}, std={:.4}", i, h_mean, h_std);
                }
            }
            let _ = self.final_norm.forward(&hidden)?;

            eprintln!("[FlowLM] Voice processed, KV cache size: {}", self.cache_seq_len());
        }

        // Phase 2: Process text embeddings (appends to KV cache)
        // This matches Python's _generate() text prompting step
        eprintln!("[FlowLM] Phase 2: Processing text embeddings");
        eprintln!("[FlowLM] text embeddings shape: {:?}", text_embeddings.dims());

        // Diagnostic: Check text embedding stats
        let t_flat: Vec<f32> = text_embeddings.flatten_all()?.to_vec1()?;
        let t_mean = t_flat.iter().sum::<f32>() / t_flat.len() as f32;
        let t_std = (t_flat.iter().map(|x| (x - t_mean).powi(2)).sum::<f32>() / t_flat.len() as f32).sqrt();
        eprintln!("[FlowLM] text emb: mean={:.6}, std={:.4}", t_mean, t_std);

        let mut hidden = text_embeddings;

        // Diagnostic: Log RoPE offset for text processing
        let text_rope_offset = self.cache_seq_len();
        eprintln!("[FlowLM] text RoPE offset: {} (should be voice seq len)", text_rope_offset);

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mut self.kv_caches[i]))?;
            // Diagnostic: Check hidden stats after each layer
            if i == 0 || i == 5 {
                let h_flat: Vec<f32> = hidden.flatten_all()?.to_vec1()?;
                let h_mean = h_flat.iter().sum::<f32>() / h_flat.len() as f32;
                let h_std = (h_flat.iter().map(|x| (x - h_mean).powi(2)).sum::<f32>() / h_flat.len() as f32).sqrt();
                eprintln!("[FlowLM] text after layer {}: mean={:.6}, std={:.4}", i, h_mean, h_std);
            }
        }
        let text_final_hidden = self.final_norm.forward(&hidden)?;

        // Diagnostic: Check final text hidden stats
        let tf_flat: Vec<f32> = text_final_hidden.flatten_all()?.to_vec1()?;
        let tf_mean = tf_flat.iter().sum::<f32>() / tf_flat.len() as f32;
        let tf_std = (tf_flat.iter().map(|x| (x - tf_mean).powi(2)).sum::<f32>() / tf_flat.len() as f32).sqrt();
        eprintln!("[FlowLM] text final hidden: mean={:.6}, std={:.4}", tf_mean, tf_std);

        eprintln!("[FlowLM] Text processed, KV cache size: {}", self.cache_seq_len());

        // Step 2: Autoregressive latent generation
        // Estimate max generation length: ~12.5 frames per second of speech
        // Roughly 1 second of audio per 10-12 words
        let num_words = token_ids.dim(1)?;
        let max_gen_len = (num_words as f32 * 5.0 + 30.0) as usize; // Allow more frames (~45 for short phrases)
        eprintln!("[FlowLM] starting autoregressive generation, max_len={}", max_gen_len);

        // Debug: check BOS projection
        let bos_test = self.bos_emb.clone().unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 32]
        let bos_proj = self.input_linear.forward(&bos_test)?;
        if let Ok(vals) = bos_proj.flatten_all()?.to_vec1::<f32>() {
            eprintln!("[FlowLM] BOS projected first 8: {:?}", &vals[..8.min(vals.len())]);
            let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
            eprintln!("[FlowLM] BOS projected mean: {:.6}", mean);
        }

        // Use same defaults as Python reference:
        // - EOS threshold: -4.0 (logit must exceed this to trigger EOS)
        // - frames_after_eos: 2-3 (generate a few more frames after EOS)
        let eos_threshold = -4.0; // Match Python DEFAULT_EOS_THRESHOLD
        let frames_after_eos = 3; // Generate a few more frames after EOS detected
        let min_gen_steps = 40; // Force minimum frames to match Python (~41 frames)

        let mut all_latents: Vec<Tensor> = Vec::new();
        let mut eos_step: Option<usize> = None;

        // Start with BOS embedding
        let mut current_latent = self.bos_emb.clone().unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 32]

        for step in 0..max_gen_len {
            // Project latent to hidden dimension
            let latent_hidden = self.input_linear.forward(&current_latent)?; // [1, 1, 1024]

            // Run through transformer (using KV cache)
            let mut step_hidden = latent_hidden;
            for (i, layer) in self.layers.iter().enumerate() {
                step_hidden = layer.forward(&step_hidden, &self.rotary, Some(&mut self.kv_caches[i]))?;
            }
            let step_hidden = self.final_norm.forward(&step_hidden)?;

            // Get the last position's hidden state
            let last_hidden = step_hidden.squeeze(1)?; // [1, 1024]

            // DIAGNOSTIC: Log hidden state stats at first and last steps
            // Python hook captures LAST call to out_norm during generation
            if step == 0 || step >= max_gen_len.saturating_sub(5) {
                let h_flat: Vec<f32> = last_hidden.flatten_all()?.to_vec1()?;
                let h_mean = h_flat.iter().sum::<f32>() / h_flat.len() as f32;
                let h_std = (h_flat.iter().map(|x| (x - h_mean).powi(2)).sum::<f32>() / h_flat.len() as f32).sqrt();
                eprintln!(
                    "[FlowLM] step {} hidden: mean={:.6}, std={:.4}, first 8: {:?}",
                    step,
                    h_mean,
                    h_std,
                    &h_flat[..8.min(h_flat.len())]
                );
                // Python out_norm: mean=-0.003252, std=0.340720
                // Python first 8: [-0.10488168, -0.26733553, 0.00387744, -0.23025721, 0.29963714, 0.6678712, 0.5796935, 0.6726278]
            }

            // Check EOS prediction (but only after min_gen_steps for debugging)
            let eos_logit = self.out_eos.forward(&last_hidden)?; // [1, 1]
            let eos_val: f32 = eos_logit.squeeze(1)?.to_vec1::<f32>()?[0];

            if step >= min_gen_steps && eos_val > eos_threshold && eos_step.is_none() {
                eprintln!("[FlowLM] EOS detected at step {}, logit={:.4}", step, eos_val);
                eos_step = Some(step);
            }

            // Check if we should stop (only after min_gen_steps)
            if let Some(eos) = eos_step {
                if step >= eos + frames_after_eos {
                    eprintln!("[FlowLM] stopping after {} frames post-EOS", frames_after_eos);
                    break;
                }
            }

            // Generate next latent via FlowNet
            // FlowNet expects [batch, seq, hidden] but we have [batch, hidden]
            let cond = last_hidden.unsqueeze(1)?; // [1, 1, 1024]
            let next_normalized = self.flow_net.generate(&cond, num_flow_steps, temperature, &self.device)?;

            // IMPORTANT: Do NOT denormalize here!
            // Python feeds the raw FlowNet output back to transformer.
            // Denormalization only happens when passing to Mimi decoder.
            // Store normalized latent for later denormalization
            all_latents.push(next_normalized.clone());
            current_latent = next_normalized;

            if step % 10 == 0 {
                eprintln!("[FlowLM] step {}/{}, eos_logit={:.4}", step, max_gen_len, eos_val);
            }
        }

        if eos_step.is_none() {
            eprintln!("[FlowLM] WARNING: reached max length without EOS");
        }

        eprintln!("[FlowLM] generated {} latent frames", all_latents.len());

        // Concatenate all latents: [1, num_frames, 32]
        if all_latents.is_empty() {
            return Err(candle_core::Error::Msg("No latents generated".to_string()));
        }

        let latents = Tensor::cat(&all_latents, 1)?;
        eprintln!("[FlowLM] final latents shape: {:?}", latents.dims());

        Ok(latents)
    }

    /// Reset KV caches for new sequence
    pub fn reset_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.clear();
        }
    }

    /// Get current cache sequence length
    pub fn cache_seq_len(&self) -> usize {
        self.kv_caches.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    /// Denormalize latents before passing to Mimi decoder
    /// Python: mimi_decoding_input = latent * emb_std + emb_mean
    pub fn denormalize_latents(&self, latents: &Tensor) -> Result<Tensor> {
        latents.broadcast_mul(&self.emb_std)?.broadcast_add(&self.emb_mean)
    }

    pub fn config(&self) -> &FlowLMConfig {
        &self.config
    }
}
