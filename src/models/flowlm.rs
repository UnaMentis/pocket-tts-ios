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
use std::path::Path;

use crate::modules::{
    attention::{FusedMultiHeadAttention, KVCache},
    embeddings::{TextEmbedding, VoiceEmbedding},
    flownet::{FlowNet, FlowNetConfig},
    layer_norm::LayerNorm,
    mlp::SimpleMLP,
    rotary::RotaryEmbedding,
};

/// Write a 1-D f32 tensor to a .npy file for cross-implementation comparison.
#[cfg(feature = "diagnostics")]
fn dump_npy(dir: &Path, name: &str, tensor: &Tensor) -> Result<()> {
    use std::io::Write;
    let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let shape_str = format!("({},)", flat.len());
    // Minimal NumPy .npy v1.0 header
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}", shape_str);
    // Pad header to align total (magic 6 + ver 2 + hdr_len 2 + header) to 64 bytes
    let prefix_len = 10; // 6 magic + 2 version + 2 header_len
    let pad = 64 - ((prefix_len + header.len() + 1) % 64); // +1 for \n
    let padded_header = format!("{}{}\n", header, " ".repeat(pad));
    let hdr_len = padded_header.len() as u16;

    let path = dir.join(format!("{}.npy", name));
    let mut f = std::fs::File::create(&path).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    // Magic + version
    f.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0])
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    f.write_all(&hdr_len.to_le_bytes())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    f.write_all(padded_header.as_bytes())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    // Raw f32 data in little-endian
    let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
    let std = (flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat.len() as f32).sqrt();
    eprintln!(
        "  {}: shape=({},) mean={:.6} std={:.6} first4=[{:.6},{:.6},{:.6},{:.6}]",
        name,
        flat.len(),
        mean,
        std,
        flat[0],
        flat[1],
        flat[2],
        flat[3]
    );
    Ok(())
}

/// Dump one named per-layer tensor when the `diagnostics` feature is enabled.
/// Compiles to a no-op otherwise, so call sites stay branch-free.
#[cfg(feature = "diagnostics")]
fn dump_layer_tensor(dump: Option<(&Path, usize)>, stage: &str, tensor: &Tensor) -> Result<()> {
    if let Some((dir, li)) = dump {
        dump_npy(dir, &format!("layer{}_{}", li, stage), &tensor.flatten_all()?)?;
    }
    Ok(())
}

#[cfg(not(feature = "diagnostics"))]
#[inline(always)]
fn dump_layer_tensor(_dump: Option<(&Path, usize)>, _stage: &str, _tensor: &Tensor) -> Result<()> {
    Ok(())
}

/// Dump one named per-step tensor when the `diagnostics` feature is enabled.
#[cfg(feature = "diagnostics")]
fn dump_step_tensor(dir: Option<&Path>, name: &str, tensor: &Tensor) -> Result<()> {
    if let Some(dir) = dir {
        dump_npy(dir, name, &tensor.flatten_all()?)?;
    }
    Ok(())
}

#[cfg(not(feature = "diagnostics"))]
#[inline(always)]
fn dump_step_tensor(_dir: Option<&Path>, _name: &str, _tensor: &Tensor) -> Result<()> {
    Ok(())
}

/// Control flow for streaming latent generation
///
/// Returned by the callback to indicate whether generation should continue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentStreamControl {
    /// Continue generating the next latent
    Continue,
    /// Stop generation early (e.g., user cancelled)
    Stop,
}

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

    /// Run one transformer layer.
    /// If `dump` is Some((dir, layer_idx)), save intermediate tensors as .npy files.
    fn forward(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        kv_cache: Option<&mut KVCache>,
        dump: Option<(&Path, usize)>,
    ) -> Result<Tensor> {
        // Pre-norm attention (Kyutai Pocket architecture)
        let residual = x;
        dump_layer_tensor(dump, "input", x)?;

        let normed = self.norm1.forward(x)?;
        dump_layer_tensor(dump, "norm1", &normed)?;

        let attn_out = self.attn.forward(&normed, Some(rotary), kv_cache, true)?;
        dump_layer_tensor(dump, "attn", &attn_out)?;

        let x = (residual + &attn_out)?;
        dump_layer_tensor(dump, "post_attn", &x)?;

        // Pre-norm MLP
        let residual = &x;
        let normed2 = self.norm2.forward(&x)?;
        dump_layer_tensor(dump, "norm2", &normed2)?;

        let mlp_out = self.mlp.forward(&normed2)?;
        dump_layer_tensor(dump, "mlp", &mlp_out)?;

        let output = (residual + mlp_out)?;
        dump_layer_tensor(dump, "output", &output)?;

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
            hidden = layer.forward(&hidden, &self.rotary, cache, None)?;
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
    ///
    /// The `noise_tensors` parameter allows loading pre-captured Python noise tensors
    /// for correlation testing. When provided, noise_tensors[step] is used instead of
    /// random sampling at each generation step.
    pub fn generate_latents(
        &mut self,
        token_ids: &Tensor,
        voice_embedding: Option<&VoiceEmbedding>,
        num_flow_steps: usize,
        temperature: f32,
        seed: Option<u64>,
        noise_tensors: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        // Reset caches before generation
        self.reset_cache();

        let text_embeddings = self.text_embedding.forward(token_ids)?;
        let (batch_size, _seq_len, _hidden_dim) = text_embeddings.dims3()?;

        // Phase 1: Voice conditioning (shared with the streaming path)
        if let Some(voice) = voice_embedding {
            self.prime_voice_conditioning(voice, batch_size)?;
        }

        // Phase 2: Process text embeddings (appends to KV cache)
        // This matches Python's _generate() text prompting step
        let mut hidden = text_embeddings;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mut self.kv_caches[i]), None)?;
        }
        let _ = self.final_norm.forward(&hidden)?;
        log::debug!("[FlowLM] text processed, KV cache size: {}", self.cache_seq_len());

        // Step 2: Autoregressive latent generation
        // Estimate max generation length: ~12.5 frames per second of speech
        // Roughly 1 second of audio per 10-12 words
        let num_words = token_ids.dim(1)?;
        let max_gen_len = (num_words as f32 * 5.0 + 30.0) as usize; // Allow more frames (~45 for short phrases)

        // Use same defaults as Python reference:
        // - EOS threshold: -4.0 (logit must exceed this to trigger EOS)
        // - frames_after_eos: calculated from num_text_tokens (Python formula)
        let eos_threshold = -4.0; // Match Python DEFAULT_EOS_THRESHOLD
        let num_text_tokens = token_ids.dim(1)?;
        // Python: frames_after_eos = min(5, ceil(num_text_tokens / 4))
        let frames_after_eos = std::cmp::min(5, (num_text_tokens + 3) / 4);
        // Remove debug min_gen_steps - allow natural EOS detection
        let min_gen_steps = 0; // Natural EOS detection

        let mut all_latents: Vec<Tensor> = Vec::new();
        let mut eos_step: Option<usize> = None;

        // Optional step-0 intermediate tensor dump for parity debugging
        // (only active in `diagnostics` builds; the env var is ignored otherwise).
        let dump_dir = std::env::var("DUMP_STEP0").ok().map(std::path::PathBuf::from);
        if cfg!(feature = "diagnostics") {
            if let Some(ref dir) = dump_dir {
                std::fs::create_dir_all(dir).ok();
            }
        }

        // Start with BOS embedding
        let mut current_latent = self.bos_emb.clone().unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 32]

        for step in 0..max_gen_len {
            // Project latent to hidden dimension
            let latent_hidden = self.input_linear.forward(&current_latent)?; // [1, 1, 1024]

            // Run through transformer (using KV cache)
            let mut step_hidden = latent_hidden;

            // Dump intermediates for steps 0-2 to track divergence accumulation
            let step_dump = if step <= 2 { dump_dir.as_deref() } else { None };
            dump_step_tensor(step_dump, &format!("step{}_input_linear", step), &step_hidden)?;

            for (i, layer) in self.layers.iter().enumerate() {
                // Only dump per-layer detail for step 0; steps 1-2 just get layer outputs
                let layer_dump = if step == 0 { step_dump.map(|d| (d, i)) } else { None };
                step_hidden = layer.forward(&step_hidden, &self.rotary, Some(&mut self.kv_caches[i]), layer_dump)?;
            }
            let step_hidden = self.final_norm.forward(&step_hidden)?;
            dump_step_tensor(step_dump, &format!("step{}_out_norm", step), &step_hidden)?;

            // Get the last position's hidden state
            let last_hidden = step_hidden.squeeze(1)?; // [1, 1024]

            // Check EOS prediction
            let eos_logit = self.out_eos.forward(&last_hidden)?; // [1, 1]
            let eos_val: f32 = eos_logit.squeeze(1)?.to_vec1::<f32>()?[0];

            if step >= min_gen_steps && eos_val > eos_threshold && eos_step.is_none() {
                log::debug!("[FlowLM] EOS detected at step {}, logit={:.4}", step, eos_val);
                eos_step = Some(step);
            }

            // Check if we should stop (only after min_gen_steps)
            if let Some(eos) = eos_step {
                if step >= eos + frames_after_eos {
                    break;
                }
            }

            // Generate next latent via FlowNet
            let cond = last_hidden.unsqueeze(1)?; // [1, 1, 1024]
            let step_seed = seed.map(|s| s.wrapping_add(step as u64));
            // Noise-matched runs use the captured Python noise, never RNG:
            // a silent RNG fallback would produce a deterministic-but-WRONG
            // parity result. Offset by 1: noise_step_000 is Python's text
            // prompting draw (discarded). When the capture runs out, the
            // parity region is over — stop cleanly (Python stopped there too).
            let noise_override = match noise_tensors {
                None => None,
                Some(nt) => match nt.get(step + 1) {
                    Some(t) => Some(t),
                    None if step == 0 => {
                        return Err(candle_core::Error::Msg(format!(
                            "noise capture has only {} tensor(s) — too short to noise-match even one \
                             step (corrupt or wrong noise_dir)",
                            nt.len()
                        )))
                    },
                    None => {
                        log::debug!(
                            "[FlowLM] captured noise exhausted at step {} — ending noise-matched generation",
                            step
                        );
                        break;
                    },
                },
            };
            let next_normalized =
                self.flow_net
                    .generate(&cond, num_flow_steps, temperature, &self.device, step_seed, noise_override)?;

            // Dump FlowNet output (latent) for steps 0-2
            if step <= 2 {
                dump_step_tensor(dump_dir.as_deref(), &format!("step{}_latent", step), &next_normalized)?;
            }

            all_latents.push(next_normalized.clone());
            current_latent = next_normalized;
        }

        if eos_step.is_none() {
            log::warn!("[FlowLM] reached max generation length ({}) without EOS", max_gen_len);
        }
        log::debug!("[FlowLM] generated {} latent frames", all_latents.len());

        // Concatenate all latents: [1, num_frames, 32]
        if all_latents.is_empty() {
            return Err(candle_core::Error::Msg("No latents generated".to_string()));
        }

        Tensor::cat(&all_latents, 1)
    }

    /// Phase 1 of generation: prime the KV caches with voice conditioning.
    ///
    /// Shared by `generate_latents` and `generate_latents_streaming` so the two
    /// paths cannot diverge. Handles both voice formats:
    /// - v2 (Pocket TTS ≥2026-04): the voice file ships a precomputed self-attention
    ///   KV cache (`bos_before_voice` + speaker projection baked in offline) which is
    ///   loaded directly into each layer's cache.
    /// - v1: a voice embedding sequence that is run through the transformer to
    ///   populate the cache (positions 0-124), matching Python's
    ///   `get_state_for_audio_prompt()`.
    fn prime_voice_conditioning(&mut self, voice: &VoiceEmbedding, batch_size: usize) -> Result<()> {
        if let Some(state) = voice.kv_state() {
            if state.layers.len() != self.kv_caches.len() {
                return Err(candle_core::Error::Msg(format!(
                    "v2 voice KV state has {} transformer layers but the model has {} — \
                     this voice file belongs to a different Pocket TTS variant",
                    state.layers.len(),
                    self.kv_caches.len()
                )));
            }
            for (i, (k, v)) in state.layers.iter().enumerate() {
                self.kv_caches[i].set(k.clone(), v.clone());
            }
            log::debug!(
                "[FlowLM] loaded v2 voice KV state ({} layers, {} positions)",
                state.layers.len(),
                self.cache_seq_len()
            );
        } else {
            let voice_emb = voice.embedding().unsqueeze(0)?;
            let voice_emb = voice_emb.broadcast_as((batch_size, voice_emb.dim(1)?, voice_emb.dim(2)?))?;

            // Run voice through transformer (populates KV cache positions 0-124)
            let mut hidden = voice_emb;
            for (i, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward(&hidden, &self.rotary, Some(&mut self.kv_caches[i]), None)?;
            }
            let _ = self.final_norm.forward(&hidden)?;
            log::debug!("[FlowLM] voice processed, KV cache size: {}", self.cache_seq_len());
        }
        Ok(())
    }

    /// Generate latents autoregressively with streaming callback
    ///
    /// Same as `generate_latents()` but invokes a callback for each latent
    /// as it's generated. This enables low TTFA (Time To First Audio) by
    /// allowing the Mimi decoder to start processing immediately.
    ///
    /// The callback receives:
    /// - `latent`: The normalized latent tensor [1, 1, 32]
    /// - `step`: The generation step (0-indexed)
    /// - `is_eos`: Whether EOS was detected at this step
    ///
    /// Returns `LatentStreamControl::Stop` from the callback to terminate early.
    pub fn generate_latents_streaming<F>(
        &mut self,
        token_ids: &Tensor,
        voice_embedding: Option<&VoiceEmbedding>,
        num_flow_steps: usize,
        temperature: f32,
        seed: Option<u64>,
        noise_tensors: Option<&[Tensor]>,
        mut callback: F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, usize, bool) -> LatentStreamControl,
    {
        // Reset caches before generation
        self.reset_cache();

        let text_embeddings = self.text_embedding.forward(token_ids)?;
        let (batch_size, _seq_len, _hidden_dim) = text_embeddings.dims3()?;

        // Phase 1: Voice conditioning (shared with the sync path; handles v1
        // embedding voices and v2 precomputed KV-state voices identically).
        if let Some(voice) = voice_embedding {
            self.prime_voice_conditioning(voice, batch_size)?;
        }

        // Phase 2: Process text embeddings (appends to KV cache)
        let mut hidden = text_embeddings;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mut self.kv_caches[i]), None)?;
        }
        let _ = self.final_norm.forward(&hidden)?;

        // Phase 3: Autoregressive latent generation with streaming
        let num_words = token_ids.dim(1)?;
        let max_gen_len = (num_words as f32 * 5.0 + 30.0) as usize;

        let eos_threshold = -4.0;
        let num_text_tokens = token_ids.dim(1)?;
        let frames_after_eos = std::cmp::min(5, (num_text_tokens + 3) / 4);
        let min_gen_steps = 0; // Match batch generate_latents for consistent EOS detection

        let mut all_latents: Vec<Tensor> = Vec::new();
        let mut eos_step: Option<usize> = None;
        let mut current_latent = self.bos_emb.clone().unsqueeze(0)?.unsqueeze(0)?;

        for step in 0..max_gen_len {
            // Project latent to hidden dimension
            let latent_hidden = self.input_linear.forward(&current_latent)?;

            // Run through transformer (using KV cache)
            let mut step_hidden = latent_hidden;
            for (i, layer) in self.layers.iter().enumerate() {
                step_hidden = layer.forward(&step_hidden, &self.rotary, Some(&mut self.kv_caches[i]), None)?;
            }
            let step_hidden = self.final_norm.forward(&step_hidden)?;
            let last_hidden = step_hidden.squeeze(1)?;

            // Check EOS prediction
            let eos_logit = self.out_eos.forward(&last_hidden)?;
            let eos_val: f32 = eos_logit.squeeze(1)?.to_vec1::<f32>()?[0];

            let is_eos = step >= min_gen_steps && eos_val > eos_threshold && eos_step.is_none();
            if is_eos {
                eos_step = Some(step);
            }

            // Generate next latent via FlowNet
            let cond = last_hidden.unsqueeze(1)?;
            // Derive per-step seed for different-but-deterministic noise at each step
            let step_seed = seed.map(|s| s.wrapping_add(step as u64));
            // Same noise-matching contract as generate_latents: captured noise or
            // nothing — stop at capture end, never fall back to RNG mid-run.
            let noise_override = match noise_tensors {
                None => None,
                Some(nt) => match nt.get(step + 1) {
                    Some(t) => Some(t),
                    None if step == 0 => {
                        return Err(candle_core::Error::Msg(format!(
                            "noise capture has only {} tensor(s) — too short to noise-match even one \
                             step (corrupt or wrong noise_dir)",
                            nt.len()
                        )))
                    },
                    None => {
                        log::debug!(
                            "[FlowLM] captured noise exhausted at step {} — ending noise-matched generation",
                            step
                        );
                        break;
                    },
                },
            };
            let next_normalized =
                self.flow_net
                    .generate(&cond, num_flow_steps, temperature, &self.device, step_seed, noise_override)?;

            // *** STREAMING CALLBACK: Yield latent immediately ***
            let control = callback(&next_normalized, step, is_eos);

            // Store for final return
            all_latents.push(next_normalized.clone());

            // Check early termination from callback
            if control == LatentStreamControl::Stop {
                break;
            }

            // Check EOS-based termination
            if let Some(eos) = eos_step {
                if step >= eos + frames_after_eos {
                    break;
                }
            }

            current_latent = next_normalized;
        }

        // Return all latents (even if terminated early)
        if all_latents.is_empty() {
            return Err(candle_core::Error::Msg("No latents generated".to_string()));
        }

        Tensor::cat(&all_latents, 1)
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

    /// Get the embedding mean for denormalization
    pub fn emb_mean(&self) -> &Tensor {
        &self.emb_mean
    }

    /// Get the embedding std for denormalization
    pub fn emb_std(&self) -> &Tensor {
        &self.emb_std
    }
}
