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

        if let Some((dir, li)) = dump {
            dump_npy(dir, &format!("layer{}_input", li), &x.flatten_all()?)?;
        }

        let normed = self.norm1.forward(x)?;
        if let Some((dir, li)) = dump {
            dump_npy(dir, &format!("layer{}_norm1", li), &normed.flatten_all()?)?;
        }

        let attn_out = self.attn.forward(&normed, Some(rotary), kv_cache, true)?;
        if let Some((dir, li)) = dump {
            dump_npy(dir, &format!("layer{}_attn", li), &attn_out.flatten_all()?)?;
        }

        let x = (residual + &attn_out)?;
        if let Some((dir, li)) = dump {
            dump_npy(dir, &format!("layer{}_post_attn", li), &x.flatten_all()?)?;
        }

        // Pre-norm MLP
        let residual = &x;
        let normed2 = self.norm2.forward(&x)?;
        if let Some((dir, li)) = dump {
            dump_npy(dir, &format!("layer{}_norm2", li), &normed2.flatten_all()?)?;
        }

        let mlp_out = self.mlp.forward(&normed2)?;
        if let Some((dir, li)) = dump {
            dump_npy(dir, &format!("layer{}_mlp", li), &mlp_out.flatten_all()?)?;
        }

        let output = (residual + mlp_out)?;
        if let Some((dir, li)) = dump {
            dump_npy(dir, &format!("layer{}_output", li), &output.flatten_all()?)?;
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
                hidden = layer.forward(&hidden, &self.rotary, Some(&mut self.kv_caches[i]), None)?;
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
            hidden = layer.forward(&hidden, &self.rotary, Some(&mut self.kv_caches[i]), None)?;
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
        // - frames_after_eos: calculated from num_text_tokens (Python formula)
        let eos_threshold = -4.0; // Match Python DEFAULT_EOS_THRESHOLD
        let num_text_tokens = token_ids.dim(1)?;
        // Python: frames_after_eos = min(5, ceil(num_text_tokens / 4))
        let frames_after_eos = std::cmp::min(5, (num_text_tokens + 3) / 4);
        // Remove debug min_gen_steps - allow natural EOS detection
        let min_gen_steps = 0; // Natural EOS detection

        let mut all_latents: Vec<Tensor> = Vec::new();
        let mut eos_step: Option<usize> = None;
        let mut eos_logits: Vec<f32> = Vec::new(); // Track EOS trajectory for debugging

        // Check env var for step-0 intermediate tensor dump
        let dump_dir = std::env::var("DUMP_STEP0").ok().map(std::path::PathBuf::from);
        if let Some(ref dir) = dump_dir {
            std::fs::create_dir_all(dir).ok();
            eprintln!("--- Dumping step 0 intermediates to {} ---", dir.display());
        }

        // Start with BOS embedding
        let mut current_latent = self.bos_emb.clone().unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 32]

        for step in 0..max_gen_len {
            // DIAGNOSTIC: Capture the raw latent at early steps and around step 36
            if step <= 5 || step == 35 || step == 36 {
                let lat_flat: Vec<f32> = current_latent.flatten_all()?.to_vec1()?;
                let lat_mean = lat_flat.iter().sum::<f32>() / lat_flat.len() as f32;
                let lat_std =
                    (lat_flat.iter().map(|x| (x - lat_mean).powi(2)).sum::<f32>() / lat_flat.len() as f32).sqrt();
                eprintln!(
                    "[LATENT] Rust step={}: mean={:.6}, std={:.6}, first 8: {:?}",
                    step,
                    lat_mean,
                    lat_std,
                    &lat_flat[..8.min(lat_flat.len())]
                );
            }

            // Project latent to hidden dimension
            let latent_hidden = self.input_linear.forward(&current_latent)?; // [1, 1, 1024]

            // Run through transformer (using KV cache)
            let mut step_hidden = latent_hidden.clone();

            // Dump intermediates for steps 0-2 to track divergence accumulation
            let step_dump = if step <= 2 { dump_dir.as_deref() } else { None };

            if let Some(dir) = step_dump {
                if step > 0 {
                    eprintln!("--- Step {} intermediates ---", step);
                }
                dump_npy(dir, &format!("step{}_input_linear", step), &step_hidden.flatten_all()?)?;
            }

            for (i, layer) in self.layers.iter().enumerate() {
                // Only dump per-layer detail for step 0; steps 1-2 just get layer outputs
                let layer_dump = if step == 0 { step_dump.map(|d| (d, i)) } else { None };
                step_hidden = layer.forward(&step_hidden, &self.rotary, Some(&mut self.kv_caches[i]), layer_dump)?;
            }
            let step_hidden = self.final_norm.forward(&step_hidden)?;

            if let Some(dir) = step_dump {
                dump_npy(dir, &format!("step{}_out_norm", step), &step_hidden.flatten_all()?)?;
            }

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

            // Check EOS prediction
            let eos_logit = self.out_eos.forward(&last_hidden)?; // [1, 1]
            let eos_val: f32 = eos_logit.squeeze(1)?.to_vec1::<f32>()?[0];
            eos_logits.push(eos_val);

            // Log EOS at every step for trajectory analysis
            if step % 10 == 0 || step == 0 || eos_val > eos_threshold - 1.0 || (36..=42).contains(&step) {
                eprintln!(
                    "[EOS-TRAJ] step={:3}, eos_logit={:7.4}, threshold={}",
                    step, eos_val, eos_threshold
                );
                // Extra debug at critical steps around divergence point
                if (36..=42).contains(&step) {
                    let h_flat: Vec<f32> = last_hidden.flatten_all()?.to_vec1()?;
                    let h_mean = h_flat.iter().sum::<f32>() / h_flat.len() as f32;
                    let h_std = (h_flat.iter().map(|x| (x - h_mean).powi(2)).sum::<f32>() / h_flat.len() as f32).sqrt();
                    let h_min = h_flat.iter().cloned().fold(f32::INFINITY, f32::min);
                    let h_max = h_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!(
                        "[EOS-DEBUG] step={}, hidden: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
                        step, h_mean, h_std, h_min, h_max
                    );
                    eprintln!("[EOS-DEBUG] step={}, hidden first 8: {:?}", step, &h_flat[..8]);
                }
            }

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
            let cond = last_hidden.unsqueeze(1)?; // [1, 1, 1024]
            let step_seed = seed.map(|s| s.wrapping_add(step as u64));
            // Offset by 1: noise_step_000 = Python's text prompting (discarded)
            let noise_override = noise_tensors.and_then(|nt| nt.get(step + 1));
            let next_normalized =
                self.flow_net
                    .generate(&cond, num_flow_steps, temperature, &self.device, step_seed, noise_override)?;

            // Dump FlowNet output (latent) and input_linear output for steps 0-2
            if let Some(ref dir) = dump_dir {
                if step <= 2 {
                    dump_npy(dir, &format!("step{}_latent", step), &next_normalized.flatten_all()?)?;
                }
            }

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

        // Log EOS trajectory summary for debugging
        if !eos_logits.is_empty() {
            let eos_max = eos_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let eos_min = eos_logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let eos_mean = eos_logits.iter().sum::<f32>() / eos_logits.len() as f32;
            eprintln!(
                "[EOS-SUMMARY] min={:.4}, max={:.4}, mean={:.4}, count={}",
                eos_min,
                eos_max,
                eos_mean,
                eos_logits.len()
            );
        }

        // Concatenate all latents: [1, num_frames, 32]
        if all_latents.is_empty() {
            return Err(candle_core::Error::Msg("No latents generated".to_string()));
        }

        let latents = Tensor::cat(&all_latents, 1)?;
        eprintln!("[FlowLM] final latents shape: {:?}", latents.dims());

        Ok(latents)
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

        // Phase 1: Process voice embeddings FIRST (if provided)
        if let Some(voice) = voice_embedding {
            let voice_emb = voice.embedding().unsqueeze(0)?;
            let voice_emb = voice_emb.broadcast_as((batch_size, voice_emb.dim(1)?, voice_emb.dim(2)?))?;

            let mut hidden = voice_emb;
            for (i, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward(&hidden, &self.rotary, Some(&mut self.kv_caches[i]), None)?;
            }
            let _ = self.final_norm.forward(&hidden)?;
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
            // Use pre-captured noise tensor if available for this step.
            // Offset by 1: noise_step_000 is Python's text prompting noise (discarded).
            let noise_override = noise_tensors.and_then(|nt| nt.get(step + 1));
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
