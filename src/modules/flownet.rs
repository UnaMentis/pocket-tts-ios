//! FlowNet module for Kyutai Pocket TTS
//!
//! Flow matching network that generates latent representations from
//! transformer hidden states. Uses AdaLN (adaptive layer normalization)
//! for conditioning on time and hidden states.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::layer_norm::LayerNorm;

/// FlowNet configuration
#[derive(Debug, Clone)]
pub struct FlowNetConfig {
    pub hidden_dim: usize,      // 512
    pub cond_dim: usize,        // 1024 (from transformer)
    pub latent_dim: usize,      // 32
    pub num_res_blocks: usize,  // 6
    pub time_embed_dim: usize,  // 256 (freqs * 2)
}

impl Default for FlowNetConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 512,
            cond_dim: 1024,
            latent_dim: 32,
            num_res_blocks: 6,
            time_embed_dim: 256,
        }
    }
}

/// Time embedding with sinusoidal encoding and MLP
#[derive(Debug)]
struct TimeEmbedding {
    freqs: Tensor,
    mlp_0: Linear,
    mlp_2: Linear,
    alpha: Tensor,
}

impl TimeEmbedding {
    fn new(hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        // Load pre-computed frequencies
        let freqs = vb.get((128,), "freqs")?;

        // DIAGNOSTIC: Log frequency range
        let freqs_vec: Vec<f32> = freqs.to_vec1()?;
        let f_min = freqs_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let f_max = freqs_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[TimeEmbed] freqs range: [{:.2}, {:.2}]", f_min, f_max);

        // MLP: 256 -> 512 -> 512
        let mlp_0 = candle_nn::linear(256, hidden_dim, vb.pp("mlp.0"))?;
        let mlp_2 = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("mlp.2"))?;

        // Learnable scale parameter
        let alpha = vb.get((hidden_dim,), "mlp.3.alpha")?;

        // DIAGNOSTIC: Log alpha range
        let alpha_vec: Vec<f32> = alpha.to_vec1()?;
        let a_min = alpha_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let a_max = alpha_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[TimeEmbed] alpha range: [{:.4}, {:.4}]", a_min, a_max);

        Ok(Self { freqs, mlp_0, mlp_2, alpha })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // Create sinusoidal embedding from time
        // CRITICAL: Python uses [cos, sin] order, not [sin, cos]!
        let t_expanded = t.unsqueeze(1)?;  // [batch, 1]
        let freqs_expanded = self.freqs.unsqueeze(0)?;  // [1, 128]
        let angles = t_expanded.broadcast_mul(&freqs_expanded)?;

        let cos_emb = angles.cos()?;
        let sin_emb = angles.sin()?;
        let time_emb = Tensor::cat(&[cos_emb, sin_emb], 1)?;  // [batch, 256] - COS first!

        // DIAGNOSTIC: Log sinusoidal embedding stats before MLP
        let t_val: f32 = t.to_vec1()?[0];
        if (t_val - 1.0).abs() < 0.01 || (t_val - 0.0).abs() < 0.1 {
            let emb_flat: Vec<f32> = time_emb.flatten_all()?.to_vec1()?;
            let emb_mean = emb_flat.iter().sum::<f32>() / emb_flat.len() as f32;
            let emb_std = (emb_flat.iter().map(|x| (x - emb_mean).powi(2)).sum::<f32>() / emb_flat.len() as f32).sqrt();
            eprintln!("[TimeEmbed] t={:.3}: sinusoidal emb mean={:.6}, std={:.4}", t_val, emb_mean, emb_std);
        }

        // MLP with SiLU (not GELU!) - matches Python TimestepEmbedder
        let x = self.mlp_0.forward(&time_emb)?;

        // DIAGNOSTIC: Log after first MLP layer
        if (t_val - 1.0).abs() < 0.01 || (t_val - 0.0).abs() < 0.1 {
            let mlp0_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let mlp0_mean = mlp0_flat.iter().sum::<f32>() / mlp0_flat.len() as f32;
            let mlp0_std = (mlp0_flat.iter().map(|v| (v - mlp0_mean).powi(2)).sum::<f32>() / mlp0_flat.len() as f32).sqrt();
            eprintln!("[TimeEmbed] t={:.3}: after mlp_0 mean={:.6}, std={:.4}", t_val, mlp0_mean, mlp0_std);
        }

        let x = candle_nn::ops::silu(&x)?;  // Python uses SiLU, not GELU
        let x = self.mlp_2.forward(&x)?;

        // DIAGNOSTIC: Log after second MLP layer (before alpha)
        if (t_val - 1.0).abs() < 0.01 || (t_val - 0.0).abs() < 0.1 {
            let mlp2_flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let mlp2_mean = mlp2_flat.iter().sum::<f32>() / mlp2_flat.len() as f32;
            let mlp2_std = (mlp2_flat.iter().map(|v| (v - mlp2_mean).powi(2)).sum::<f32>() / mlp2_flat.len() as f32).sqrt();
            eprintln!("[TimeEmbed] t={:.3}: after mlp_2 mean={:.6}, std={:.4}", t_val, mlp2_mean, mlp2_std);
        }

        // Apply learnable scale
        // For consistency models (trained with 1-step distillation), alpha scaling
        // is intentional and produces the correct target-pointing velocity
        x.broadcast_mul(&self.alpha)
    }
}

/// AdaLN modulation for a residual block
/// Produces scale, shift, gate (3 * hidden_dim outputs)
#[derive(Debug)]
struct AdaLNModulation {
    linear: Linear,
}

impl AdaLNModulation {
    fn new(hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        // Output dim is 3 * hidden_dim for shift, scale, gate
        // Python: nn.Sequential(nn.SiLU(), nn.Linear(...))
        // The ".1" suffix indicates the Linear is at index 1
        let linear = candle_nn::linear(hidden_dim, hidden_dim * 3, vb.pp("1"))?;
        Ok(Self { linear })
    }

    fn forward(&self, cond: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // Python applies SiLU BEFORE the linear layer
        let cond_activated = candle_nn::ops::silu(cond)?;
        let out = self.linear.forward(&cond_activated)?;
        // Chunk along the last dimension (hidden dim, not sequence)
        // For 3D tensors [batch, seq, hidden*3], this is dimension 2
        // Python order is [shift, scale, gate] - return (shift, scale, gate)
        let chunk_dim = out.dims().len() - 1;
        let chunks = out.chunk(3, chunk_dim)?;
        Ok((chunks[0].clone(), chunks[1].clone(), chunks[2].clone()))  // shift, scale, gate
    }
}

/// AdaLN modulation for final layer (no gate, only scale and shift)
/// Produces scale, shift (2 * hidden_dim outputs)
#[derive(Debug)]
struct FinalLayerAdaLN {
    linear: Linear,
}

impl FinalLayerAdaLN {
    fn new(hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        // Output dim is 2 * hidden_dim for scale and shift only (no gate)
        // Python: nn.Sequential(nn.SiLU(), nn.Linear(...))
        // ".1" indicates the Linear is at index 1 (SiLU is index 0, which is parameterless)
        let linear = candle_nn::linear(hidden_dim, hidden_dim * 2, vb.pp("1"))?;
        Ok(Self { linear })
    }

    fn forward(&self, cond: &Tensor) -> Result<(Tensor, Tensor)> {
        // Python applies SiLU BEFORE the linear layer
        let cond_activated = candle_nn::ops::silu(cond)?;
        let out = self.linear.forward(&cond_activated)?;
        // Chunk along the last dimension (hidden dim, not sequence)
        // For 3D tensors [batch, seq, hidden*2], this is dimension 2
        // Python order is [shift, scale] - return (shift, scale)
        let chunk_dim = out.dims().len() - 1;
        let chunks = out.chunk(2, chunk_dim)?;
        Ok((chunks[0].clone(), chunks[1].clone()))  // shift, scale
    }
}

/// Residual block with AdaLN modulation
#[derive(Debug)]
struct ResBlock {
    in_ln: LayerNorm,
    mlp_0: Linear,
    mlp_2: Linear,
    adaln: AdaLNModulation,
}

impl ResBlock {
    fn new(hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let in_ln = LayerNorm::new(hidden_dim, 1e-6, vb.pp("in_ln"))?;
        let mlp_0 = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("mlp.0"))?;
        let mlp_2 = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("mlp.2"))?;
        let adaln = AdaLNModulation::new(hidden_dim, vb.pp("adaLN_modulation"))?;

        Ok(Self { in_ln, mlp_0, mlp_2, adaln })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        // Get AdaLN modulation parameters (Python order: shift, scale, gate)
        let (shift, scale, gate) = self.adaln.forward(cond)?;

        // Normalize and modulate: x * (1 + scale) + shift
        let h = self.in_ln.forward(x)?;
        let h = h.broadcast_mul(&(scale + 1.0)?)?;
        let h = h.broadcast_add(&shift)?;

        // MLP with SiLU (Python uses SiLU in ResBlock too)
        let h = self.mlp_0.forward(&h)?;
        let h = candle_nn::ops::silu(&h)?;  // SiLU, not GELU
        let h = self.mlp_2.forward(&h)?;

        // Gated residual
        let h = h.broadcast_mul(&gate)?;
        x + h
    }
}

/// Final layer with AdaLN (scale and shift only, no gate)
#[derive(Debug)]
struct FinalLayer {
    adaln: FinalLayerAdaLN,
    linear: Linear,
}

impl FinalLayer {
    fn new(hidden_dim: usize, latent_dim: usize, vb: VarBuilder) -> Result<Self> {
        let adaln = FinalLayerAdaLN::new(hidden_dim, vb.pp("adaLN_modulation"))?;
        let linear = candle_nn::linear(hidden_dim, latent_dim, vb.pp("linear"))?;

        Ok(Self { adaln, linear })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        // Python order: shift, scale
        let (shift, scale) = self.adaln.forward(cond)?;

        // Python's FinalLayer applies modulation AFTER norm_final (which has no affine)
        // modulate(norm_final(x), shift, scale) = x * (1 + scale) + shift
        // But our FinalLayer doesn't have norm_final - we need to add it!
        // For now, apply modulation directly (may need fix later)
        let h = x.broadcast_mul(&(scale + 1.0)?)?;
        let h = h.broadcast_add(&shift)?;

        // Project to latent
        self.linear.forward(&h)
    }
}

/// FlowNet - Flow matching network for latent generation
#[derive(Debug)]
pub struct FlowNet {
    config: FlowNetConfig,
    cond_embed: Linear,
    input_proj: Linear,
    time_embed_0: TimeEmbedding,
    time_embed_1: TimeEmbedding,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
}

impl FlowNet {
    pub fn new(config: FlowNetConfig, vb: VarBuilder) -> Result<Self> {
        // Conditioning embedding from transformer hidden states
        let cond_embed = candle_nn::linear(config.cond_dim, config.hidden_dim, vb.pp("cond_embed"))?;

        // Input projection from latent space
        let input_proj = candle_nn::linear(config.latent_dim, config.hidden_dim, vb.pp("input_proj"))?;

        // Time embeddings (two separate ones in the model)
        let time_embed_0 = TimeEmbedding::new(config.hidden_dim, vb.pp("time_embed.0"))?;
        let time_embed_1 = TimeEmbedding::new(config.hidden_dim, vb.pp("time_embed.1"))?;

        // Residual blocks
        let mut res_blocks = Vec::with_capacity(config.num_res_blocks);
        for i in 0..config.num_res_blocks {
            res_blocks.push(ResBlock::new(config.hidden_dim, vb.pp(format!("res_blocks.{}", i)))?);
        }

        // Final layer
        let final_layer = FinalLayer::new(config.hidden_dim, config.latent_dim, vb.pp("final_layer"))?;

        Ok(Self {
            config,
            cond_embed,
            input_proj,
            time_embed_0,
            time_embed_1,
            res_blocks,
            final_layer,
        })
    }

    /// Generate latents using Lagrangian Self Distillation (LSD) flow matching
    ///
    /// LSD decoding (https://arxiv.org/pdf/2505.18825) uses TWO time values:
    /// - s (start time): where we currently are
    /// - t (target time): where we're going
    ///
    /// # Arguments
    /// * `hidden` - Conditioning from transformer [batch, seq, 1024]
    /// * `num_steps` - Number of flow steps (more = higher quality)
    /// * `temperature` - Sampling temperature
    pub fn generate(
        &self,
        hidden: &Tensor,
        num_steps: usize,
        _temperature: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden.dims3()?;

        // Get conditioning embedding
        let cond = self.cond_embed.forward(hidden)?;  // [batch, seq, 512]

        // DIAGNOSTIC: Log conditioning stats
        let cond_flat: Vec<f32> = cond.flatten_all()?.to_vec1()?;
        let c_mean = cond_flat.iter().sum::<f32>() / cond_flat.len() as f32;
        let c_std = (cond_flat.iter().map(|x| (x - c_mean).powi(2)).sum::<f32>() / cond_flat.len() as f32).sqrt();
        eprintln!("[FlowNet] conditioning: mean={:.4}, std={:.4}", c_mean, c_std);

        // Start from noise (x_0 in LSD notation)
        let mut current = Tensor::randn(
            0f32,
            1f32,
            (batch_size, seq_len, self.config.latent_dim),
            device,
        )?;

        // LSD decoding: integrate from s=0 toward t=1
        // For i in 0..num_steps:
        //   s = i / num_steps
        //   t = (i + 1) / num_steps
        //   flow_dir = v_t(s, t, current)
        //   current += flow_dir / num_steps
        let dt = 1.0 / num_steps as f32;

        for step in 0..num_steps {
            // LSD time progression
            let s = step as f32 / num_steps as f32;
            let t = (step + 1) as f32 / num_steps as f32;

            // Create time tensors
            let s_tensor = Tensor::full(s, (batch_size,), device)?;
            let t_tensor = Tensor::full(t, (batch_size,), device)?;

            // Get velocity prediction using both s and t
            let velocity = self.forward_step(&current, &cond, &s_tensor, &t_tensor)?;

            // DIAGNOSTIC: Log velocity stats at first and last steps
            if step == 0 || step == num_steps - 1 {
                let vel_flat: Vec<f32> = velocity.flatten_all()?.to_vec1()?;
                let v_mean = vel_flat.iter().sum::<f32>() / vel_flat.len() as f32;
                let v_max = vel_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                eprintln!("[FlowNet] step {} (s={:.3}, t={:.3}): vel mean={:.4}, max={:.4}", step, s, t, v_mean, v_max);
            }

            // LSD Euler step: current += flow_dir / num_steps
            current = (current + (velocity * dt as f64)?)?;
        }

        // DIAGNOSTIC: Log final latent stats
        let lat_flat: Vec<f32> = current.flatten_all()?.to_vec1()?;
        let l_mean = lat_flat.iter().sum::<f32>() / lat_flat.len() as f32;
        let l_std = (lat_flat.iter().map(|x| (x - l_mean).powi(2)).sum::<f32>() / lat_flat.len() as f32).sqrt();
        eprintln!("[FlowNet] final latent: mean={:.4}, std={:.4}", l_mean, l_std);

        Ok(current)
    }

    /// Single forward step of the flow network with LSD time conditioning
    ///
    /// Python's SimpleMLPAdaLN:
    /// 1. Embeds s with time_embed[0] and t with time_embed[1]
    /// 2. AVERAGES the two time embeddings together
    /// 3. Adds averaged time embedding to conditioning
    fn forward_step(&self, x: &Tensor, cond: &Tensor, s: &Tensor, t: &Tensor) -> Result<Tensor> {
        // Project input latent
        let h = self.input_proj.forward(x)?;

        // Embed start time (s) with time_embed_0
        let time_emb_s = self.time_embed_0.forward(s)?;
        // Embed target time (t) with time_embed_1
        let time_emb_t = self.time_embed_1.forward(t)?;

        // AVERAGE the two time embeddings (this is critical for LSD!)
        // Python: sum(time_embed[i](ts[i]) for i in range(num_time_conds)) / num_time_conds
        let time_emb_avg = ((time_emb_s + time_emb_t)? * 0.5)?;

        // DIAGNOSTIC: Check time embedding
        let s_val: f32 = s.to_vec1()?[0];
        let t_val: f32 = t.to_vec1()?[0];
        if s_val < 0.01 || t_val > 0.99 {
            let te_flat: Vec<f32> = time_emb_avg.flatten_all()?.to_vec1()?;
            let te_mean = te_flat.iter().sum::<f32>() / te_flat.len() as f32;
            let te_max = te_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[FlowNet] s={:.3}, t={:.3}: avg_time_emb mean={:.6}, max={:.4}", s_val, t_val, te_mean, te_max);
        }

        // Add averaged time embedding to input hidden states
        let h = h.broadcast_add(&time_emb_avg.unsqueeze(1)?)?;

        // Combine conditioning with averaged time embedding
        let cond_combined = cond.broadcast_add(&time_emb_avg.unsqueeze(1)?)?;

        // Residual blocks
        let mut h = h;
        for block in &self.res_blocks {
            h = block.forward(&h, &cond_combined)?;
        }

        // Final layer outputs velocity
        self.final_layer.forward(&h, &cond_combined)
    }

    pub fn config(&self) -> &FlowNetConfig {
        &self.config
    }
}
