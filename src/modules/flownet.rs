//! FlowNet module for Kyutai Pocket TTS
//!
//! Flow matching network that generates latent representations from
//! transformer hidden states. Uses AdaLN (adaptive layer normalization)
//! for conditioning on time and hidden states.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::layer_norm::{layer_norm_no_affine, LayerNorm};

/// FlowNet configuration
#[derive(Debug, Clone)]
pub struct FlowNetConfig {
    pub hidden_dim: usize,     // 512
    pub cond_dim: usize,       // 1024 (from transformer)
    pub latent_dim: usize,     // 32
    pub num_res_blocks: usize, // 6
    pub time_embed_dim: usize, // 256 (freqs * 2)
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

        // Learnable scale parameter for RMSNorm (NOT just a simple multiply!)
        let alpha = vb.get((hidden_dim,), "mlp.3.alpha")?;

        // DIAGNOSTIC: Log alpha range
        let alpha_vec: Vec<f32> = alpha.to_vec1()?;
        let a_min = alpha_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let a_max = alpha_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[TimeEmbed] alpha range: [{:.4}, {:.4}]", a_min, a_max);

        Ok(Self {
            freqs,
            mlp_0,
            mlp_2,
            alpha,
        })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // Create sinusoidal embedding from time
        // CRITICAL: Python uses [cos, sin] order, not [sin, cos]!
        let t_expanded = t.unsqueeze(1)?; // [batch, 1]
        let freqs_expanded = self.freqs.unsqueeze(0)?; // [1, 128]
        let angles = t_expanded.broadcast_mul(&freqs_expanded)?;

        let cos_emb = angles.cos()?;
        let sin_emb = angles.sin()?;
        let time_emb = Tensor::cat(&[cos_emb, sin_emb], 1)?; // [batch, 256] - COS first!

        // MLP with SiLU (not GELU!) - matches Python TimestepEmbedder
        let x = self.mlp_0.forward(&time_emb)?;
        let x = candle_nn::ops::silu(&x)?; // Python uses SiLU, not GELU
        let x = self.mlp_2.forward(&x)?;

        // Apply RMSNorm with learnable alpha (NOT just x * alpha!)
        // Python's RMSNorm: y = x * alpha / sqrt(var + eps)
        // where var = x.var(dim=-1, keepdim=True, unbiased=True) [Python default]
        //
        // variance = mean((x - mean(x))^2) * n / (n-1) for unbiased=True
        let eps = 1e-5f64;
        let n = x.dim(1)? as f64; // hidden_dim = 512
        let x_mean = x.mean_keepdim(1)?; // mean along hidden dim
        let x_centered = x.broadcast_sub(&x_mean)?;
        let x_centered_sq = x_centered.sqr()?;
        // unbiased variance: sum / (n-1)
        let var_sum = x_centered_sq.sum_keepdim(1)?;
        let var = (var_sum / (n - 1.0))?;
        let sqrt_var = (var + eps)?.sqrt()?;
        let x_normed = x.broadcast_div(&sqrt_var)?;
        x_normed.broadcast_mul(&self.alpha)
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
        Ok((chunks[0].clone(), chunks[1].clone(), chunks[2].clone())) // shift, scale, gate
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
        Ok((chunks[0].clone(), chunks[1].clone())) // shift, scale
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

        Ok(Self {
            in_ln,
            mlp_0,
            mlp_2,
            adaln,
        })
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
        let h = candle_nn::ops::silu(&h)?; // SiLU, not GELU
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

        // DIAGNOSTIC: Check shift/scale
        static DIAG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = DIAG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count < 3 {
            let shift_flat: Vec<f32> = shift.flatten_all()?.to_vec1()?;
            let scale_flat: Vec<f32> = scale.flatten_all()?.to_vec1()?;
            let shift_min = shift_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let shift_max = shift_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale_min = scale_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let scale_max = scale_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[FinalLayer] shift range=[{:.4}, {:.4}], scale range=[{:.4}, {:.4}]",
                shift_min, shift_max, scale_min, scale_max
            );
        }

        // Python's FinalLayer applies modulation AFTER norm_final (which has no affine)
        // modulate(norm_final(x), shift, scale) = norm(x) * (1 + scale) + shift
        // norm_final uses eps=1e-6 (typical for layer norm in this codebase)
        let h = layer_norm_no_affine(x, 1e-6)?;
        let h = h.broadcast_mul(&(scale + 1.0)?)?;
        let h = h.broadcast_add(&shift)?;

        if count < 3 {
            let h_flat: Vec<f32> = h.flatten_all()?.to_vec1()?;
            let h_min = h_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let h_max = h_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[FinalLayer] after modulation: range=[{:.4}, {:.4}]", h_min, h_max);
        }

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
    pub fn generate(&self, hidden: &Tensor, num_steps: usize, temperature: f32, device: &Device) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden.dims3()?;

        // Get conditioning embedding
        let cond = self.cond_embed.forward(hidden)?; // [batch, seq, 512]

        // DIAGNOSTIC: Log conditioning stats
        let cond_flat: Vec<f32> = cond.flatten_all()?.to_vec1()?;
        let c_mean = cond_flat.iter().sum::<f32>() / cond_flat.len() as f32;
        let c_std = (cond_flat.iter().map(|x| (x - c_mean).powi(2)).sum::<f32>() / cond_flat.len() as f32).sqrt();
        eprintln!(
            "[FlowNet] conditioning (after cond_embed) first 8: {:?}",
            &cond_flat[..8.min(cond_flat.len())]
        );
        eprintln!("[FlowNet] conditioning: mean={:.4}, std={:.4}", c_mean, c_std);

        // Start from noise (x_0 in LSD notation)
        // Python uses: std = temp^0.5, and samples from Normal(0, std)
        // Python default temperature = 0.7, so std ≈ 0.8367
        let std = temperature.sqrt();
        let mut current = Tensor::randn(
            0f32,
            std,
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
                if step == 0 {
                    eprintln!("[FlowNet-Step0] velocity first 8: {:?}", &vel_flat[..8.min(vel_flat.len())]);
                }
                eprintln!(
                    "[FlowNet] step {} (s={:.3}, t={:.3}): vel mean={:.4}, max={:.4}",
                    step, s, t, v_mean, v_max
                );
            }

            // LSD Euler step: current += flow_dir / num_steps
            current = (current + (velocity * dt as f64)?)?;
        }

        // DIAGNOSTIC: Log final latent stats
        static FLOWNET_STEP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let step = FLOWNET_STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let lat_flat: Vec<f32> = current.flatten_all()?.to_vec1()?;
        let l_mean = lat_flat.iter().sum::<f32>() / lat_flat.len() as f32;
        let l_std = (lat_flat.iter().map(|x| (x - l_mean).powi(2)).sum::<f32>() / lat_flat.len() as f32).sqrt();
        if step == 0 {
            eprintln!("[FlowNet-Step0] final latent first 8: {:?}", &lat_flat[..8.min(lat_flat.len())]);
            eprintln!("[FlowNet-Step0] mean={:.4}, std={:.4}", l_mean, l_std);
            // Python: first 8: [-1.7723137, -0.8588331, 0.23708725, 1.6371508, -1.6427871, 1.103927, -1.926453, 1.8970976]
            // Python: mean=-0.051385, std=1.170856
        } else {
            eprintln!("[FlowNet] final latent: mean={:.4}, std={:.4}", l_mean, l_std);
        }

        Ok(current)
    }

    /// Single forward step of the flow network with LSD time conditioning
    ///
    /// Python's SimpleMLPAdaLN:
    /// 1. Embeds s with time_embed[0] and t with time_embed[1]
    /// 2. AVERAGES the two time embeddings together
    /// 3. Adds averaged time embedding to conditioning (NOT to input!)
    fn forward_step(&self, x: &Tensor, cond: &Tensor, s: &Tensor, t: &Tensor) -> Result<Tensor> {
        // Project input latent (NO time added here - Python doesn't add time to x!)
        let h = self.input_proj.forward(x)?;

        let s_val: f32 = s.to_vec1()?[0];
        let t_val: f32 = t.to_vec1()?[0];

        // DIAGNOSTIC: Log input projection output at step 0
        if s_val < 0.01 {
            let h_flat: Vec<f32> = h.flatten_all()?.to_vec1()?;
            eprintln!("[FlowNet-Step0] after input_proj first 8: {:?}", &h_flat[..8.min(h_flat.len())]);
        }

        // Embed start time (s) with time_embed_0
        let time_emb_s = self.time_embed_0.forward(s)?;
        // Embed target time (t) with time_embed_1
        let time_emb_t = self.time_embed_1.forward(t)?;

        // DIAGNOSTIC: Log individual time embeddings at step 0
        if s_val < 0.01 {
            let te_s_flat: Vec<f32> = time_emb_s.flatten_all()?.to_vec1()?;
            let te_t_flat: Vec<f32> = time_emb_t.flatten_all()?.to_vec1()?;
            eprintln!(
                "[FlowNet-Step0] time_embed_s (s=0) first 8: {:?}",
                &te_s_flat[..8.min(te_s_flat.len())]
            );
            eprintln!(
                "[FlowNet-Step0] time_embed_t (t=1) first 8: {:?}",
                &te_t_flat[..8.min(te_t_flat.len())]
            );
        }

        // AVERAGE the two time embeddings (this is critical for LSD!)
        // Python: sum(time_embed[i](ts[i]) for i in range(num_time_conds)) / num_time_conds
        let time_emb_avg = ((time_emb_s + time_emb_t)? * 0.5)?;

        // DIAGNOSTIC: Check time embedding
        if s_val < 0.01 || t_val > 0.99 {
            let te_flat: Vec<f32> = time_emb_avg.flatten_all()?.to_vec1()?;
            let te_mean = te_flat.iter().sum::<f32>() / te_flat.len() as f32;
            let te_max = te_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[FlowNet] s={:.3}, t={:.3}: avg_time_emb mean={:.6}, max={:.4}",
                s_val, t_val, te_mean, te_max
            );
            eprintln!("[FlowNet-Step0] avg_time_emb first 8: {:?}", &te_flat[..8.min(te_flat.len())]);
        }

        // Python: y = t_combined + c (time only added to conditioning, NOT to x!)
        // The time embedding is broadcast to match [batch, seq, hidden]
        let cond_combined = cond.broadcast_add(&time_emb_avg.unsqueeze(1)?)?;

        // DIAGNOSTIC: Log combined conditioning at step 0
        if s_val < 0.01 {
            let cc_flat: Vec<f32> = cond_combined.flatten_all()?.to_vec1()?;
            eprintln!(
                "[FlowNet-Step0] cond_combined (c + time) first 8: {:?}",
                &cc_flat[..8.min(cc_flat.len())]
            );
        }

        // Residual blocks
        let mut h = h;
        for (i, block) in self.res_blocks.iter().enumerate() {
            h = block.forward(&h, &cond_combined)?;
            // DIAGNOSTIC: Check for value explosion
            if s_val < 0.01 && i == 0 {
                let h_flat: Vec<f32> = h.flatten_all()?.to_vec1()?;
                let h_mean = h_flat.iter().sum::<f32>() / h_flat.len() as f32;
                let h_max = h_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let h_min = h_flat.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!(
                    "[FlowNet] after ResBlock 0: mean={:.4}, range=[{:.4}, {:.4}]",
                    h_mean, h_min, h_max
                );
            }
        }

        // Final layer outputs velocity
        let velocity = self.final_layer.forward(&h, &cond_combined)?;
        if s_val < 0.01 {
            let v_flat: Vec<f32> = velocity.flatten_all()?.to_vec1()?;
            let v_mean = v_flat.iter().sum::<f32>() / v_flat.len() as f32;
            let v_min = v_flat.iter().cloned().fold(f32::INFINITY, f32::min);
            let v_max = v_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[FlowNet] velocity after FinalLayer: mean={:.4}, range=[{:.4}, {:.4}]",
                v_mean, v_min, v_max
            );
        }
        Ok(velocity)
    }

    pub fn config(&self) -> &FlowNetConfig {
        &self.config
    }
}
