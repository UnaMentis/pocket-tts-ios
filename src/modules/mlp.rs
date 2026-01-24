//! MLP (Feed-Forward) modules
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// Standard MLP with GELU activation
#[derive(Debug)]
pub struct MLP {
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let up_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self { up_proj, down_proj })
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.up_proj.forward(x)?;
        let x = x.gelu_erf()?;
        self.down_proj.forward(&x)
    }
}

/// Simple 2-layer MLP (Kyutai Pocket style)
/// Uses linear1/linear2 naming with GELU activation
#[derive(Debug)]
pub struct SimpleMLP {
    linear1: Linear,
    linear2: Linear,
}

impl SimpleMLP {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // Kyutai Pocket uses linear1 (up) and linear2 (down) naming - NO BIAS
        let linear1 = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("linear2"))?;

        Ok(Self { linear1, linear2 })
    }
}

impl Module for SimpleMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.gelu_erf()?;
        self.linear2.forward(&x)
    }
}

/// Gated MLP (SwiGLU variant used in modern transformers)
#[derive(Debug)]
pub struct GatedMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GatedMLP {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for GatedMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

/// MLP Sampler for consistency sampling in Pocket TTS
#[derive(Debug)]
pub struct MLPSampler {
    layers: Vec<Linear>,
    num_steps: usize,
}

impl MLPSampler {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        num_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let in_dim = if i == 0 { input_dim } else { hidden_dim };
            let out_dim = if i == num_layers - 1 { output_dim } else { hidden_dim };
            layers.push(candle_nn::linear(in_dim, out_dim, vb.pp(format!("layer_{}", i)))?);
        }

        Ok(Self { layers, num_steps: 2 })
    }

    pub fn set_num_steps(&mut self, steps: usize) {
        self.num_steps = steps.clamp(1, 4);
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden)?;
            if i < self.layers.len() - 1 {
                hidden = hidden.gelu_erf()?;
            }
        }

        Ok(hidden)
    }

    /// Consistency sampling with multiple refinement steps
    pub fn sample(&self, x: &Tensor, temperature: f32, top_p: f32) -> Result<Tensor> {
        let mut current = x.clone();

        for _ in 0..self.num_steps {
            let logits = self.forward(&current)?;

            // Apply temperature
            let scaled_logits = if temperature != 1.0 {
                (logits / temperature as f64)?
            } else {
                logits
            };

            // Top-p sampling
            current = self.top_p_sample(&scaled_logits, top_p)?;
        }

        Ok(current)
    }

    fn top_p_sample(&self, logits: &Tensor, _top_p: f32) -> Result<Tensor> {
        // Softmax to get probabilities
        let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)?;

        // For simplicity, just return the argmax weighted by probs
        // A full implementation would do proper nucleus sampling
        probs.argmax(candle_core::D::Minus1)?.to_dtype(logits.dtype())
    }
}
