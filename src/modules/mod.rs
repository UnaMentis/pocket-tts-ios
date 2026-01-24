//! Neural network modules for Pocket TTS
//!
//! These modules implement the building blocks for the FlowLM transformer,
//! FlowNet for latent generation, and Mimi VAE decoder.

pub mod attention;
pub mod embeddings;
pub mod mlp;
pub mod conv;
pub mod rotary;
pub mod layer_norm;
pub mod flownet;

#[cfg(test)]
mod tests;

pub use attention::{MultiHeadAttention, CausalSelfAttention};
pub use embeddings::{TextEmbedding, VoiceEmbedding};
pub use mlp::{MLP, GatedMLP};
pub use conv::{Conv1d, ConvTranspose1d, CausalConv1d};
pub use rotary::RotaryEmbedding;
pub use layer_norm::RMSNorm;
pub use flownet::{FlowNet, FlowNetConfig};
