//! Neural network modules for Pocket TTS
//!
//! These modules implement the building blocks for the FlowLM transformer,
//! FlowNet for latent generation, and Mimi VAE decoder.

pub mod attention;
pub mod conv;
pub mod embeddings;
pub mod flownet;
pub mod layer_norm;
pub mod mlp;
pub mod rotary;
pub mod streaming;

#[cfg(test)]
mod tests;

pub use attention::{CausalSelfAttention, MultiHeadAttention};
pub use conv::{CausalConv1d, Conv1d, ConvTranspose1d, PadMode, StreamableConv1d, StreamableConvTranspose1d};
pub use embeddings::{TextEmbedding, VoiceEmbedding};
pub use flownet::{FlowNet, FlowNetConfig};
pub use layer_norm::RMSNorm;
pub use mlp::{GatedMLP, MLP};
pub use rotary::RotaryEmbedding;
pub use streaming::{StreamTensor, StreamingModule, TensorPadding};
