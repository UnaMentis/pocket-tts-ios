//! Model implementations for Pocket TTS
//!
//! Architecture:
//! - FlowLM: 6-layer transformer backbone with FlowNet (~80M params)
//! - MimiDecoder: VAE decoder with transformer and SEANet (~37M params)

pub mod flowlm;
pub mod mimi;
pub mod pocket_tts;

pub use flowlm::FlowLM;
pub use mimi::MimiDecoder;
pub use pocket_tts::PocketTTSModel;
