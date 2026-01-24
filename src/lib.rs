//! Kyutai Pocket TTS for iOS
//!
//! This crate provides native iOS inference for Kyutai Pocket TTS using Candle.
//! It exposes a UniFFI interface for Swift integration.

pub mod config;
pub mod error;
pub mod models;
pub mod modules;
pub mod tokenizer;
pub mod audio;
pub mod engine;

// Test modules
#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod audio_tests;
#[cfg(test)]
mod error_tests;
#[cfg(test)]
mod lib_tests;

pub use config::{TTSConfig, PocketVoiceInfo};
pub use error::PocketTTSError;
pub use engine::PocketTTSEngine;

// UniFFI scaffolding
uniffi::include_scaffolding!("pocket_tts");

/// Library version
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get available voices
pub fn available_voices() -> Vec<PocketVoiceInfo> {
    vec![
        PocketVoiceInfo { index: 0, name: "Alba".into(), gender: "female".into(), description: "Clear, neutral female voice".into() },
        PocketVoiceInfo { index: 1, name: "Marius".into(), gender: "male".into(), description: "Warm male voice".into() },
        PocketVoiceInfo { index: 2, name: "Javert".into(), gender: "male".into(), description: "Authoritative male voice".into() },
        PocketVoiceInfo { index: 3, name: "Jean".into(), gender: "male".into(), description: "Gentle male voice".into() },
        PocketVoiceInfo { index: 4, name: "Fantine".into(), gender: "female".into(), description: "Soft female voice".into() },
        PocketVoiceInfo { index: 5, name: "Cosette".into(), gender: "female".into(), description: "Young female voice".into() },
        PocketVoiceInfo { index: 6, name: "Eponine".into(), gender: "female".into(), description: "Expressive female voice".into() },
        PocketVoiceInfo { index: 7, name: "Azelma".into(), gender: "female".into(), description: "Bright female voice".into() },
    ]
}

/// Audio chunk for streaming synthesis
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub audio_data: Vec<u8>,
    pub sample_rate: u32,
    pub is_final: bool,
}

/// Synthesis result
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    pub audio_data: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u32,
    pub duration_seconds: f64,
}

/// Event handler trait for streaming synthesis
pub trait TTSEventHandler: Send + Sync {
    fn on_audio_chunk(&self, chunk: AudioChunk);
    fn on_progress(&self, progress: f32);
    fn on_complete(&self);
    fn on_error(&self, message: String);
}
