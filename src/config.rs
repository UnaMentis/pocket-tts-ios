//! Configuration types for Pocket TTS

use serde::{Deserialize, Serialize};

/// Voice information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketVoiceInfo {
    pub index: u32,
    pub name: String,
    pub gender: String,
    pub description: String,
}

/// TTS synthesis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSConfig {
    /// Voice index (0-7 for built-in voices)
    pub voice_index: u32,

    /// Temperature for sampling (0.0-1.0)
    /// Lower = more deterministic, Higher = more creative
    pub temperature: f32,

    /// Top-P (nucleus) sampling threshold (0.1-1.0)
    pub top_p: f32,

    /// Speech speed multiplier (0.5-2.0)
    pub speed: f32,

    /// Number of consistency steps (1-4)
    /// Higher = better quality, slower
    pub consistency_steps: u32,

    /// Use fixed seed for reproducibility
    pub use_fixed_seed: bool,

    /// Random seed (if use_fixed_seed is true)
    pub seed: u32,
}

impl Default for TTSConfig {
    fn default() -> Self {
        Self {
            voice_index: 0,          // Alba
            temperature: 0.7,
            top_p: 0.9,
            speed: 1.0,
            consistency_steps: 2,
            use_fixed_seed: false,
            seed: 42,
        }
    }
}

impl TTSConfig {
    /// Create a low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            consistency_steps: 1,
            ..Default::default()
        }
    }

    /// Create a high-quality configuration
    pub fn high_quality() -> Self {
        Self {
            consistency_steps: 4,
            temperature: 0.5,
            ..Default::default()
        }
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), String> {
        if self.voice_index > 7 {
            return Err(format!("Voice index must be 0-7, got {}", self.voice_index));
        }
        if self.temperature < 0.0 || self.temperature > 1.0 {
            return Err(format!("Temperature must be 0.0-1.0, got {}", self.temperature));
        }
        if self.top_p < 0.1 || self.top_p > 1.0 {
            return Err(format!("Top-P must be 0.1-1.0, got {}", self.top_p));
        }
        if self.speed < 0.5 || self.speed > 2.0 {
            return Err(format!("Speed must be 0.5-2.0, got {}", self.speed));
        }
        if self.consistency_steps < 1 || self.consistency_steps > 4 {
            return Err(format!("Consistency steps must be 1-4, got {}", self.consistency_steps));
        }
        Ok(())
    }
}

/// Model configuration loaded from manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub version: String,
    pub model_id: String,
    pub license: String,
    pub parameters: u64,
    pub sample_rate: u32,
    pub frame_rate: f32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self {
            version: "1.0.2".into(),
            model_id: "kyutai/pocket-tts".into(),
            license: "CC-BY-4.0".into(),
            parameters: 117_856_642,
            sample_rate: 24000,
            frame_rate: 12.5,
            hidden_size: 1024,
            num_layers: 6,
            num_heads: 16,
            vocab_size: 32000,
        }
    }
}
