//! Complete Pocket TTS Model
//!
//! Combines FlowLM transformer (with FlowNet) and Mimi decoder
//! into a complete text-to-speech pipeline.
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::flowlm::{FlowLM, FlowLMConfig};
use super::mimi::{MimiConfig, MimiDecoder};
use crate::config::TTSConfig;
use crate::modules::embeddings::{VoiceBank, VoiceEmbedding};
use crate::tokenizer::PocketTokenizer;
use crate::error::PocketTTSError;

/// Complete Pocket TTS Model
pub struct PocketTTSModel {
    flowlm: FlowLM,
    mimi: MimiDecoder,
    tokenizer: PocketTokenizer,
    voice_bank: VoiceBank,
    device: Device,
    config: TTSConfig,
    custom_voice: Option<VoiceEmbedding>,
}

impl PocketTTSModel {
    /// Load model from directory containing all components
    pub fn load<P: AsRef<Path>>(model_dir: P, device: &Device) -> std::result::Result<Self, PocketTTSError> {
        let model_dir = model_dir.as_ref();

        // Load model weights using memory-mapped file
        let model_path = model_dir.join("model.safetensors");

        // Create VarBuilder from safetensors file
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, device)
                .map_err(|e| PocketTTSError::ModelLoadFailed(e.to_string()))?
        };

        // Load tokenizer (SentencePiece .model format)
        let tokenizer_path = model_dir.join("tokenizer.model");
        let tokenizer = PocketTokenizer::from_file(&tokenizer_path)?;

        // Load voice embeddings
        let voices_dir = model_dir.join("voices");
        let voice_bank = VoiceBank::load_from_dir(&voices_dir, device)
            .map_err(|e| PocketTTSError::ModelLoadFailed(format!("Failed to load voices: {}", e)))?;

        // Initialize model components
        let flowlm_config = FlowLMConfig::default();
        let flowlm = FlowLM::new(flowlm_config.clone(), vb.pp("flow_lm"), device)
            .map_err(|e| PocketTTSError::ModelLoadFailed(format!("FlowLM: {}", e)))?;

        let mimi_config = MimiConfig {
            latent_dim: flowlm_config.latent_dim,
            ..MimiConfig::default()
        };
        let mimi = MimiDecoder::new(mimi_config, vb.pp("mimi"))
            .map_err(|e| PocketTTSError::ModelLoadFailed(format!("Mimi: {}", e)))?;

        Ok(Self {
            flowlm,
            mimi,
            tokenizer,
            voice_bank,
            device: device.clone(),
            config: TTSConfig::default(),
            custom_voice: None,
        })
    }

    /// Configure synthesis parameters
    pub fn configure(&mut self, config: TTSConfig) -> std::result::Result<(), PocketTTSError> {
        config.validate().map_err(PocketTTSError::InvalidConfig)?;
        self.config = config;
        Ok(())
    }

    /// Set custom voice from reference audio embedding
    pub fn set_custom_voice(&mut self, embedding: VoiceEmbedding) {
        self.custom_voice = Some(embedding);
    }

    /// Clear custom voice (use built-in)
    pub fn clear_custom_voice(&mut self) {
        self.custom_voice = None;
    }

    /// Synthesize text to audio
    pub fn synthesize(&mut self, text: &str) -> std::result::Result<Vec<f32>, PocketTTSError> {
        eprintln!("[PocketTTS] synthesize called with text len: {}", text.len());

        // Tokenize text
        let token_ids = self.tokenizer.encode(text)?;
        eprintln!("[PocketTTS] tokenized to {} tokens: {:?}", token_ids.len(), token_ids);

        // Create tensor
        let token_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, token_ids.len()),
            &self.device,
        ).map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        eprintln!("[PocketTTS] token tensor shape: {:?}", token_tensor.dims());

        // Get voice embedding
        let voice = if let Some(ref custom) = self.custom_voice {
            Some(custom)
        } else {
            self.voice_bank.get(self.config.voice_index as usize)
        };
        eprintln!("[PocketTTS] voice embedding loaded: {}", voice.is_some());

        // Reset caches for new sequence
        self.flowlm.reset_cache();

        // Generate latents with FlowLM + FlowNet
        // Reference implementation uses lsd_decode_steps = 1 (consistency model)
        // Single step is sufficient as the model is trained with consistency distillation
        let num_flow_steps = 1;
        eprintln!("[PocketTTS] generating latents with {} flow step (consistency model)", num_flow_steps);
        let latents = self.flowlm.generate_latents(
            &token_tensor,
            voice,
            num_flow_steps,
            self.config.temperature,
        ).map_err(|e| PocketTTSError::InferenceFailed(format!("FlowLM: {}", e)))?;
        eprintln!("[PocketTTS] latents shape: {:?}", latents.dims());

        // DIAGNOSTIC: Log latent statistics to verify FlowLM output quality
        let latents_flat: Vec<f32> = latents.flatten_all()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?
            .to_vec1()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let lat_mean = latents_flat.iter().sum::<f32>() / latents_flat.len() as f32;
        let lat_max = latents_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lat_min = latents_flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let lat_std = (latents_flat.iter().map(|x| (x - lat_mean).powi(2)).sum::<f32>() / latents_flat.len() as f32).sqrt();
        eprintln!("[PocketTTS] latent stats: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", lat_mean, lat_std, lat_min, lat_max);

        // Decode to audio
        eprintln!("[PocketTTS] decoding with Mimi...");
        let audio = self.mimi.forward(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Mimi: {}", e)))?;
        eprintln!("[PocketTTS] audio tensor shape: {:?}", audio.dims());

        // Convert to Vec<f32>
        let audio = audio.squeeze(0)
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let audio_vec: Vec<f32> = audio.to_vec1()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        // Debug: check amplitude to verify audio has signal
        let audio_max = audio_vec.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let audio_mean = audio_vec.iter().map(|s| s.abs()).sum::<f32>() / audio_vec.len() as f32;
        eprintln!("[PocketTTS] final audio samples: {} (expect ~78720 for test phrase)", audio_vec.len());
        eprintln!("[PocketTTS] audio max amplitude: {:.4} (expect > 0.01)", audio_max);
        eprintln!("[PocketTTS] audio mean amplitude: {:.4}", audio_mean);

        Ok(audio_vec)
    }

    /// Streaming synthesis - yields audio chunks
    pub fn synthesize_streaming<F>(
        &mut self,
        text: &str,
        chunk_callback: F,
    ) -> std::result::Result<(), PocketTTSError>
    where
        F: Fn(&[f32], bool) -> bool, // Returns false to stop
    {
        // Tokenize text
        let token_ids = self.tokenizer.encode(text)?;

        // Get voice embedding
        let voice = if let Some(ref custom) = self.custom_voice {
            Some(custom)
        } else {
            self.voice_bank.get(self.config.voice_index as usize)
        };

        // Reset caches
        self.flowlm.reset_cache();

        // Process in chunks for streaming
        let chunk_size = 32; // tokens per chunk
        let overlap_samples = (self.mimi.sample_rate() as f32 * 0.05) as usize; // 50ms overlap
        let mut previous_tail: Option<Tensor> = None;
        let num_flow_steps = self.config.consistency_steps.max(1) as usize;

        for (i, chunk) in token_ids.chunks(chunk_size).enumerate() {
            let is_last = i == (token_ids.len() / chunk_size);

            // Create tensor for chunk
            let token_tensor = Tensor::from_vec(
                chunk.iter().map(|&id| id as i64).collect::<Vec<_>>(),
                (1, chunk.len()),
                &self.device,
            ).map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

            // Generate latents with FlowLM + FlowNet
            let latents = self.flowlm.generate_latents(
                &token_tensor,
                voice,
                num_flow_steps,
                self.config.temperature,
            ).map_err(|e| PocketTTSError::InferenceFailed(format!("FlowLM: {}", e)))?;

            // Decode with overlap-add
            let (audio, tail) = self.mimi.decode_streaming(
                &latents,
                overlap_samples,
                previous_tail.as_ref(),
            ).map_err(|e| PocketTTSError::InferenceFailed(format!("Mimi: {}", e)))?;

            previous_tail = Some(tail);

            // Convert to Vec<f32>
            let audio = audio.squeeze(0)
                .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
            let audio_vec: Vec<f32> = audio.to_vec1()
                .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

            // Callback with audio chunk
            if !chunk_callback(&audio_vec, is_last) {
                break; // User requested stop
            }
        }

        Ok(())
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.mimi.sample_rate() as u32
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> u64 {
        117_856_642 // From model manifest
    }

    /// Get model version
    pub fn version(&self) -> &str {
        "1.0.2"
    }
}

impl std::fmt::Debug for PocketTTSModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PocketTTSModel")
            .field("version", &self.version())
            .field("parameter_count", &self.parameter_count())
            .field("sample_rate", &self.sample_rate())
            .field("voice_count", &self.voice_bank.len())
            .finish()
    }
}
