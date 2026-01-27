//! PocketTTSEngine - UniFFI interface implementation
//!
//! This module implements the public API exposed to Swift via UniFFI.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_core::Device;

use crate::audio;
use crate::config::TTSConfig;
use crate::error::PocketTTSError;
use crate::models::PocketTTSModel;
use crate::{AudioChunk, SynthesisResult, TTSEventHandler};

/// Main TTS engine exposed to Swift
pub struct PocketTTSEngine {
    model: Arc<Mutex<Option<PocketTTSModel>>>,
    #[allow(dead_code)]
    model_path: PathBuf,
    config: Mutex<TTSConfig>,
    is_cancelled: Mutex<bool>,
}

impl PocketTTSEngine {
    /// Create new engine and load model from path
    pub fn new(model_path: String) -> Result<Self, PocketTTSError> {
        let path = PathBuf::from(&model_path);

        if !path.exists() {
            return Err(PocketTTSError::IoError(format!("Model path does not exist: {}", model_path)));
        }

        // Use CPU device (Metal not supported on iOS in Candle)
        let device = Device::Cpu;

        // Load model
        let model = PocketTTSModel::load(&path, &device)?;

        Ok(Self {
            model: Arc::new(Mutex::new(Some(model))),
            model_path: path,
            config: Mutex::new(TTSConfig::default()),
            is_cancelled: Mutex::new(false),
        })
    }

    /// Check if model is loaded and ready
    pub fn is_ready(&self) -> bool {
        self.model.lock().map(|m| m.is_some()).unwrap_or(false)
    }

    /// Get model version
    pub fn model_version(&self) -> String {
        "1.0.2".to_string()
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> u64 {
        117_856_642
    }

    /// Configure synthesis parameters
    pub fn configure(&self, config: TTSConfig) -> Result<(), PocketTTSError> {
        config.validate().map_err(PocketTTSError::InvalidConfig)?;

        if let Ok(mut model_guard) = self.model.lock() {
            if let Some(ref mut model) = *model_guard {
                model.configure(config.clone())?;
            }
        }

        *self.config.lock().unwrap() = config;
        Ok(())
    }

    /// Get current configuration
    pub fn get_config(&self) -> TTSConfig {
        self.config.lock().unwrap().clone()
    }

    /// Synchronous synthesis - returns complete audio
    pub fn synthesize(&self, text: String) -> Result<SynthesisResult, PocketTTSError> {
        let mut model_guard = self
            .model
            .lock()
            .map_err(|_| PocketTTSError::InferenceFailed("Lock error".into()))?;

        let model = model_guard.as_mut().ok_or(PocketTTSError::ModelNotLoaded)?;

        // Generate audio
        let samples = model.synthesize(&text)?;

        // Convert to WAV bytes
        let wav_data = audio::samples_to_wav(&samples, model.sample_rate())?;

        let duration = audio::duration_seconds(samples.len(), model.sample_rate());

        Ok(SynthesisResult {
            audio_data: wav_data,
            sample_rate: model.sample_rate(),
            channels: 1,
            duration_seconds: duration,
        })
    }

    /// Synchronous synthesis with specific voice
    pub fn synthesize_with_voice(&self, text: String, voice_index: u32) -> Result<SynthesisResult, PocketTTSError> {
        // Temporarily set voice
        let original_config = self.get_config();
        let mut new_config = original_config.clone();
        new_config.voice_index = voice_index;
        self.configure(new_config)?;

        let result = self.synthesize(text);

        // Restore original config
        self.configure(original_config)?;

        result
    }

    /// Start streaming synthesis
    pub fn start_streaming(&self, text: String, handler: Box<dyn TTSEventHandler>) -> Result<(), PocketTTSError> {
        // Reset cancellation flag
        *self.is_cancelled.lock().unwrap() = false;

        let mut model_guard = self
            .model
            .lock()
            .map_err(|_| PocketTTSError::InferenceFailed("Lock error".into()))?;

        let model = model_guard.as_mut().ok_or(PocketTTSError::ModelNotLoaded)?;

        let sample_rate = model.sample_rate();
        let is_cancelled = Arc::new(Mutex::new(false));
        let is_cancelled_clone = is_cancelled.clone();

        // Store reference for cancellation
        {
            let mut cancelled = self.is_cancelled.lock().unwrap();
            *cancelled = false;
        }

        // Streaming synthesis with callback
        let result = model.synthesize_streaming(&text, |samples, is_final| {
            // Check cancellation
            if *is_cancelled_clone.lock().unwrap() {
                return false;
            }

            // Convert to bytes
            let audio_bytes = audio::samples_to_bytes(samples);

            // Create chunk
            let chunk = AudioChunk {
                audio_data: audio_bytes,
                sample_rate,
                is_final,
            };

            // Calculate progress (approximate)
            let progress = if is_final { 1.0 } else { 0.5 };
            handler.on_progress(progress);

            // Send chunk
            handler.on_audio_chunk(chunk);

            if is_final {
                handler.on_complete();
            }

            true // Continue
        });

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                handler.on_error(e.to_string());
                Err(e)
            },
        }
    }

    /// Start true streaming synthesis with optimized TTFA
    ///
    /// Unlike `start_streaming` which chunks by tokens, this method:
    /// 1. Tokenizes ALL text upfront
    /// 2. Generates latents using streaming FlowLM
    /// 3. Decodes and yields audio in small batches
    ///
    /// This achieves lower TTFA by avoiding token chunking overhead.
    pub fn start_true_streaming(&self, text: String, handler: Box<dyn TTSEventHandler>) -> Result<(), PocketTTSError> {
        // Reset cancellation flag
        *self.is_cancelled.lock().unwrap() = false;

        let mut model_guard = self
            .model
            .lock()
            .map_err(|_| PocketTTSError::InferenceFailed("Lock error".into()))?;

        let model = model_guard.as_mut().ok_or(PocketTTSError::ModelNotLoaded)?;

        let sample_rate = model.sample_rate();
        let is_cancelled = Arc::new(Mutex::new(false));
        let is_cancelled_clone = is_cancelled.clone();

        // True streaming synthesis with callback
        let result = model.synthesize_true_streaming(&text, |samples, is_final| {
            // Check cancellation
            if *is_cancelled_clone.lock().unwrap() {
                return false;
            }

            // Convert to bytes
            let audio_bytes = audio::samples_to_bytes(samples);

            // Create chunk
            let chunk = AudioChunk {
                audio_data: audio_bytes,
                sample_rate,
                is_final,
            };

            // Calculate progress (approximate)
            let progress = if is_final { 1.0 } else { 0.5 };
            handler.on_progress(progress);

            // Send chunk
            handler.on_audio_chunk(chunk);

            if is_final {
                handler.on_complete();
            }

            true // Continue
        });

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                handler.on_error(e.to_string());
                Err(e)
            },
        }
    }

    /// Cancel ongoing synthesis
    pub fn cancel(&self) {
        *self.is_cancelled.lock().unwrap() = true;
    }

    /// Set reference audio for voice cloning
    pub fn set_reference_audio(&self, audio_data: Vec<u8>, sample_rate: u32) -> Result<(), PocketTTSError> {
        // Convert audio bytes to samples
        let samples = audio::bytes_to_samples(&audio_data);

        // Resample to 24kHz if needed
        let _samples = if sample_rate != 24000 {
            audio::resample(&samples, sample_rate, 24000)?
        } else {
            samples
        };

        // Extract voice embedding from reference audio
        // Note: This would require a voice encoder model, which is a separate component
        // For now, we'll return an error indicating this feature needs the encoder
        Err(PocketTTSError::InferenceFailed(
            "Voice cloning requires the voice encoder model (not yet implemented)".into(),
        ))
    }

    /// Clear reference audio
    pub fn clear_reference_audio(&self) {
        if let Ok(mut model_guard) = self.model.lock() {
            if let Some(ref mut model) = *model_guard {
                model.clear_custom_voice();
            }
        }
    }

    /// Decode raw latents to audio (for reference testing)
    ///
    /// Takes raw f32 latent data and decodes it through Mimi.
    /// The latents_data should be [num_frames * 32] f32 values as bytes.
    pub fn decode_latents(&self, latents_data: Vec<u8>, num_frames: u32) -> Result<SynthesisResult, PocketTTSError> {
        let mut model_guard = self
            .model
            .lock()
            .map_err(|_| PocketTTSError::InferenceFailed("Lock error".into()))?;

        let model = model_guard.as_mut().ok_or(PocketTTSError::ModelNotLoaded)?;

        // Convert bytes to f32 samples
        let expected_floats = (num_frames * 32) as usize;
        let expected_bytes = expected_floats * 4;

        if latents_data.len() != expected_bytes {
            return Err(PocketTTSError::InferenceFailed(format!(
                "Expected {} bytes for {} frames, got {}",
                expected_bytes,
                num_frames,
                latents_data.len()
            )));
        }

        // Convert bytes to f32 vec
        let latents_f32: Vec<f32> = latents_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Decode through Mimi
        let samples = model.decode_latents(&latents_f32, num_frames as usize)?;

        // Convert to WAV bytes
        let wav_data = audio::samples_to_wav(&samples, model.sample_rate())?;

        let duration = audio::duration_seconds(samples.len(), model.sample_rate());

        Ok(SynthesisResult {
            audio_data: wav_data,
            sample_rate: model.sample_rate(),
            channels: 1,
            duration_seconds: duration,
        })
    }

    /// Unload model to free memory
    pub fn unload(&self) {
        if let Ok(mut model_guard) = self.model.lock() {
            *model_guard = None;
        }
    }
}

// Implement Send + Sync for thread safety
unsafe impl Send for PocketTTSEngine {}
unsafe impl Sync for PocketTTSEngine {}
