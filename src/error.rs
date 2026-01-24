//! Error types for Pocket TTS

use thiserror::Error;

/// Errors that can occur during TTS operations
#[derive(Debug, Error)]
pub enum PocketTTSError {
    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Failed to load model: {0}")]
    ModelLoadFailed(String),

    #[error("Tokenization failed: {0}")]
    TokenizationFailed(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid voice index: {0}")]
    InvalidVoice(u32),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Audio encoding failed: {0}")]
    AudioEncodingFailed(String),

    #[error("IO error: {0}")]
    IoError(String),
}

impl From<std::io::Error> for PocketTTSError {
    fn from(err: std::io::Error) -> Self {
        PocketTTSError::IoError(err.to_string())
    }
}

impl From<candle_core::Error> for PocketTTSError {
    fn from(err: candle_core::Error) -> Self {
        PocketTTSError::InferenceFailed(err.to_string())
    }
}

impl From<safetensors::SafeTensorError> for PocketTTSError {
    fn from(err: safetensors::SafeTensorError) -> Self {
        PocketTTSError::ModelLoadFailed(err.to_string())
    }
}

// UniFFI requires this specific error enum format
impl From<PocketTTSError> for uniffi::UnexpectedUniFFICallbackError {
    fn from(err: PocketTTSError) -> Self {
        uniffi::UnexpectedUniFFICallbackError::new(err.to_string())
    }
}
