//! Tests for error types

#[cfg(test)]
mod tests {
    use crate::error::PocketTTSError;
    use std::error::Error;

    #[test]
    fn test_model_not_loaded_error() {
        let error = PocketTTSError::ModelNotLoaded;
        assert_eq!(error.to_string(), "Model not loaded");
    }

    #[test]
    fn test_model_load_failed_error() {
        let error = PocketTTSError::ModelLoadFailed("file not found".to_string());
        assert!(error.to_string().contains("Failed to load model"));
        assert!(error.to_string().contains("file not found"));
    }

    #[test]
    fn test_tokenization_failed_error() {
        let error = PocketTTSError::TokenizationFailed("invalid token".to_string());
        assert!(error.to_string().contains("Tokenization failed"));
        assert!(error.to_string().contains("invalid token"));
    }

    #[test]
    fn test_inference_failed_error() {
        let error = PocketTTSError::InferenceFailed("out of memory".to_string());
        assert!(error.to_string().contains("Inference failed"));
        assert!(error.to_string().contains("out of memory"));
    }

    #[test]
    fn test_invalid_voice_error() {
        let error = PocketTTSError::InvalidVoice(99);
        assert!(error.to_string().contains("Invalid voice index"));
        assert!(error.to_string().contains("99"));
    }

    #[test]
    fn test_invalid_config_error() {
        let error = PocketTTSError::InvalidConfig("temperature out of range".to_string());
        assert!(error.to_string().contains("Invalid configuration"));
        assert!(error.to_string().contains("temperature out of range"));
    }

    #[test]
    fn test_audio_encoding_failed_error() {
        let error = PocketTTSError::AudioEncodingFailed("WAV write failed".to_string());
        assert!(error.to_string().contains("Audio encoding failed"));
        assert!(error.to_string().contains("WAV write failed"));
    }

    #[test]
    fn test_io_error() {
        let error = PocketTTSError::IoError("permission denied".to_string());
        assert!(error.to_string().contains("IO error"));
        assert!(error.to_string().contains("permission denied"));
    }

    #[test]
    fn test_from_std_io_error() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let pocket_error: PocketTTSError = io_error.into();

        match pocket_error {
            PocketTTSError::IoError(msg) => {
                assert!(msg.contains("file not found"));
            }
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PocketTTSError>();
    }

    #[test]
    fn test_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<PocketTTSError>();
    }

    #[test]
    fn test_error_implements_error_trait() {
        let error = PocketTTSError::ModelNotLoaded;
        let _: &dyn Error = &error;
    }

    #[test]
    fn test_error_debug_format() {
        let error = PocketTTSError::InvalidVoice(5);
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("InvalidVoice"));
        assert!(debug_str.contains("5"));
    }

    #[test]
    fn test_error_display_format() {
        let error = PocketTTSError::ModelNotLoaded;
        let display_str = format!("{}", error);
        assert_eq!(display_str, "Model not loaded");
    }
}
