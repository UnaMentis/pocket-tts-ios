//! Tests for configuration types

#[cfg(test)]
mod tests {
    use crate::config::{TTSConfig, PocketVoiceInfo, ModelManifest};

    #[test]
    fn test_default_config() {
        let config = TTSConfig::default();
        assert_eq!(config.voice_index, 0);
        assert!((config.temperature - 0.7).abs() < 0.001);
        assert!((config.top_p - 0.9).abs() < 0.001);
        assert!((config.speed - 1.0).abs() < 0.001);
        assert_eq!(config.consistency_steps, 2);
        assert!(!config.use_fixed_seed);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_low_latency_config() {
        let config = TTSConfig::low_latency();
        assert_eq!(config.consistency_steps, 1);
    }

    #[test]
    fn test_high_quality_config() {
        let config = TTSConfig::high_quality();
        assert_eq!(config.consistency_steps, 4);
        assert!((config.temperature - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_config_validation_valid() {
        let config = TTSConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_voice() {
        let config = TTSConfig {
            voice_index: 8, // Invalid: must be 0-7
            ..Default::default()
        };
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("Voice index"));
    }

    #[test]
    fn test_config_validation_invalid_temperature_high() {
        let config = TTSConfig {
            temperature: 1.5, // Invalid: must be 0.0-1.0
            ..Default::default()
        };
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("Temperature"));
    }

    #[test]
    fn test_config_validation_invalid_temperature_low() {
        let config = TTSConfig {
            temperature: -0.1, // Invalid: must be 0.0-1.0
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_top_p_high() {
        let config = TTSConfig {
            top_p: 1.5, // Invalid: must be 0.1-1.0
            ..Default::default()
        };
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("Top-P"));
    }

    #[test]
    fn test_config_validation_invalid_top_p_low() {
        let config = TTSConfig {
            top_p: 0.05, // Invalid: must be 0.1-1.0
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_speed_high() {
        let config = TTSConfig {
            speed: 3.0, // Invalid: must be 0.5-2.0
            ..Default::default()
        };
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("Speed"));
    }

    #[test]
    fn test_config_validation_invalid_speed_low() {
        let config = TTSConfig {
            speed: 0.3, // Invalid: must be 0.5-2.0
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_consistency_steps_high() {
        let config = TTSConfig {
            consistency_steps: 5, // Invalid: must be 1-4
            ..Default::default()
        };
        assert!(config.validate().is_err());
        assert!(config.validate().unwrap_err().contains("Consistency"));
    }

    #[test]
    fn test_config_validation_invalid_consistency_steps_zero() {
        let config = TTSConfig {
            consistency_steps: 0, // Invalid: must be 1-4
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_boundary_values() {
        // Test minimum valid values
        let min_config = TTSConfig {
            voice_index: 0,
            temperature: 0.0,
            top_p: 0.1,
            speed: 0.5,
            consistency_steps: 1,
            use_fixed_seed: false,
            seed: 0,
        };
        assert!(min_config.validate().is_ok());

        // Test maximum valid values
        let max_config = TTSConfig {
            voice_index: 7,
            temperature: 1.0,
            top_p: 1.0,
            speed: 2.0,
            consistency_steps: 4,
            use_fixed_seed: true,
            seed: u32::MAX,
        };
        assert!(max_config.validate().is_ok());
    }

    #[test]
    fn test_voice_info_creation() {
        let voice = PocketVoiceInfo {
            index: 0,
            name: "Alba".to_string(),
            gender: "female".to_string(),
            description: "Clear, neutral female voice".to_string(),
        };
        assert_eq!(voice.index, 0);
        assert_eq!(voice.name, "Alba");
        assert_eq!(voice.gender, "female");
    }

    #[test]
    fn test_model_manifest_default() {
        let manifest = ModelManifest::default();
        assert_eq!(manifest.version, "1.0.2");
        assert_eq!(manifest.model_id, "kyutai/pocket-tts");
        assert_eq!(manifest.license, "CC-BY-4.0");
        assert_eq!(manifest.parameters, 117_856_642);
        assert_eq!(manifest.sample_rate, 24000);
        assert!((manifest.frame_rate - 12.5).abs() < 0.001);
        assert_eq!(manifest.hidden_size, 1024);
        assert_eq!(manifest.num_layers, 6);
        assert_eq!(manifest.num_heads, 16);
        assert_eq!(manifest.vocab_size, 32000);
    }
}
