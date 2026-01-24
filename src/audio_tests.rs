//! Tests for audio processing utilities

#[cfg(test)]
mod tests {
    use crate::audio::{
        samples_to_bytes, bytes_to_samples, normalize, duration_seconds,
        samples_to_wav, SAMPLE_RATE, CHANNELS,
    };

    #[test]
    fn test_samples_to_bytes_empty() {
        let samples: Vec<f32> = vec![];
        let bytes = samples_to_bytes(&samples);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_samples_to_bytes_single() {
        let samples = vec![1.0f32];
        let bytes = samples_to_bytes(&samples);
        assert_eq!(bytes.len(), 4); // f32 = 4 bytes
    }

    #[test]
    fn test_samples_to_bytes_roundtrip() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0, 0.123456];
        let bytes = samples_to_bytes(&samples);
        let recovered = bytes_to_samples(&bytes);

        assert_eq!(samples.len(), recovered.len());
        for (original, recovered) in samples.iter().zip(recovered.iter()) {
            assert!((original - recovered).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bytes_to_samples_empty() {
        let bytes: Vec<u8> = vec![];
        let samples = bytes_to_samples(&bytes);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_bytes_to_samples_partial() {
        // Only 3 bytes - should produce 0 samples (needs 4 bytes per f32)
        let bytes = vec![0u8, 0, 0];
        let samples = bytes_to_samples(&bytes);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_normalize_already_normalized() {
        let mut samples = vec![0.0f32, 0.5, -0.5, 0.95, -0.95];
        normalize(&mut samples);

        let max = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!((max - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_normalize_loud_signal() {
        let mut samples = vec![0.0f32, 2.0, -2.0, 1.0, -1.0];
        normalize(&mut samples);

        let max = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!((max - 0.95).abs() < 0.01); // Should be normalized to 0.95 headroom
    }

    #[test]
    fn test_normalize_quiet_signal() {
        let mut samples = vec![0.0f32, 0.1, -0.1, 0.05, -0.05];
        normalize(&mut samples);

        let max = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!((max - 0.95).abs() < 0.01); // Should be amplified to 0.95
    }

    #[test]
    fn test_normalize_silent() {
        let mut samples = vec![0.0f32, 0.0, 0.0];
        normalize(&mut samples);

        // Should remain silent
        for s in &samples {
            assert!(*s == 0.0);
        }
    }

    #[test]
    fn test_normalize_single_sample() {
        let mut samples = vec![0.5f32];
        normalize(&mut samples);
        assert!((samples[0] - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_duration_seconds_one_second() {
        let duration = duration_seconds(24000, 24000);
        assert!((duration - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_duration_seconds_half_second() {
        let duration = duration_seconds(12000, 24000);
        assert!((duration - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_duration_seconds_two_seconds() {
        let duration = duration_seconds(48000, 24000);
        assert!((duration - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_duration_seconds_zero() {
        let duration = duration_seconds(0, 24000);
        assert!(duration == 0.0);
    }

    #[test]
    fn test_sample_rate_constant() {
        assert_eq!(SAMPLE_RATE, 24000);
    }

    #[test]
    fn test_channels_constant() {
        assert_eq!(CHANNELS, 1); // Mono
    }

    #[test]
    fn test_samples_to_wav_creates_valid_header() {
        let samples = vec![0.0f32; 100];
        let wav = samples_to_wav(&samples, 24000).unwrap();

        // Check RIFF header
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");

        // Check fmt chunk
        assert_eq!(&wav[12..16], b"fmt ");

        // The data chunk location varies based on format (hound adds a "fact" chunk for float)
        // Just verify data is present somewhere in the file
        assert!(wav.windows(4).any(|w| w == b"data"), "WAV should contain data chunk");
    }

    #[test]
    fn test_samples_to_wav_correct_size() {
        let samples = vec![0.0f32; 100];
        let wav = samples_to_wav(&samples, 24000).unwrap();

        // hound adds extra chunks for float format (fact chunk = 12 bytes, padding = 12 bytes)
        // Header is 44 bytes base + 24 bytes extra = 68 bytes, then data (100 samples * 4 bytes)
        // Just verify it contains the data
        let data_size = 100 * 4; // 400 bytes of sample data
        assert!(wav.len() >= data_size, "WAV should contain at least the sample data");
        assert!(wav.len() < data_size + 200, "WAV header shouldn't be excessive");
    }

    #[test]
    fn test_samples_to_wav_empty() {
        let samples: Vec<f32> = vec![];
        let wav = samples_to_wav(&samples, 24000).unwrap();

        // Should still have valid header (hound adds fact chunk for float format)
        assert!(wav.len() >= 44, "WAV should have at least minimal header");
        assert!(wav.len() < 200, "Empty WAV shouldn't be too large");
    }

    #[test]
    fn test_preserve_sample_values() {
        let original = vec![0.1f32, -0.2, 0.3, -0.4, 0.5];
        let bytes = samples_to_bytes(&original);
        let recovered = bytes_to_samples(&bytes);

        for (o, r) in original.iter().zip(recovered.iter()) {
            assert!((o - r).abs() < 1e-7, "Sample mismatch: {} vs {}", o, r);
        }
    }

    #[test]
    fn test_special_float_values() {
        let samples = vec![0.0f32, -0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
        let bytes = samples_to_bytes(&samples);
        let recovered = bytes_to_samples(&bytes);

        assert_eq!(samples.len(), recovered.len());
        // 0.0 and -0.0 should both recover as 0.0
        assert!(recovered[0] == 0.0);
        assert!(recovered[1] == 0.0 || recovered[1] == -0.0);
    }
}
