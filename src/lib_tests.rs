//! Tests for library-level exports and functions

#[cfg(test)]
mod tests {
    use crate::{version, available_voices, AudioChunk, SynthesisResult};

    #[test]
    fn test_version_format() {
        let v = version();
        // Should be in semver format: X.Y.Z
        let parts: Vec<&str> = v.split('.').collect();
        assert_eq!(parts.len(), 3, "Version should have 3 parts: {}", v);

        for part in parts {
            assert!(part.parse::<u32>().is_ok(), "Version part should be numeric: {}", part);
        }
    }

    #[test]
    fn test_version_matches_cargo() {
        let v = version();
        assert_eq!(v, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_available_voices_count() {
        let voices = available_voices();
        assert_eq!(voices.len(), 8, "Should have 8 built-in voices");
    }

    #[test]
    fn test_available_voices_indices() {
        let voices = available_voices();
        for (i, voice) in voices.iter().enumerate() {
            assert_eq!(voice.index as usize, i, "Voice index should match position");
        }
    }

    #[test]
    fn test_available_voices_names() {
        let voices = available_voices();
        let expected_names = ["Alba", "Marius", "Javert", "Jean", "Fantine", "Cosette", "Eponine", "Azelma"];

        for (voice, expected) in voices.iter().zip(expected_names.iter()) {
            assert_eq!(voice.name, *expected);
        }
    }

    #[test]
    fn test_available_voices_genders() {
        let voices = available_voices();

        // Alba, Fantine, Cosette, Eponine, Azelma are female
        // Marius, Javert, Jean are male
        let female_indices = [0, 4, 5, 6, 7];
        let male_indices = [1, 2, 3];

        for voice in &voices {
            if female_indices.contains(&(voice.index as usize)) {
                assert_eq!(voice.gender, "female", "Voice {} should be female", voice.name);
            } else if male_indices.contains(&(voice.index as usize)) {
                assert_eq!(voice.gender, "male", "Voice {} should be male", voice.name);
            }
        }
    }

    #[test]
    fn test_available_voices_have_descriptions() {
        let voices = available_voices();

        for voice in &voices {
            assert!(!voice.description.is_empty(), "Voice {} should have a description", voice.name);
        }
    }

    #[test]
    fn test_audio_chunk_creation() {
        let chunk = AudioChunk {
            audio_data: vec![0, 1, 2, 3],
            sample_rate: 24000,
            is_final: false,
        };

        assert_eq!(chunk.audio_data.len(), 4);
        assert_eq!(chunk.sample_rate, 24000);
        assert!(!chunk.is_final);
    }

    #[test]
    fn test_audio_chunk_final() {
        let chunk = AudioChunk {
            audio_data: vec![],
            sample_rate: 24000,
            is_final: true,
        };

        assert!(chunk.is_final);
    }

    #[test]
    fn test_synthesis_result_creation() {
        let result = SynthesisResult {
            audio_data: vec![0; 1000],
            sample_rate: 24000,
            channels: 1,
            duration_seconds: 0.5,
        };

        assert_eq!(result.audio_data.len(), 1000);
        assert_eq!(result.sample_rate, 24000);
        assert_eq!(result.channels, 1);
        assert!((result.duration_seconds - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_audio_chunk_clone() {
        let chunk = AudioChunk {
            audio_data: vec![1, 2, 3],
            sample_rate: 24000,
            is_final: true,
        };

        let cloned = chunk.clone();
        assert_eq!(chunk.audio_data, cloned.audio_data);
        assert_eq!(chunk.sample_rate, cloned.sample_rate);
        assert_eq!(chunk.is_final, cloned.is_final);
    }

    #[test]
    fn test_synthesis_result_clone() {
        let result = SynthesisResult {
            audio_data: vec![4, 5, 6],
            sample_rate: 48000,
            channels: 2,
            duration_seconds: 1.5,
        };

        let cloned = result.clone();
        assert_eq!(result.audio_data, cloned.audio_data);
        assert_eq!(result.sample_rate, cloned.sample_rate);
        assert_eq!(result.channels, cloned.channels);
        assert_eq!(result.duration_seconds, cloned.duration_seconds);
    }

    #[test]
    fn test_audio_chunk_debug() {
        let chunk = AudioChunk {
            audio_data: vec![0],
            sample_rate: 24000,
            is_final: false,
        };

        let debug = format!("{:?}", chunk);
        assert!(debug.contains("AudioChunk"));
        assert!(debug.contains("24000"));
    }

    #[test]
    fn test_synthesis_result_debug() {
        let result = SynthesisResult {
            audio_data: vec![0],
            sample_rate: 24000,
            channels: 1,
            duration_seconds: 0.1,
        };

        let debug = format!("{:?}", result);
        assert!(debug.contains("SynthesisResult"));
    }
}
