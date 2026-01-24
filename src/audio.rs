//! Audio processing utilities for Pocket TTS
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use std::io::Cursor;
use hound::{WavWriter, WavSpec, SampleFormat};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};

use crate::error::PocketTTSError;

/// Audio sample format
pub const SAMPLE_RATE: u32 = 24000;
pub const CHANNELS: u16 = 1;
pub const BITS_PER_SAMPLE: u16 = 32;

/// Convert raw f32 PCM samples to WAV bytes
pub fn samples_to_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, PocketTTSError> {
    let spec = WavSpec {
        channels: CHANNELS,
        sample_rate,
        bits_per_sample: BITS_PER_SAMPLE,
        sample_format: SampleFormat::Float,
    };

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut buffer, spec)
            .map_err(|e| PocketTTSError::AudioEncodingFailed(e.to_string()))?;

        for &sample in samples {
            writer.write_sample(sample)
                .map_err(|e| PocketTTSError::AudioEncodingFailed(e.to_string()))?;
        }

        writer.finalize()
            .map_err(|e| PocketTTSError::AudioEncodingFailed(e.to_string()))?;
    }

    Ok(buffer.into_inner())
}

/// Convert raw f32 PCM samples to raw bytes (for streaming)
pub fn samples_to_bytes(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&s| s.to_le_bytes())
        .collect()
}

/// Convert raw bytes back to f32 samples
pub fn bytes_to_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Resample audio to a different sample rate
pub fn resample(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<f32>, PocketTTSError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let chunk_size = 1024;

    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        chunk_size,
        1, // mono
    ).map_err(|e| PocketTTSError::AudioEncodingFailed(e.to_string()))?;

    let mut output = Vec::new();
    let input_frames: Vec<Vec<f32>> = vec![samples.to_vec()];

    // Process in chunks
    for chunk in input_frames[0].chunks(chunk_size) {
        let input = vec![chunk.to_vec()];
        let resampled = resampler.process(&input, None)
            .map_err(|e| PocketTTSError::AudioEncodingFailed(e.to_string()))?;

        if !resampled.is_empty() {
            output.extend_from_slice(&resampled[0]);
        }
    }

    Ok(output)
}

/// Normalize audio samples to [-1.0, 1.0] range
pub fn normalize(samples: &mut [f32]) {
    let max_abs = samples.iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);

    if max_abs > 0.0 && max_abs != 1.0 {
        let scale = 0.95 / max_abs; // Leave some headroom
        for sample in samples.iter_mut() {
            *sample *= scale;
        }
    }
}

/// Apply speed change by resampling
pub fn apply_speed(samples: &[f32], speed: f32) -> Result<Vec<f32>, PocketTTSError> {
    if (speed - 1.0).abs() < 0.01 {
        return Ok(samples.to_vec());
    }

    // Speed up = lower sample rate equivalent
    let effective_rate = (SAMPLE_RATE as f32 / speed) as u32;
    resample(samples, effective_rate, SAMPLE_RATE)
}

/// Calculate audio duration in seconds
pub fn duration_seconds(sample_count: usize, sample_rate: u32) -> f64 {
    sample_count as f64 / sample_rate as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_to_bytes_roundtrip() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = samples_to_bytes(&samples);
        let recovered = bytes_to_samples(&bytes);
        assert_eq!(samples, recovered);
    }

    #[test]
    fn test_normalize() {
        let mut samples = vec![0.0f32, 0.25, -0.25, 0.5, -0.5];
        normalize(&mut samples);
        let max = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!((max - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_duration() {
        let duration = duration_seconds(24000, 24000);
        assert!((duration - 1.0).abs() < 0.001);
    }
}
