//! Audio analysis utilities for comparing sync vs streaming synthesis output.
//!
//! This module provides functions to compare audio waveforms and detect
//! discrepancies between different synthesis modes.

/// Information about a detected discontinuity at a chunk boundary
#[derive(Debug, Clone)]
pub struct DiscontinuityInfo {
    /// Index of the chunk where discontinuity was detected (at its start)
    pub chunk_index: usize,
    /// Sample index in the concatenated audio
    pub sample_index: usize,
    /// Amplitude delta at the boundary
    pub delta: f32,
    /// Severity level based on threshold
    pub severity: DiscontinuitySeverity,
}

/// Severity classification for discontinuities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscontinuitySeverity {
    /// Minor discontinuity (delta 0.05-0.1)
    Minor,
    /// Moderate discontinuity (delta 0.1-0.2)
    Moderate,
    /// Severe discontinuity (delta > 0.2)
    Severe,
}

/// Statistics for a single audio chunk
#[derive(Debug, Clone)]
pub struct ChunkStats {
    /// Number of samples in the chunk
    pub sample_count: usize,
    /// Maximum amplitude in the chunk
    pub max_amplitude: f32,
    /// RMS (root mean square) level
    pub rms: f32,
    /// First sample value (for boundary analysis)
    pub first_sample: f32,
    /// Last sample value (for boundary analysis)
    pub last_sample: f32,
}

/// Compute Pearson correlation coefficient between two audio signals.
///
/// Returns a value between -1.0 and 1.0, where:
/// - 1.0 means perfectly identical
/// - 0.0 means no correlation
/// - -1.0 means perfectly inverted
///
/// Uses the shorter of the two signals for comparison length.
pub fn pearson_correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let n_f64 = n as f64;

    // Compute sums
    let sum_a: f64 = a[..n].iter().map(|&x| x as f64).sum();
    let sum_b: f64 = b[..n].iter().map(|&x| x as f64).sum();
    let sum_ab: f64 = a[..n].iter().zip(b[..n].iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let sum_a2: f64 = a[..n].iter().map(|&x| (x as f64).powi(2)).sum();
    let sum_b2: f64 = b[..n].iter().map(|&x| (x as f64).powi(2)).sum();

    // Pearson formula
    let numerator = n_f64 * sum_ab - sum_a * sum_b;
    let denominator = ((n_f64 * sum_a2 - sum_a.powi(2)) * (n_f64 * sum_b2 - sum_b.powi(2))).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute RMS (root mean square) difference between two signals.
///
/// Lower values indicate more similar signals.
pub fn rms_difference(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let sum_sq_diff: f64 = a[..n].iter().zip(b[..n].iter()).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();

    (sum_sq_diff / n as f64).sqrt()
}

/// Compute RMS level of a signal.
pub fn rms_level(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_sq: f64 = samples.iter().map(|&x| (x as f64).powi(2)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Detect discontinuities at chunk boundaries.
///
/// A discontinuity is detected when the delta between the last sample of
/// one chunk and the first sample of the next exceeds the threshold.
///
/// # Arguments
/// * `chunks` - Vector of audio chunks
/// * `threshold` - Amplitude delta threshold (e.g., 0.05 for 5%)
///
/// # Returns
/// Vector of discontinuity information for each detected issue
pub fn detect_discontinuities(chunks: &[Vec<f32>], threshold: f32) -> Vec<DiscontinuityInfo> {
    let mut issues = Vec::new();
    let mut cumulative_samples = 0usize;

    for i in 0..chunks.len() {
        if i > 0 {
            let prev_last = chunks[i - 1].last().copied().unwrap_or(0.0);
            let curr_first = chunks[i].first().copied().unwrap_or(0.0);
            let delta = (curr_first - prev_last).abs();

            if delta > threshold {
                let severity = if delta > 0.2 {
                    DiscontinuitySeverity::Severe
                } else if delta > 0.1 {
                    DiscontinuitySeverity::Moderate
                } else {
                    DiscontinuitySeverity::Minor
                };

                issues.push(DiscontinuityInfo {
                    chunk_index: i,
                    sample_index: cumulative_samples,
                    delta,
                    severity,
                });
            }
        }

        cumulative_samples += chunks[i].len();
    }

    issues
}

/// Compute statistics for each chunk.
pub fn chunk_statistics(chunks: &[Vec<f32>]) -> Vec<ChunkStats> {
    chunks
        .iter()
        .map(|chunk| {
            let sample_count = chunk.len();
            let max_amplitude = chunk.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            let rms = rms_level(chunk) as f32;
            let first_sample = chunk.first().copied().unwrap_or(0.0);
            let last_sample = chunk.last().copied().unwrap_or(0.0);

            ChunkStats {
                sample_count,
                max_amplitude,
                rms,
                first_sample,
                last_sample,
            }
        })
        .collect()
}

/// Align two signals by padding the shorter one with zeros.
///
/// Returns two vectors of equal length.
pub fn align_signals(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let max_len = a.len().max(b.len());

    let mut a_aligned = a.to_vec();
    let mut b_aligned = b.to_vec();

    a_aligned.resize(max_len, 0.0);
    b_aligned.resize(max_len, 0.0);

    (a_aligned, b_aligned)
}

/// Find the maximum absolute difference between two signals.
pub fn max_abs_difference(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute mean absolute error between two signals.
pub fn mean_abs_error(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let sum: f64 = a[..n].iter().zip(b[..n].iter()).map(|(&x, &y)| (x - y).abs() as f64).sum();

    sum / n as f64
}

/// Summary of comparison between sync and streaming outputs
#[derive(Debug, Clone)]
pub struct ComparisonSummary {
    /// Phrase identifier
    pub phrase_id: String,
    /// Sample count from sync mode
    pub sync_samples: usize,
    /// Sample count from streaming mode
    pub streaming_samples: usize,
    /// Length difference as percentage
    pub length_diff_percent: f64,
    /// Pearson correlation coefficient
    pub correlation: f64,
    /// RMS difference
    pub rms_diff: f64,
    /// Mean absolute error
    pub mae: f64,
    /// Maximum absolute difference
    pub max_diff: f32,
    /// Number of chunks in streaming output
    pub chunk_count: usize,
    /// Detected discontinuities
    pub discontinuities: Vec<DiscontinuityInfo>,
    /// Whether comparison passes all thresholds
    pub passed: bool,
    /// Issues found (empty if passed)
    pub issues: Vec<String>,
}

impl ComparisonSummary {
    /// Create a new comparison summary from sync and streaming outputs.
    ///
    /// Thresholds:
    /// - Length difference: < 0.5%
    /// - Correlation: > 0.999
    /// - Discontinuities: 0
    pub fn from_comparison(phrase_id: &str, sync_audio: &[f32], streaming_chunks: &[Vec<f32>]) -> Self {
        // Concatenate streaming chunks
        let streaming_audio: Vec<f32> = streaming_chunks.iter().flatten().copied().collect();

        // Compute metrics
        let sync_samples = sync_audio.len();
        let streaming_samples = streaming_audio.len();
        let length_diff_percent = if sync_samples > 0 {
            ((streaming_samples as f64 - sync_samples as f64) / sync_samples as f64).abs() * 100.0
        } else {
            100.0
        };

        let correlation = pearson_correlation(sync_audio, &streaming_audio);
        let rms_diff = rms_difference(sync_audio, &streaming_audio);
        let mae = mean_abs_error(sync_audio, &streaming_audio);
        let max_diff = max_abs_difference(sync_audio, &streaming_audio);
        let chunk_count = streaming_chunks.len();
        let discontinuities = detect_discontinuities(streaming_chunks, 0.05);

        // Check thresholds
        let mut issues = Vec::new();
        let mut passed = true;

        if length_diff_percent > 0.5 {
            issues.push(format!("Length difference {:.2}% exceeds 0.5% threshold", length_diff_percent));
            passed = false;
        }

        if correlation < 0.999 {
            issues.push(format!("Correlation {:.6} below 0.999 threshold", correlation));
            passed = false;
        }

        if !discontinuities.is_empty() {
            issues.push(format!(
                "{} discontinuities detected at chunk boundaries",
                discontinuities.len()
            ));
            passed = false;
        }

        Self {
            phrase_id: phrase_id.to_string(),
            sync_samples,
            streaming_samples,
            length_diff_percent,
            correlation,
            rms_diff,
            mae,
            max_diff,
            chunk_count,
            discontinuities,
            passed,
            issues,
        }
    }

    /// Print a human-readable report
    pub fn print_report(&self) {
        println!("\n=== {} ===", self.phrase_id);
        println!("Sync samples:      {}", self.sync_samples);
        println!("Streaming samples: {}", self.streaming_samples);
        println!(
            "Length diff:       {:.3}% {}",
            self.length_diff_percent,
            if self.length_diff_percent <= 0.5 {
                "[PASS]"
            } else {
                "[FAIL]"
            }
        );
        println!(
            "Correlation:       {:.6} {}",
            self.correlation,
            if self.correlation >= 0.999 { "[PASS]" } else { "[FAIL]" }
        );
        println!("RMS diff:          {:.6}", self.rms_diff);
        println!("Max diff:          {:.6}", self.max_diff);
        println!("Chunk count:       {}", self.chunk_count);
        println!(
            "Discontinuities:   {} {}",
            self.discontinuities.len(),
            if self.discontinuities.is_empty() {
                "[PASS]"
            } else {
                "[FAIL]"
            }
        );

        if !self.discontinuities.is_empty() {
            for d in &self.discontinuities {
                println!(
                    "  - Chunk {}: delta={:.4} ({:?}) at sample {}",
                    d.chunk_index, d.delta, d.severity, d.sample_index
                );
            }
        }

        println!("\nResult: {}", if self.passed { "PASS" } else { "FAIL" });
        if !self.issues.is_empty() {
            for issue in &self.issues {
                println!("  - {}", issue);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation_identical() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let corr = pearson_correlation(&a, &b);
        assert!((corr - 1.0).abs() < 1e-10, "Expected 1.0, got {}", corr);
    }

    #[test]
    fn test_pearson_correlation_inverted() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let b = vec![-0.1, -0.2, -0.3, -0.4, -0.5];
        let corr = pearson_correlation(&a, &b);
        assert!((corr + 1.0).abs() < 1e-10, "Expected -1.0, got {}", corr);
    }

    #[test]
    fn test_rms_difference_identical() {
        let a = vec![0.1, 0.2, 0.3];
        let b = vec![0.1, 0.2, 0.3];
        let diff = rms_difference(&a, &b);
        assert!(diff < 1e-10, "Expected 0, got {}", diff);
    }

    #[test]
    fn test_discontinuity_detection() {
        let chunks = vec![
            vec![0.1, 0.2, 0.3],   // ends at 0.3
            vec![0.8, 0.7, 0.6],   // starts at 0.8 - delta = 0.5
            vec![0.55, 0.5, 0.45], // starts at 0.55 - delta = 0.05
        ];
        let issues = detect_discontinuities(&chunks, 0.1);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].chunk_index, 1);
        assert!((issues[0].delta - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_chunk_statistics() {
        let chunks = vec![vec![0.1, 0.5, 0.2], vec![-0.3, 0.0, 0.3]];
        let stats = chunk_statistics(&chunks);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].sample_count, 3);
        assert!((stats[0].max_amplitude - 0.5).abs() < 0.01);
        assert!((stats[0].first_sample - 0.1).abs() < 0.01);
        assert!((stats[0].last_sample - 0.2).abs() < 0.01);
    }
}
