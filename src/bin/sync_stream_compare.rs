//! Sync vs Streaming Comparison Test for Pocket TTS
//!
//! Compares synchronous synthesis with streaming synthesis to validate
//! that both modes produce identical audio. This catches stitching bugs
//! in the streaming path.
//!
//! Usage:
//!   cargo run --release --bin sync-stream-compare -- -m ./kyutai-pocket-ios
//!   cargo run --release --bin sync-stream-compare -- -m ./kyutai-pocket-ios --verbose
//!   cargo run --release --bin sync-stream-compare -- -m ./kyutai-pocket-ios --export-wav ./output/

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::Device;
use pocket_tts_ios::audio;
use pocket_tts_ios::audio_analysis::{chunk_statistics, ComparisonSummary};
use pocket_tts_ios::config::TTSConfig;
use pocket_tts_ios::models::pocket_tts::PocketTTSModel;

/// Test phrases matching the iOS app manifest
const TEST_PHRASES: &[(&str, &str)] = &[
    ("short", "Hello, this is a test of the Pocket TTS system."),
    ("medium", "The quick brown fox jumps over the lazy dog."),
    ("numbers", "One two three four five six seven eight nine ten."),
    ("question", "How are you doing today?"),
];

/// Configuration for the comparison test
struct CompareConfig {
    model_dir: PathBuf,
    verbose: bool,
    export_wav: Option<PathBuf>,
    compare_token_chunked: bool,
    compare_true_streaming: bool,
}

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Sync vs Streaming Comparison Test                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let config = parse_args();

    // Validate model directory
    if !config.model_dir.exists() {
        eprintln!("ERROR: Model directory not found: {}", config.model_dir.display());
        std::process::exit(1);
    }

    // Load model
    let model = load_model(&config.model_dir);

    // Create export directory if needed
    if let Some(ref export_dir) = config.export_wav {
        if let Err(e) = fs::create_dir_all(export_dir) {
            eprintln!("ERROR: Could not create export directory: {}", e);
            std::process::exit(1);
        }
    }

    // Run comparisons
    let mut all_passed = true;
    let mut summaries = Vec::new();

    if config.compare_true_streaming {
        println!("\n========== TRUE STREAMING vs SYNC ==========\n");
        let results = run_true_streaming_comparison(model, &config);
        for summary in &results {
            if !summary.passed {
                all_passed = false;
            }
            summary.print_report();
        }
        summaries.extend(results);
    }

    // Print final summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    if all_passed {
        println!("║  OVERALL RESULT: PASS                                        ║");
    } else {
        println!("║  OVERALL RESULT: FAIL                                        ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Summary table
    println!("\nSummary:");
    println!("┌──────────┬────────────┬────────────┬────────────┬────────┐");
    println!("│ Phrase   │ Sync Samp  │ Stream Samp│ Correlation│ Status │");
    println!("├──────────┼────────────┼────────────┼────────────┼────────┤");
    for s in &summaries {
        let status = if s.passed { "PASS" } else { "FAIL" };
        println!(
            "│ {:8} │ {:>10} │ {:>10} │ {:>10.6} │ {:6} │",
            s.phrase_id, s.sync_samples, s.streaming_samples, s.correlation, status
        );
    }
    println!("└──────────┴────────────┴────────────┴────────────┴────────┘");

    if !all_passed {
        std::process::exit(1);
    }
}

fn parse_args() -> CompareConfig {
    let args: Vec<String> = env::args().collect();
    let mut config = CompareConfig {
        model_dir: PathBuf::from("./kyutai-pocket-ios"),
        verbose: false,
        export_wav: None,
        compare_token_chunked: false,
        compare_true_streaming: true, // Default to true streaming
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" | "-m" => {
                if i + 1 < args.len() {
                    config.model_dir = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            },
            "--verbose" | "-v" => {
                config.verbose = true;
            },
            "--export-wav" | "-e" => {
                if i + 1 < args.len() {
                    config.export_wav = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            },
            "--token-chunked" => {
                config.compare_token_chunked = true;
                config.compare_true_streaming = false;
            },
            "--all" => {
                config.compare_token_chunked = true;
                config.compare_true_streaming = true;
            },
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            },
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                std::process::exit(1);
            },
        }
        i += 1;
    }

    config
}

fn print_usage() {
    println!("Sync vs Streaming Comparison Test for Pocket TTS");
    println!("\nUsage:");
    println!("  cargo run --release --bin sync-stream-compare -- [OPTIONS]");
    println!("\nOptions:");
    println!("  -m, --model-dir PATH   Path to model directory (default: ./kyutai-pocket-ios)");
    println!("  -v, --verbose          Show detailed per-chunk analysis");
    println!("  -e, --export-wav PATH  Export WAV files to directory for manual inspection");
    println!("      --token-chunked    Compare token-chunked streaming instead of true streaming");
    println!("      --all              Compare both streaming modes");
    println!("  -h, --help             Show this help message");
    println!("\nThresholds:");
    println!("  Length difference: < 0.5%");
    println!("  Correlation: > 0.999");
    println!("  Discontinuities: 0");
}

fn load_model(model_dir: &Path) -> PocketTTSModel {
    println!("Loading model from: {}", model_dir.display());

    let start = Instant::now();
    let device = Device::Cpu;
    let mut model = match PocketTTSModel::load(model_dir, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: Failed to load model: {:?}", e);
            std::process::exit(1);
        },
    };

    let load_time = start.elapsed();
    println!("Model loaded in {:.2}ms\n", load_time.as_secs_f64() * 1000.0);

    // Configure for deterministic comparison: temperature=0 eliminates randomness
    let config = TTSConfig {
        temperature: 0.0, // Critical: deterministic latent generation
        ..TTSConfig::default()
    };
    if let Err(e) = model.configure(config) {
        eprintln!("Warning: Could not configure model: {:?}", e);
    }

    model
}

fn run_true_streaming_comparison(mut model: PocketTTSModel, config: &CompareConfig) -> Vec<ComparisonSummary> {
    let sample_rate = model.sample_rate();
    let mut summaries = Vec::new();

    for (phrase_id, text) in TEST_PHRASES {
        println!("Testing phrase: {} - \"{}\"", phrase_id, text);

        // 1. Generate with sync mode
        print!("  Sync synthesis... ");
        let sync_start = Instant::now();
        let sync_audio = match model.synthesize(text) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("ERROR: Sync synthesis failed: {:?}", e);
                continue;
            },
        };
        let sync_time = sync_start.elapsed();
        println!("{} samples in {:.1}ms", sync_audio.len(), sync_time.as_secs_f64() * 1000.0);

        // 2. Generate with true streaming mode, collecting chunks
        print!("  Streaming synthesis... ");
        let streaming_chunks: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
        let chunks_clone = streaming_chunks.clone();

        let stream_start = Instant::now();
        let result = model.synthesize_true_streaming(text, move |samples, _is_final| {
            let mut chunks = chunks_clone.lock().unwrap();
            chunks.push(samples.to_vec());
            true
        });

        let stream_time = stream_start.elapsed();

        if let Err(e) = result {
            eprintln!("ERROR: Streaming synthesis failed: {:?}", e);
            continue;
        }

        let chunks = streaming_chunks.lock().unwrap().clone();
        let total_streaming_samples: usize = chunks.iter().map(|c| c.len()).sum();
        println!(
            "{} samples ({} chunks) in {:.1}ms",
            total_streaming_samples,
            chunks.len(),
            stream_time.as_secs_f64() * 1000.0
        );

        // 3. Compute comparison summary
        let summary = ComparisonSummary::from_comparison(phrase_id, &sync_audio, &chunks);

        // 4. Verbose chunk analysis
        if config.verbose {
            println!("\n  Chunk analysis:");
            let stats = chunk_statistics(&chunks);
            for (i, stat) in stats.iter().enumerate() {
                println!(
                    "    Chunk {}: {} samples, max_amp={:.4}, rms={:.4}, first={:.4}, last={:.4}",
                    i, stat.sample_count, stat.max_amplitude, stat.rms, stat.first_sample, stat.last_sample
                );
            }

            // Show boundary deltas
            println!("\n  Boundary analysis:");
            for i in 1..stats.len() {
                let delta = (stats[i].first_sample - stats[i - 1].last_sample).abs();
                let flag = if delta > 0.05 { " <-- ISSUE" } else { "" };
                println!(
                    "    Boundary {}->{}: last={:.4}, first={:.4}, delta={:.4}{}",
                    i - 1,
                    i,
                    stats[i - 1].last_sample,
                    stats[i].first_sample,
                    delta,
                    flag
                );
            }
        }

        // 5. Export WAV files if requested
        if let Some(ref export_dir) = config.export_wav {
            // Export sync audio
            let sync_path = export_dir.join(format!("{}_sync.wav", phrase_id));
            match audio::samples_to_wav(&sync_audio, sample_rate) {
                Ok(sync_wav) => {
                    if let Err(e) = fs::write(&sync_path, &sync_wav) {
                        eprintln!("  Warning: Could not export sync WAV: {}", e);
                    } else {
                        println!("  Exported: {}", sync_path.display());
                    }
                },
                Err(e) => {
                    eprintln!("  Warning: Could not encode sync WAV: {:?}", e);
                },
            }

            // Export streaming audio (concatenated)
            let streaming_audio: Vec<f32> = chunks.iter().flatten().copied().collect();
            let stream_path = export_dir.join(format!("{}_streaming.wav", phrase_id));
            match audio::samples_to_wav(&streaming_audio, sample_rate) {
                Ok(stream_wav) => {
                    if let Err(e) = fs::write(&stream_path, &stream_wav) {
                        eprintln!("  Warning: Could not export streaming WAV: {}", e);
                    } else {
                        println!("  Exported: {}", stream_path.display());
                    }
                },
                Err(e) => {
                    eprintln!("  Warning: Could not encode streaming WAV: {:?}", e);
                },
            }
        }

        summaries.push(summary);
    }

    summaries
}
