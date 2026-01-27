//! Latency Benchmark for Pocket TTS
//!
//! Measures Time To First Audio (TTFA) and other latency metrics for both
//! synchronous and streaming synthesis modes.
//!
//! Reference baselines (from documentation):
//!   - TTFA: ~200ms
//!   - Per-latent generation: ~50ms
//!   - RTF (Real-Time Factor): 3-4x on iPhone
//!
//! Usage:
//!   cargo run --release --bin latency-bench -- --model-dir /path/to/model
//!   cargo run --release --bin latency-bench -- --model-dir /path/to/model --iterations 10
//!   cargo run --release --bin latency-bench -- --model-dir /path/to/model --streaming

use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::Device;
use pocket_tts_ios::models::pocket_tts::PocketTTSModel;

/// Test phrases of varying lengths
const TEST_PHRASES: &[(&str, &str)] = &[
    ("short", "Hello world."),
    ("medium", "The quick brown fox jumps over the lazy dog."),
    (
        "long",
        "Machine learning has transformed the way we approach complex problems. Neural networks can now recognize images, understand speech, and generate creative content.",
    ),
];

/// Latency results for a single synthesis run
#[derive(Debug, Clone)]
struct LatencyResult {
    phrase_type: String,
    text_length: usize,
    token_count: usize,

    // Timing metrics (all in milliseconds)
    ttfa_ms: f64,           // Time to first audio chunk
    total_synthesis_ms: f64, // Total time for complete synthesis
    audio_duration_ms: f64,  // Duration of generated audio
    rtf: f64,                // Real-time factor (audio_duration / synthesis_time)

    // Streaming-specific (if applicable)
    chunk_count: usize,
    avg_chunk_latency_ms: f64,
    first_chunk_samples: usize,
}

impl LatencyResult {
    fn to_json(&self) -> String {
        format!(
            r#"{{
    "phrase_type": "{}",
    "text_length": {},
    "token_count": {},
    "ttfa_ms": {:.2},
    "total_synthesis_ms": {:.2},
    "audio_duration_ms": {:.2},
    "rtf": {:.2},
    "chunk_count": {},
    "avg_chunk_latency_ms": {:.2},
    "first_chunk_samples": {}
  }}"#,
            self.phrase_type,
            self.text_length,
            self.token_count,
            self.ttfa_ms,
            self.total_synthesis_ms,
            self.audio_duration_ms,
            self.rtf,
            self.chunk_count,
            self.avg_chunk_latency_ms,
            self.first_chunk_samples
        )
    }
}

/// Benchmark configuration
struct BenchConfig {
    model_dir: PathBuf,
    iterations: usize,
    streaming: bool,
    warmup: usize,
    json_output: Option<PathBuf>,
}

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Pocket TTS Latency Benchmark                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let config = parse_args();

    // Load model
    let model = load_model(&config.model_dir);

    // Run benchmarks
    let results = if config.streaming {
        run_streaming_benchmarks(model, &config)
    } else {
        run_sync_benchmarks(model, &config)
    };

    // Print results
    print_results(&results, &config);

    // Save JSON if requested
    if let Some(ref json_path) = config.json_output {
        save_json_results(&results, json_path, &config);
    }
}

fn parse_args() -> BenchConfig {
    let args: Vec<String> = env::args().collect();
    let mut config = BenchConfig {
        model_dir: PathBuf::from("./model"),
        iterations: 5,
        streaming: false,
        warmup: 1,
        json_output: None,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" | "-m" => {
                if i + 1 < args.len() {
                    config.model_dir = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--iterations" | "-n" => {
                if i + 1 < args.len() {
                    config.iterations = args[i + 1].parse().unwrap_or(5);
                    i += 1;
                }
            }
            "--streaming" | "-s" => {
                config.streaming = true;
            }
            "--warmup" | "-w" => {
                if i + 1 < args.len() {
                    config.warmup = args[i + 1].parse().unwrap_or(1);
                    i += 1;
                }
            }
            "--json" | "-j" => {
                if i + 1 < args.len() {
                    config.json_output = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    config
}

fn print_usage() {
    println!("Pocket TTS Latency Benchmark");
    println!("\nUsage:");
    println!("  cargo run --release --bin latency-bench -- [OPTIONS]");
    println!("\nOptions:");
    println!("  -m, --model-dir PATH   Path to model directory (default: ./model)");
    println!("  -n, --iterations N     Number of iterations per phrase (default: 5)");
    println!("  -s, --streaming        Use streaming synthesis mode");
    println!("  -w, --warmup N         Number of warmup iterations (default: 1)");
    println!("  -j, --json PATH        Save results to JSON file");
    println!("  -h, --help             Show this help message");
    println!("\nReference Baselines:");
    println!("  TTFA: ~200ms");
    println!("  RTF: 3-4x realtime");
    println!("  Per-latent: ~50ms");
}

fn load_model(model_dir: &Path) -> PocketTTSModel {
    println!("Loading model from: {}", model_dir.display());

    if !model_dir.exists() {
        eprintln!("ERROR: Model directory not found: {}", model_dir.display());
        std::process::exit(1);
    }

    let start = Instant::now();
    let device = Device::Cpu;
    let model = match PocketTTSModel::load(model_dir, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: Failed to load model: {:?}", e);
            std::process::exit(1);
        }
    };

    let load_time = start.elapsed();
    println!("Model loaded in {:.2}ms\n", load_time.as_secs_f64() * 1000.0);

    model
}

/// Run synchronous synthesis benchmarks
fn run_sync_benchmarks(mut model: PocketTTSModel, config: &BenchConfig) -> Vec<LatencyResult> {
    println!("Mode: Synchronous synthesis");
    println!("Iterations: {} (warmup: {})\n", config.iterations, config.warmup);

    let sample_rate = model.sample_rate();
    let mut all_results = Vec::new();

    for (phrase_type, text) in TEST_PHRASES {
        println!("─── {} phrase: \"{}...\" ───", phrase_type, &text[..text.len().min(30)]);

        let mut results = Vec::new();

        // Warmup runs
        for _ in 0..config.warmup {
            let _ = model.synthesize(text);
        }

        // Timed runs
        for iteration in 0..config.iterations {
            let start = Instant::now();
            let audio = match model.synthesize(text) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("  Iteration {}: ERROR - {:?}", iteration + 1, e);
                    continue;
                }
            };
            let elapsed = start.elapsed();

            let audio_duration_ms = (audio.len() as f64 / sample_rate as f64) * 1000.0;
            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let rtf = audio_duration_ms / total_ms;

            // For sync mode, TTFA equals total synthesis time (no streaming)
            let result = LatencyResult {
                phrase_type: phrase_type.to_string(),
                text_length: text.len(),
                token_count: 0, // Would need tokenizer access
                ttfa_ms: total_ms,
                total_synthesis_ms: total_ms,
                audio_duration_ms,
                rtf,
                chunk_count: 1,
                avg_chunk_latency_ms: total_ms,
                first_chunk_samples: audio.len(),
            };

            println!(
                "  Run {}: {:.1}ms total, {:.1}ms audio, {:.2}x RTF",
                iteration + 1,
                total_ms,
                audio_duration_ms,
                rtf
            );

            results.push(result);
        }

        // Compute averages
        if !results.is_empty() {
            let avg_ttfa = results.iter().map(|r| r.ttfa_ms).sum::<f64>() / results.len() as f64;
            let avg_rtf = results.iter().map(|r| r.rtf).sum::<f64>() / results.len() as f64;
            println!(
                "  Average: {:.1}ms TTFA, {:.2}x RTF\n",
                avg_ttfa, avg_rtf
            );
        }

        all_results.extend(results);
    }

    all_results
}

/// Run streaming synthesis benchmarks with TTFA measurement
fn run_streaming_benchmarks(mut model: PocketTTSModel, config: &BenchConfig) -> Vec<LatencyResult> {
    println!("Mode: Streaming synthesis");
    println!("Iterations: {} (warmup: {})\n", config.iterations, config.warmup);

    let sample_rate = model.sample_rate();
    let mut all_results = Vec::new();

    for (phrase_type, text) in TEST_PHRASES {
        println!("─── {} phrase: \"{}...\" ───", phrase_type, &text[..text.len().min(30)]);

        let mut results = Vec::new();

        // Warmup runs
        for _ in 0..config.warmup {
            let _ = model.synthesize_streaming(text, |_, _| true);
        }

        // Timed runs
        for iteration in 0..config.iterations {
            // Shared state for timing
            let first_chunk_time = Arc::new(AtomicU64::new(0));
            let first_chunk_recorded = Arc::new(AtomicBool::new(false));
            let chunk_count = Arc::new(AtomicU64::new(0));
            let total_samples = Arc::new(AtomicU64::new(0));
            let first_chunk_samples = Arc::new(AtomicU64::new(0));
            let chunk_times = Arc::new(std::sync::Mutex::new(Vec::new()));

            let first_chunk_time_clone = first_chunk_time.clone();
            let first_chunk_recorded_clone = first_chunk_recorded.clone();
            let chunk_count_clone = chunk_count.clone();
            let total_samples_clone = total_samples.clone();
            let first_chunk_samples_clone = first_chunk_samples.clone();
            let chunk_times_clone = chunk_times.clone();

            let start = Instant::now();

            let result = model.synthesize_streaming(text, move |samples, _is_final| {
                let now = start.elapsed().as_nanos() as u64;

                // Record first chunk time (TTFA)
                if !first_chunk_recorded_clone.swap(true, Ordering::SeqCst) {
                    first_chunk_time_clone.store(now, Ordering::SeqCst);
                    first_chunk_samples_clone.store(samples.len() as u64, Ordering::SeqCst);
                }

                // Track chunk timing
                if let Ok(mut times) = chunk_times_clone.lock() {
                    times.push(now);
                }

                chunk_count_clone.fetch_add(1, Ordering::SeqCst);
                total_samples_clone.fetch_add(samples.len() as u64, Ordering::SeqCst);

                true // Continue
            });

            let total_elapsed = start.elapsed();

            if let Err(e) = result {
                eprintln!("  Iteration {}: ERROR - {:?}", iteration + 1, e);
                continue;
            }

            let ttfa_ns = first_chunk_time.load(Ordering::SeqCst);
            let ttfa_ms = ttfa_ns as f64 / 1_000_000.0;
            let total_ms = total_elapsed.as_secs_f64() * 1000.0;
            let samples = total_samples.load(Ordering::SeqCst);
            let audio_duration_ms = (samples as f64 / sample_rate as f64) * 1000.0;
            let rtf = audio_duration_ms / total_ms;
            let chunks = chunk_count.load(Ordering::SeqCst);

            // Calculate average inter-chunk latency
            let avg_chunk_latency = if chunks > 1 {
                let times = chunk_times.lock().unwrap();
                let mut deltas = Vec::new();
                for i in 1..times.len() {
                    deltas.push((times[i] - times[i - 1]) as f64 / 1_000_000.0);
                }
                deltas.iter().sum::<f64>() / deltas.len() as f64
            } else {
                total_ms
            };

            let result = LatencyResult {
                phrase_type: phrase_type.to_string(),
                text_length: text.len(),
                token_count: 0,
                ttfa_ms,
                total_synthesis_ms: total_ms,
                audio_duration_ms,
                rtf,
                chunk_count: chunks as usize,
                avg_chunk_latency_ms: avg_chunk_latency,
                first_chunk_samples: first_chunk_samples.load(Ordering::SeqCst) as usize,
            };

            println!(
                "  Run {}: TTFA={:.1}ms, total={:.1}ms, {:.2}x RTF, {} chunks",
                iteration + 1,
                ttfa_ms,
                total_ms,
                rtf,
                chunks
            );

            results.push(result);
        }

        // Compute averages
        if !results.is_empty() {
            let avg_ttfa = results.iter().map(|r| r.ttfa_ms).sum::<f64>() / results.len() as f64;
            let avg_rtf = results.iter().map(|r| r.rtf).sum::<f64>() / results.len() as f64;
            let avg_chunks = results.iter().map(|r| r.chunk_count).sum::<usize>() / results.len();
            println!(
                "  Average: TTFA={:.1}ms, {:.2}x RTF, {} chunks\n",
                avg_ttfa, avg_rtf, avg_chunks
            );
        }

        all_results.extend(results);
    }

    all_results
}

fn print_results(results: &[LatencyResult], config: &BenchConfig) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK SUMMARY                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Group by phrase type
    for phrase_type in ["short", "medium", "long"] {
        let phrase_results: Vec<_> = results
            .iter()
            .filter(|r| r.phrase_type == phrase_type)
            .collect();

        if phrase_results.is_empty() {
            continue;
        }

        let avg_ttfa = phrase_results.iter().map(|r| r.ttfa_ms).sum::<f64>() / phrase_results.len() as f64;
        let min_ttfa = phrase_results.iter().map(|r| r.ttfa_ms).fold(f64::INFINITY, f64::min);
        let max_ttfa = phrase_results.iter().map(|r| r.ttfa_ms).fold(f64::NEG_INFINITY, f64::max);

        let avg_rtf = phrase_results.iter().map(|r| r.rtf).sum::<f64>() / phrase_results.len() as f64;

        let avg_total = phrase_results.iter().map(|r| r.total_synthesis_ms).sum::<f64>() / phrase_results.len() as f64;

        println!("  {} phrase:", phrase_type.to_uppercase());
        println!("    TTFA:     {:.1}ms (min: {:.1}ms, max: {:.1}ms)", avg_ttfa, min_ttfa, max_ttfa);
        println!("    Total:    {:.1}ms", avg_total);
        println!("    RTF:      {:.2}x", avg_rtf);

        if config.streaming {
            let avg_chunks = phrase_results.iter().map(|r| r.chunk_count).sum::<usize>() / phrase_results.len();
            let avg_chunk_lat = phrase_results.iter().map(|r| r.avg_chunk_latency_ms).sum::<f64>() / phrase_results.len() as f64;
            println!("    Chunks:   {} (avg {:.1}ms each)", avg_chunks, avg_chunk_lat);
        }
        println!();
    }

    // Overall averages
    if !results.is_empty() {
        let overall_ttfa = results.iter().map(|r| r.ttfa_ms).sum::<f64>() / results.len() as f64;
        let overall_rtf = results.iter().map(|r| r.rtf).sum::<f64>() / results.len() as f64;

        println!("─────────────────────────────────────────────────────────────────");
        println!("  OVERALL AVERAGES:");
        println!("    TTFA:     {:.1}ms (target: ~200ms)", overall_ttfa);
        println!("    RTF:      {:.2}x (target: 3-4x)", overall_rtf);
        println!();

        // Pass/fail check against baselines
        let ttfa_ok = overall_ttfa <= 300.0; // Allow some margin
        let rtf_ok = overall_rtf >= 2.5;

        if ttfa_ok && rtf_ok {
            println!("  ✓ PASS: Meets reference baselines");
        } else {
            if !ttfa_ok {
                println!("  ✗ TTFA exceeds target (>300ms)");
            }
            if !rtf_ok {
                println!("  ✗ RTF below target (<2.5x)");
            }
        }
    }
}

fn save_json_results(results: &[LatencyResult], path: &Path, config: &BenchConfig) {
    let results_json: Vec<String> = results.iter().map(|r| r.to_json()).collect();

    let overall_ttfa = results.iter().map(|r| r.ttfa_ms).sum::<f64>() / results.len() as f64;
    let overall_rtf = results.iter().map(|r| r.rtf).sum::<f64>() / results.len() as f64;

    let json = format!(
        r#"{{
  "benchmark": "pocket_tts_latency",
  "mode": "{}",
  "iterations": {},
  "timestamp": "{}",
  "summary": {{
    "avg_ttfa_ms": {:.2},
    "avg_rtf": {:.2},
    "target_ttfa_ms": 200,
    "target_rtf": 3.5
  }},
  "results": [
{}
  ]
}}"#,
        if config.streaming { "streaming" } else { "sync" },
        config.iterations,
        chrono::Utc::now().to_rfc3339(),
        overall_ttfa,
        overall_rtf,
        results_json.join(",\n")
    );

    if let Err(e) = std::fs::write(path, &json) {
        eprintln!("ERROR: Failed to write JSON: {:?}", e);
    } else {
        println!("\nResults saved to: {}", path.display());
    }
}
