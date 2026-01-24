//! Local Mac test harness for Pocket TTS
//!
//! This binary tests the Rust/Candle implementation directly on Mac
//! to verify model correctness before iOS integration.
//!
//! Usage:
//!   cargo run --bin test-tts -- --model-dir /path/to/model --output /path/to/output.wav
//!   cargo run --bin test-tts -- --model-dir /path/to/model --validation-mode
//!
//! The model directory should contain:
//!   - model.safetensors
//!   - tokenizer.model
//!   - voices/ directory with voice embeddings

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use hound::{WavSpec, WavWriter};

// Import from the library (requires rlib crate-type)
use pocket_tts_ios::models::pocket_tts::PocketTTSModel;

/// Standard test phrases matching reference_harness.py
const TEST_PHRASES: &[&str] = &[
    "Hello, this is a test of the Pocket TTS system.",
    "The quick brown fox jumps over the lazy dog.",
    "One two three four five six seven eight nine ten.",
    "How are you doing today?",
];

/// Audio statistics for validation
#[derive(Debug)]
struct AudioStats {
    samples: usize,
    duration_sec: f32,
    max_amplitude: f32,
    mean_amplitude: f32,
    rms: f32,
    dc_offset: f32,
    nan_count: usize,
    inf_count: usize,
    clip_count: usize,
}

impl AudioStats {
    fn compute(audio: &[f32], sample_rate: u32) -> Self {
        let samples = audio.len();
        let duration_sec = samples as f32 / sample_rate as f32;

        let max_amplitude = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let mean_amplitude = audio.iter().map(|s| s.abs()).sum::<f32>() / samples as f32;
        let rms = (audio.iter().map(|s| s * s).sum::<f32>() / samples as f32).sqrt();
        let dc_offset = audio.iter().sum::<f32>() / samples as f32;

        let nan_count = audio.iter().filter(|s| s.is_nan()).count();
        let inf_count = audio.iter().filter(|s| s.is_infinite()).count();
        let clip_count = audio.iter().filter(|s| s.abs() > 0.99).count();

        AudioStats {
            samples,
            duration_sec,
            max_amplitude,
            mean_amplitude,
            rms,
            dc_offset,
            nan_count,
            inf_count,
            clip_count,
        }
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{
        "samples": {},
        "duration_sec": {:.4},
        "max_amplitude": {},
        "mean_amplitude": {},
        "rms": {},
        "dc_offset": {},
        "nan_count": {},
        "inf_count": {},
        "clip_count": {}
      }}"#,
            self.samples,
            self.duration_sec,
            self.max_amplitude,
            self.mean_amplitude,
            self.rms,
            self.dc_offset,
            self.nan_count,
            self.inf_count,
            self.clip_count
        )
    }

    fn is_healthy(&self) -> bool {
        self.nan_count == 0 && self.inf_count == 0 && self.max_amplitude > 0.01 && self.max_amplitude <= 1.0
    }
}

fn main() {
    env_logger::init();

    println!("=== Kyutai Pocket TTS Test Harness ===\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let mut model_dir = PathBuf::from("./model");
    let mut output_path = PathBuf::from("./test_output.wav");
    let mut test_text = String::from("Hello, this is a test of the Pocket TTS system.");
    let mut validation_mode = false;
    let mut validation_output_dir = PathBuf::from("./validation/rust_outputs");
    let mut json_report: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" | "-m" => {
                if i + 1 < args.len() {
                    model_dir = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--text" | "-t" => {
                if i + 1 < args.len() {
                    test_text = args[i + 1].clone();
                    i += 1;
                }
            }
            "--validation-mode" | "-v" => {
                validation_mode = true;
            }
            "--validation-output" => {
                if i + 1 < args.len() {
                    validation_output_dir = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--json-report" => {
                if i + 1 < args.len() {
                    json_report = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--export-latents" => {
                // Latent export not yet implemented - skip the argument
                if i + 1 < args.len() {
                    i += 1;  // Skip the path argument
                }
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                return;
            }
        }
        i += 1;
    }

    // Run in validation mode or single-phrase mode
    if validation_mode {
        run_validation_mode(&model_dir, &validation_output_dir, json_report.as_ref());
    } else {
        run_single_phrase(&model_dir, &output_path, &test_text);
    }
}

/// Run validation mode: synthesize all test phrases and create manifest
fn run_validation_mode(model_dir: &PathBuf, output_dir: &PathBuf, json_report: Option<&PathBuf>) {
    println!("=== VALIDATION MODE ===\n");
    println!("Model directory: {}", model_dir.display());
    println!("Output directory: {}", output_dir.display());
    println!("Test phrases: {}\n", TEST_PHRASES.len());

    // Create output directory
    if let Err(e) = fs::create_dir_all(output_dir) {
        eprintln!("ERROR: Failed to create output directory: {:?}", e);
        std::process::exit(1);
    }

    // Verify and load model
    let model = load_model(model_dir);
    let sample_rate = model.sample_rate();

    // Process all test phrases
    let mut phrase_results: Vec<String> = Vec::new();
    let mut all_healthy = true;
    let mut model = model;

    for (idx, phrase) in TEST_PHRASES.iter().enumerate() {
        let phrase_id = format!("phrase_{:02}", idx);
        println!("\n--- {} ---", phrase_id);
        println!("Text: \"{}\"", phrase);

        // Synthesize
        let start = Instant::now();
        let audio = match model.synthesize(phrase) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("ERROR: Synthesis failed for {}: {:?}", phrase_id, e);
                all_healthy = false;
                continue;
            }
        };
        let synthesis_time = start.elapsed().as_secs_f32();
        println!("Synthesized in {:.2}s", synthesis_time);

        // Compute stats
        let stats = AudioStats::compute(&audio, sample_rate);
        println!("  Samples: {}", stats.samples);
        println!("  Duration: {:.2}s", stats.duration_sec);
        println!("  Max amplitude: {:.4}", stats.max_amplitude);
        println!("  RMS: {:.4}", stats.rms);
        println!("  DC offset: {:.6}", stats.dc_offset);
        println!("  NaN: {}, Inf: {}, Clipped: {}", stats.nan_count, stats.inf_count, stats.clip_count);

        if !stats.is_healthy() {
            println!("  ⚠️  UNHEALTHY signal detected!");
            all_healthy = false;
        }

        // Write WAV file
        let wav_path = output_dir.join(format!("{}_rust.wav", phrase_id));
        if let Err(e) = write_wav(&wav_path, &audio, sample_rate) {
            eprintln!("ERROR: Failed to write WAV: {:?}", e);
            continue;
        }
        println!("  Saved: {}", wav_path.display());

        // Build JSON entry for manifest
        let json_entry = format!(
            r#"    {{
      "id": "{}",
      "text": "{}",
      "wav_file": "{}_rust.wav",
      "audio_stats": {},
      "synthesis_time_sec": {:.4}
    }}"#,
            phrase_id,
            phrase.replace("\"", "\\\""),
            phrase_id,
            stats.to_json(),
            synthesis_time
        );
        phrase_results.push(json_entry);
    }

    // Write manifest.json
    let manifest_json = format!(
        r#"{{
  "model_version": "rust_candle_port",
  "rust_version": "{}",
  "sample_rate": {},
  "phrases": [
{}
  ],
  "all_healthy": {}
}}"#,
        pocket_tts_ios::version(),
        sample_rate,
        phrase_results.join(",\n"),
        all_healthy
    );

    let manifest_path = output_dir.join("manifest.json");
    if let Err(e) = fs::write(&manifest_path, &manifest_json) {
        eprintln!("ERROR: Failed to write manifest: {:?}", e);
    } else {
        println!("\nManifest written: {}", manifest_path.display());
    }

    // Write JSON report if requested
    if let Some(report_path) = json_report {
        if let Err(e) = fs::write(report_path, &manifest_json) {
            eprintln!("ERROR: Failed to write JSON report: {:?}", e);
        } else {
            println!("JSON report: {}", report_path.display());
        }
    }

    // Final summary
    println!("\n=== VALIDATION SUMMARY ===");
    println!("Phrases processed: {}/{}", phrase_results.len(), TEST_PHRASES.len());
    if all_healthy {
        println!("Signal health: ✓ ALL HEALTHY");
        println!("\nRun validation/validate.py to compare against Python reference.");
    } else {
        println!("Signal health: ✗ ISSUES DETECTED");
        println!("\nSome outputs have signal issues (NaN, Inf, silence, or clipping).");
        std::process::exit(1);
    }
}

/// Run single phrase mode (original behavior)
fn run_single_phrase(model_dir: &PathBuf, output_path: &PathBuf, test_text: &str) {
    println!("Configuration:");
    println!("  Model directory: {}", model_dir.display());
    println!("  Output file: {}", output_path.display());
    println!("  Test text: \"{}\"\n", test_text);

    let mut model = load_model(model_dir);
    let sample_rate = model.sample_rate();

    // Run synthesis
    println!("Synthesizing audio...");
    let start = Instant::now();
    let audio = match model.synthesize(test_text) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ERROR: Synthesis failed: {:?}", e);
            std::process::exit(1);
        }
    };
    let synthesis_time = start.elapsed().as_secs_f32();

    println!("Synthesis complete in {:.2}s", synthesis_time);
    println!("  Audio samples: {}", audio.len());
    println!("  Duration: {:.2}s", audio.len() as f32 / sample_rate as f32);
    println!("  Real-time factor: {:.2}x\n",
        (audio.len() as f32 / sample_rate as f32) / synthesis_time);

    // Compute and display stats
    let stats = AudioStats::compute(&audio, sample_rate);
    println!("Audio statistics:");
    println!("  Max amplitude: {:.6}", stats.max_amplitude);
    println!("  Mean amplitude: {:.6}", stats.mean_amplitude);
    println!("  RMS: {:.6}", stats.rms);
    println!("  DC offset: {:.6}", stats.dc_offset);
    println!("  NaN samples: {}", stats.nan_count);
    println!("  Inf samples: {}", stats.inf_count);
    println!("  Clipped samples (>0.99): {}", stats.clip_count);

    // Check for silence
    if stats.max_amplitude < 0.001 {
        println!("\n  WARNING: Audio appears to be near-silent!");
    }
    if stats.dc_offset.abs() > 0.1 {
        println!("  WARNING: Significant DC offset detected!");
    }

    // Sample first/last values
    println!("\n  First 10 samples: {:?}", &audio[..10.min(audio.len())]);
    if audio.len() > 10 {
        println!("  Last 10 samples: {:?}", &audio[audio.len()-10..]);
    }

    // Write WAV file
    println!("\nWriting WAV file to {}...", output_path.display());
    if let Err(e) = write_wav(output_path, &audio, sample_rate) {
        eprintln!("ERROR: Failed to write WAV: {:?}", e);
        std::process::exit(1);
    }

    println!("WAV file written successfully!");
    println!("\nTo play the audio:");
    println!("  afplay {}", output_path.display());

    // Final verdict
    println!("\n=== Test Results ===");
    if !stats.is_healthy() {
        if stats.nan_count > 0 || stats.inf_count > 0 {
            println!("FAIL: Audio contains NaN/Inf values");
        } else if stats.max_amplitude < 0.001 {
            println!("FAIL: Audio is near-silent");
        } else {
            println!("FAIL: Audio has signal issues");
        }
        std::process::exit(1);
    } else {
        println!("PASS: Audio has reasonable amplitude");
        println!("\nListen to the output file to verify quality:");
        println!("  afplay {}", output_path.display());
    }
}

/// Load and verify model
fn load_model(model_dir: &PathBuf) -> PocketTTSModel {
    // Verify model directory exists
    if !model_dir.exists() {
        eprintln!("ERROR: Model directory does not exist: {}", model_dir.display());
        eprintln!("\nThe model directory should contain:");
        eprintln!("  - model.safetensors");
        eprintln!("  - tokenizer.model");
        eprintln!("  - voices/ directory");
        std::process::exit(1);
    }

    // Check required files
    let model_file = model_dir.join("model.safetensors");
    let tokenizer_file = model_dir.join("tokenizer.model");
    let voices_dir = model_dir.join("voices");

    if !model_file.exists() {
        eprintln!("ERROR: model.safetensors not found in {}", model_dir.display());
        std::process::exit(1);
    }
    if !tokenizer_file.exists() {
        eprintln!("ERROR: tokenizer.model not found in {}", model_dir.display());
        std::process::exit(1);
    }
    if !voices_dir.exists() {
        eprintln!("ERROR: voices/ directory not found in {}", model_dir.display());
        std::process::exit(1);
    }

    println!("All model files found.\n");

    // Select device (CPU for now, Metal can be added later)
    let device = Device::Cpu;
    println!("Using device: CPU\n");

    // Load model
    println!("Loading model...");
    let start = Instant::now();
    let model = match PocketTTSModel::load(model_dir, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: Failed to load model: {:?}", e);
            std::process::exit(1);
        }
    };
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!("  Version: {}", model.version());
    println!("  Parameters: {}", model.parameter_count());
    println!("  Sample rate: {} Hz\n", model.sample_rate());

    model
}

/// Write audio to WAV file
fn write_wav(path: &PathBuf, audio: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;

    for sample in audio {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}

fn print_usage() {
    println!("Kyutai Pocket TTS Test Harness");
    println!("\nUsage:");
    println!("  cargo run --bin test-tts -- [OPTIONS]");
    println!("\nModes:");
    println!("  Single phrase (default):");
    println!("    cargo run --bin test-tts -- -m /path/to/model -t \"Hello world\"");
    println!("\n  Validation mode (for comparing against Python reference):");
    println!("    cargo run --bin test-tts -- -m /path/to/model --validation-mode");
    println!("\nOptions:");
    println!("  -m, --model-dir PATH       Path to model directory (default: ./model)");
    println!("  -o, --output PATH          Output WAV file path (default: ./test_output.wav)");
    println!("  -t, --text TEXT            Text to synthesize (default: test phrase)");
    println!("  -v, --validation-mode      Run all test phrases and create manifest");
    println!("  --validation-output PATH   Output dir for validation (default: ./validation/rust_outputs)");
    println!("  --json-report PATH         Write JSON report to file");
    println!("  -h, --help                 Show this help message");
    println!("\nThe model directory should contain:");
    println!("  - model.safetensors");
    println!("  - tokenizer.model");
    println!("  - voices/ directory with voice embeddings");
    println!("\nValidation mode outputs:");
    println!("  - phrase_XX_rust.wav       Audio files for each test phrase");
    println!("  - manifest.json            Statistics and metadata");
}
