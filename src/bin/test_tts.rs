// Allow excessive precision for test reference values copied from Python
#![allow(clippy::excessive_precision)]

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
use std::fs::{self, File};
use std::io::{Read as IoRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hound::{WavSpec, WavWriter};

// Import from the library (requires rlib crate-type)
use pocket_tts_ios::config::TTSConfig;
use pocket_tts_ios::models::mimi::{MimiConfig, MimiDecoder};
use pocket_tts_ios::models::pocket_tts::PocketTTSModel;

/// Write latents to .npy format for Python compatibility
fn write_npy(path: &Path, data: &[f32], shape: &[usize; 3]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;

    // NPY format header
    // Magic number
    file.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y'])?;
    // Version 1.0
    file.write_all(&[0x01, 0x00])?;

    // Build header dict
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        shape[0], shape[1], shape[2]
    );

    // Pad header to make total header + data aligned (header len is 2 bytes in v1.0)
    // Total before data: 8 (magic+version) + 2 (header len) + header + padding + newline
    let prefix_len = 10; // magic(6) + version(2) + header_len(2)
    let total_header_len = prefix_len + header.len() + 1; // +1 for newline
    let padding_needed = (64 - (total_header_len % 64)) % 64;
    let padded_header_len = header.len() + padding_needed + 1; // header + padding + newline

    // Write header length (little-endian u16)
    file.write_all(&(padded_header_len as u16).to_le_bytes())?;

    // Write header
    file.write_all(header.as_bytes())?;

    // Write padding (spaces)
    for _ in 0..padding_needed {
        file.write_all(b" ")?;
    }

    // Newline before data
    file.write_all(b"\n")?;

    // Write data as little-endian f32
    for &val in data {
        file.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Standard test phrases matching reference_harness.py
const TEST_PHRASES: &[&str] = &[
    "Hello, this is a test of the Pocket TTS system.",
    "The quick brown fox jumps over the lazy dog.",
    "One two three four five six seven eight nine ten.",
    "How are you doing today?",
];

/// Extended test phrases for longer content validation
/// These test paragraph-level generation for educational content
const LONG_TEST_PHRASES: &[&str] = &[
    // Medium length (~50 tokens, ~8-10 seconds)
    "The pharmaceutical company Pfizer and actor Arnold Schwarzenegger discussed mRNA vaccines at the café while listening to Tchaikovsky. Scientists believe this breakthrough will revolutionize medicine.",
    // Paragraph length (~100 tokens, ~15-20 seconds)
    "Machine learning has transformed the way we approach complex problems. Neural networks, inspired by the human brain, can now recognize images, understand speech, and even generate creative content. The technology continues to evolve rapidly, with new architectures emerging every year. Researchers are particularly excited about transformer models, which have shown remarkable capabilities in natural language processing tasks.",
    // Multi-sentence educational content (~80 tokens)
    "The water cycle is a fundamental process in Earth's climate system. Water evaporates from oceans and lakes, rises into the atmosphere, forms clouds, and eventually falls as precipitation. This cycle has been operating for billions of years and is essential for all life on our planet.",
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
    let mut extended_validation = false;
    let mut validation_output_dir = PathBuf::from("./validation/rust_outputs");
    let mut json_report: Option<PathBuf> = None;
    let mut export_latents_path: Option<PathBuf> = None;
    let mut load_latents_path: Option<PathBuf> = None;
    let mut test_convtr = false;
    let mut test_mimi_trace = false;
    let mut temperature: Option<f32> = None;
    let mut top_p: Option<f32> = None;
    let mut consistency_steps: Option<u32> = None;
    let mut speed: Option<f32> = None;
    let mut seed: Option<u32> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" | "-m" => {
                if i + 1 < args.len() {
                    model_dir = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            },
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            },
            "--text" | "-t" => {
                if i + 1 < args.len() {
                    test_text = args[i + 1].clone();
                    i += 1;
                }
            },
            "--validation-mode" | "-v" => {
                validation_mode = true;
            },
            "--extended" | "-e" => {
                extended_validation = true;
            },
            "--validation-output" => {
                if i + 1 < args.len() {
                    validation_output_dir = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            },
            "--json-report" => {
                if i + 1 < args.len() {
                    json_report = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            },
            "--export-latents" => {
                if i + 1 < args.len() {
                    export_latents_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            },
            "--load-latents" => {
                if i + 1 < args.len() {
                    load_latents_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            },
            "--test-convtr" => {
                test_convtr = true;
            },
            "--test-mimi-trace" => {
                test_mimi_trace = true;
            },
            "--temperature" => {
                if i + 1 < args.len() {
                    temperature = Some(args[i + 1].parse().expect("Invalid temperature value"));
                    i += 1;
                }
            },
            "--top-p" => {
                if i + 1 < args.len() {
                    top_p = Some(args[i + 1].parse().expect("Invalid top-p value"));
                    i += 1;
                }
            },
            "--consistency-steps" => {
                if i + 1 < args.len() {
                    consistency_steps = Some(args[i + 1].parse().expect("Invalid consistency-steps value"));
                    i += 1;
                }
            },
            "--speed" => {
                if i + 1 < args.len() {
                    speed = Some(args[i + 1].parse().expect("Invalid speed value"));
                    i += 1;
                }
            },
            "--seed" => {
                if i + 1 < args.len() {
                    seed = Some(args[i + 1].parse().expect("Invalid seed value"));
                    i += 1;
                }
            },
            "--help" | "-h" => {
                print_usage();
                return;
            },
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                return;
            },
        }
        i += 1;
    }

    // Build TTS config from CLI args
    let mut tts_config = TTSConfig::default();
    if let Some(v) = temperature { tts_config.temperature = v; }
    if let Some(v) = top_p { tts_config.top_p = v; }
    if let Some(v) = consistency_steps { tts_config.consistency_steps = v; }
    if let Some(v) = speed { tts_config.speed = v; }
    if let Some(v) = seed {
        tts_config.seed = v;
        tts_config.use_fixed_seed = true;
    }

    // Run in validation mode, latent-load mode, or single-phrase mode
    if test_mimi_trace {
        run_mimi_trace_test(&model_dir);
    } else if test_convtr {
        run_convtr_test(&model_dir);
    } else if let Some(latents_path) = load_latents_path {
        run_mimi_from_latents(&model_dir, &latents_path, &output_path);
    } else if validation_mode {
        run_validation_mode(
            &model_dir,
            &validation_output_dir,
            json_report.as_ref().map(|v| &**v),
            extended_validation,
        );
    } else {
        run_single_phrase(&model_dir, &output_path, &test_text, export_latents_path.as_ref().map(|v| &**v), &tts_config);
    }
}

/// Run validation mode: synthesize all test phrases and create manifest
fn run_validation_mode(model_dir: &Path, output_dir: &Path, json_report: Option<&Path>, extended: bool) {
    println!("=== VALIDATION MODE ===\n");
    println!("Model directory: {}", model_dir.display());
    println!("Output directory: {}", output_dir.display());

    // Select which phrases to test
    let test_phrases: Vec<&str> = if extended {
        println!("Mode: EXTENDED (includes long paragraph tests)\n");
        TEST_PHRASES.iter().chain(LONG_TEST_PHRASES.iter()).copied().collect()
    } else {
        println!("Mode: STANDARD (short phrases only)\n");
        TEST_PHRASES.to_vec()
    };
    println!("Test phrases: {}\n", test_phrases.len());

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

    for (idx, phrase) in test_phrases.iter().enumerate() {
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
            },
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
        println!(
            "  NaN: {}, Inf: {}, Clipped: {}",
            stats.nan_count, stats.inf_count, stats.clip_count
        );

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
    println!("Phrases processed: {}/{}", phrase_results.len(), test_phrases.len());
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
fn run_single_phrase(model_dir: &Path, output_path: &Path, test_text: &str, export_latents_path: Option<&Path>, config: &TTSConfig) {
    println!("Configuration:");
    println!("  Model directory: {}", model_dir.display());
    println!("  Output file: {}", output_path.display());
    println!("  Test text: \"{}\"\n", test_text);
    println!("  Temperature: {}", config.temperature);
    println!("  Top-P: {}", config.top_p);
    println!("  Consistency steps: {}", config.consistency_steps);
    println!("  Speed: {}", config.speed);
    println!("  Seed: {} (fixed: {})", config.seed, config.use_fixed_seed);
    if let Some(latent_path) = export_latents_path {
        println!("  Export latents to: {}", latent_path.display());
    }

    let mut model = load_model(model_dir);

    // Apply TTS configuration
    if let Err(e) = model.configure(config.clone()) {
        eprintln!("ERROR: Invalid config: {:?}", e);
        std::process::exit(1);
    }
    let sample_rate = model.sample_rate();

    // Run synthesis (with or without latent export)
    println!("Synthesizing audio...");
    let start = Instant::now();

    let (audio, latents_data, latent_shape) = if export_latents_path.is_some() {
        // Use synthesize_with_latents to get both audio and latents
        match model.synthesize_with_latents(test_text) {
            Ok((a, l, s)) => (a, Some(l), Some(s)),
            Err(e) => {
                eprintln!("ERROR: Synthesis failed: {:?}", e);
                std::process::exit(1);
            },
        }
    } else {
        // Use regular synthesize
        match model.synthesize(test_text) {
            Ok(a) => (a, None, None),
            Err(e) => {
                eprintln!("ERROR: Synthesis failed: {:?}", e);
                std::process::exit(1);
            },
        }
    };
    let synthesis_time = start.elapsed().as_secs_f32();

    // Export latents if requested
    if let (Some(latent_path), Some(latents), Some(shape)) = (export_latents_path, &latents_data, &latent_shape) {
        println!("Exporting latents to {}...", latent_path.display());
        println!("  Shape: [{}, {}, {}]", shape[0], shape[1], shape[2]);
        if let Err(e) = write_npy(latent_path, latents, shape) {
            eprintln!("ERROR: Failed to write latents: {:?}", e);
        } else {
            println!("  Latents saved successfully");
        }
    }

    println!("Synthesis complete in {:.2}s", synthesis_time);
    println!("  Audio samples: {}", audio.len());
    println!("  Duration: {:.2}s", audio.len() as f32 / sample_rate as f32);
    println!(
        "  Real-time factor: {:.2}x\n",
        (audio.len() as f32 / sample_rate as f32) / synthesis_time
    );

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
        println!("  Last 10 samples: {:?}", &audio[audio.len() - 10..]);
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
fn load_model(model_dir: &Path) -> PocketTTSModel {
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
        },
    };
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!("  Version: {}", model.version());
    println!("  Parameters: {}", model.parameter_count());
    println!("  Sample rate: {} Hz\n", model.sample_rate());

    model
}

/// Write audio to WAV file
fn write_wav(path: &Path, audio: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
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

/// Run Mimi decoder from pre-saved latents (for debugging/comparison)
///
/// This mode loads denormalized latents from a raw f32 binary file and
/// runs them through the Mimi decoder to compare intermediate values
/// with the Python reference implementation.
fn run_mimi_from_latents(model_dir: &Path, latents_path: &Path, output_path: &Path) {
    println!("=== MIMI LATENT TEST MODE ===\n");
    println!("Model directory: {}", model_dir.display());
    println!("Latents file: {}", latents_path.display());
    println!("Output file: {}", output_path.display());

    // Load raw f32 latents
    let mut latents_data = Vec::new();
    let mut file = File::open(latents_path).expect("Failed to open latents file");
    file.read_to_end(&mut latents_data).expect("Failed to read latents");

    // Convert bytes to f32 (little-endian)
    let latents_f32: Vec<f32> = latents_data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Determine shape from file size
    // Python saves [44, 32] for standard test phrase
    let latent_dim = 32;
    let seq_len = latents_f32.len() / latent_dim;
    println!("\nLoaded {} latent values: [{}, {}]", latents_f32.len(), seq_len, latent_dim);

    // Print first 8 values to verify match with Python
    println!("First 8 latent values (frame 0): {:?}", &latents_f32[..8]);

    let device = Device::Cpu;

    // Create tensor [batch=1, seq, latent_dim]
    let latents =
        Tensor::from_vec(latents_f32.clone(), (1, seq_len, latent_dim), &device).expect("Failed to create tensor");
    println!("Latent tensor shape: {:?}", latents.dims());

    // Load model weights
    let model_path = model_dir.join("model.safetensors");
    println!("\nLoading model weights...");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device).expect("Failed to load weights")
    };

    // Create Mimi decoder
    let mimi_config = MimiConfig::default();
    let mimi = MimiDecoder::new(mimi_config, vb.pp("mimi")).expect("Failed to create Mimi decoder");

    // Run forward pass
    println!("\nRunning Mimi decoder...");
    let audio = mimi.forward_streaming(&latents).expect("Failed to run Mimi");

    println!("\nAudio output shape: {:?}", audio.dims());

    // Get audio samples
    let audio_vec: Vec<f32> = audio.squeeze(0).expect("squeeze").to_vec1().expect("to_vec1");
    println!("Audio samples: {}", audio_vec.len());

    // Audio statistics
    let max_amp = audio_vec.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let mean_amp = audio_vec.iter().map(|s| s.abs()).sum::<f32>() / audio_vec.len() as f32;
    println!("Max amplitude: {:.4}", max_amp);
    println!("Mean amplitude: {:.4}", mean_amp);

    // Write WAV file
    println!("\nWriting WAV file...");
    if let Err(e) = write_wav(output_path, &audio_vec, 24000) {
        eprintln!("ERROR: Failed to write WAV: {:?}", e);
    } else {
        println!("Saved: {}", output_path.display());
    }

    // Compare with Python reference if available
    let python_audio_path = latents_path.parent().unwrap().join("python_mimi_output.npy");
    if python_audio_path.exists() {
        println!("\n=== Comparison with Python ===");
        println!("Python output: {}", python_audio_path.display());
        println!(
            "Run validation/compare_mimi_outputs.py --rust-audio {} to compare",
            output_path.display()
        );
    }
}

/// Trace Mimi pipeline and compare with Python intermediate values
fn run_mimi_trace_test(model_dir: &Path) {
    println!("=== MIMI PIPELINE TRACE TEST ===\n");

    let device = Device::Cpu;
    let model_path = model_dir.join("model.safetensors");

    // Load weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device).expect("Failed to load weights")
    };

    // Load Python's input from /tmp/mimi_trace_input.f32
    let input_path = Path::new("/tmp/mimi_trace_input.f32");
    if !input_path.exists() {
        eprintln!("ERROR: Run python3 /tmp/trace_mimi_pipeline.py first to generate test inputs");
        std::process::exit(1);
    }

    let mut input_bytes = Vec::new();
    File::open(input_path)
        .expect("Failed to open input file")
        .read_to_end(&mut input_bytes)
        .expect("Failed to read input");

    let input_f32: Vec<f32> = input_bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    println!("Input len: {} (expected 128 = 1*4*32)", input_f32.len());

    // Create input tensor [1, 4, 32]
    let input = Tensor::from_vec(input_f32.clone(), (1, 4, 32), &device).expect("Failed to create input tensor");
    println!("Input shape: {:?}", input.dims());
    println!("Input [0, 0, :8]: {:?}", &input_f32[..8]);

    // Step 1: Transpose to [1, 32, 4]
    let x = input.permute((0, 2, 1)).expect("permute failed");
    println!("\n1. After transpose: {:?}", x.dims());
    let x_vec: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    // First column (frame 0) is every 4th element
    let frame0_vals: Vec<f32> = (0..8).map(|i| x_vec[i * 4]).collect();
    println!("   [0, :8, 0]: {:?}", frame0_vals);

    // Step 2: output_proj - Conv1d(32, 512, kernel=1)
    let proj_weight = vb
        .get((512, 32, 1), "mimi.quantizer.output_proj.weight")
        .expect("get proj weight");
    let proj_bias = vb.get(512, "mimi.quantizer.output_proj.bias").ok();
    println!("\n2. output_proj weight: {:?}", proj_weight.shape());

    let x2 = x.conv1d(&proj_weight, 0, 1, 1, 1).expect("conv1d failed");
    let x2 = if let Some(bias) = &proj_bias {
        let bias = bias.reshape((1, (), 1)).unwrap();
        x2.broadcast_add(&bias).unwrap()
    } else {
        x2
    };
    println!("   After output_proj: {:?}", x2.dims());
    let x2_vec: Vec<f32> = x2.flatten_all().unwrap().to_vec1().unwrap();
    let x2_frame0: Vec<f32> = (0..8).map(|i| x2_vec[i * 4]).collect();
    println!("   [0, :8, 0]: {:?}", x2_frame0);

    // Python reference values
    let python_proj: Vec<f32> = vec![
        -0.13605336844921112,
        1.7759066820144653,
        1.2895563840866089,
        0.6906523704528809,
        1.609748125076294,
        -0.5624573230743408,
        -1.8276222944259644,
        -0.5808132886886597,
    ];
    println!("   Python [0, :8, 0]: {:?}", python_proj);

    let proj_max_diff: f32 = python_proj
        .iter()
        .zip(x2_frame0.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0f32, |a, b| a.max(b));
    println!("   Max diff from Python: {:.6}", proj_max_diff);

    // Step 3: upsample - ConvTranspose1d(512, 512, k=32, s=16, groups=512)
    let upsample_weight = vb
        .get((512, 1, 32), "mimi.upsample.convtr.convtr.weight")
        .expect("get upsample weight");
    let upsample_bias = vb.get(512, "mimi.upsample.convtr.convtr.bias").ok();
    println!("\n3. upsample weight: {:?}", upsample_weight.shape());

    let x3 = x2
        .conv_transpose1d(&upsample_weight, 0, 0, 16, 1, 512)
        .expect("conv_transpose1d failed");
    let x3 = if let Some(bias) = &upsample_bias {
        let bias = bias.reshape((1, (), 1)).unwrap();
        x3.broadcast_add(&bias).unwrap()
    } else {
        x3
    };
    println!("   After upsample (raw): {:?}", x3.dims());

    let x3_vec: Vec<f32> = x3.flatten_all().unwrap().to_vec1().unwrap();
    // x3 is [1, 512, 80], get first 8 channels at frame 0
    let x3_ch_frame0: Vec<f32> = (0..8).map(|i| x3_vec[i * 80]).collect();
    println!("   [0, :8, 0]: {:?}", x3_ch_frame0);
    // Also get first 8 time steps of channel 0
    let x3_ch0_time: Vec<f32> = (0..8).map(|i| x3_vec[i]).collect();
    println!("   [0, 0, :8]: {:?}", x3_ch0_time);

    // Python reference
    let python_upsample_ch: Vec<f32> = vec![
        -0.039062198251485825,
        0.2705482840538025,
        -0.2027525156736374,
        -0.059690169990062714,
        0.21379467844963074,
        -0.0829404816031456,
        1.0637333393096924,
        0.10720089077949524,
    ];
    let python_upsample_time: Vec<f32> = vec![
        -0.039062198251485825,
        -0.03773355111479759,
        -0.034544799476861954,
        -0.03268469497561455,
        -0.033349018543958664,
        -0.026838652789592743,
        -0.02790156938135624,
        -0.025377141311764717,
    ];
    println!("   Python [0, :8, 0]: {:?}", python_upsample_ch);
    println!("   Python [0, 0, :8]: {:?}", python_upsample_time);

    let upsample_ch_diff: f32 = python_upsample_ch
        .iter()
        .zip(x3_ch_frame0.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0f32, |a, b| a.max(b));
    let upsample_time_diff: f32 = python_upsample_time
        .iter()
        .zip(x3_ch0_time.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0f32, |a, b| a.max(b));
    println!("\n   Max diff channels: {:.6}", upsample_ch_diff);
    println!("   Max diff time: {:.6}", upsample_time_diff);

    if proj_max_diff < 1e-4 && upsample_ch_diff < 1e-4 && upsample_time_diff < 1e-4 {
        println!("\nPASS: Pipeline matches Python through upsample!");
    } else {
        println!("\nFAIL: Discrepancy detected");
        if proj_max_diff >= 1e-4 {
            println!("  - output_proj differs");
        }
        if upsample_ch_diff >= 1e-4 || upsample_time_diff >= 1e-4 {
            println!("  - upsample differs");
        }
    }
}

/// Test conv_transpose against Python reference
fn run_convtr_test(model_dir: &Path) {
    println!("=== CONV_TRANSPOSE TEST ===\n");

    let device = Device::Cpu;
    let model_path = model_dir.join("model.safetensors");

    // Load weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device).expect("Failed to load weights")
    };

    // Get first upsample weights [512, 1, 32] depthwise ConvTranspose
    let weight = vb
        .get((512, 1, 32), "mimi.upsample.convtr.convtr.weight")
        .expect("Failed to get weight");
    println!("Weight shape: {:?}", weight.shape());

    // Load Python's input from /tmp/convtr_input.f32
    let input_path = Path::new("/tmp/convtr_input.f32");
    if !input_path.exists() {
        eprintln!("ERROR: Run python3 /tmp/test_convtr_basic.py first to generate test inputs");
        std::process::exit(1);
    }

    let mut input_bytes = Vec::new();
    File::open(input_path)
        .expect("Failed to open input file")
        .read_to_end(&mut input_bytes)
        .expect("Failed to read input");

    let input_f32: Vec<f32> = input_bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    println!("Input len: {} (expected 512)", input_f32.len());
    println!("Input first 5: {:?}", &input_f32[..5]);

    // Create input tensor [1, 512, 1]
    let input = Tensor::from_vec(input_f32, (1, 512, 1), &device).expect("Failed to create input tensor");

    // Raw conv_transpose with groups=512 (depthwise)
    // kernel_size=32, stride=16
    let output = input
        .conv_transpose1d(
            &weight, 0,   // padding
            0,   // output_padding
            16,  // stride
            1,   // dilation
            512, // groups
        )
        .expect("Failed to run conv_transpose");

    println!("Output shape: {:?}", output.shape());

    // Get channel 0 output
    let out_ch0 = output.narrow(1, 0, 1).unwrap().squeeze(0).unwrap().squeeze(0).unwrap();
    let out_vals: Vec<f32> = out_ch0.to_vec1().unwrap();
    println!("Rust channel 0, first 16: {:?}", &out_vals[..16]);

    // Python reference values (from test_convtr_basic.py output)
    let python_first_16: Vec<f32> = vec![
        0.5532353520393372,
        0.5344178676605225,
        0.48925578594207764,
        0.4629112184047699,
        0.47231999039649963,
        0.3801141083240509,
        0.39516812562942505,
        0.35941481590270996,
        0.3349520266056061,
        0.23615999519824982,
        0.227692112326622,
        0.13360446691513062,
        0.06915441900491714,
        0.04939601570367813,
        0.07809274643659592,
        0.07291793078184128,
    ];

    println!("\n=== Comparison ===");
    println!("Python first 5: {:?}", &python_first_16[..5]);
    println!("Rust first 5:   {:?}", &out_vals[..5]);

    // Calculate max difference
    let max_diff: f32 = python_first_16
        .iter()
        .zip(out_vals.iter())
        .map(|(p, r)| (p - r).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("\nMax diff: {:.6}", max_diff);

    if max_diff < 1e-4 {
        println!("PASS: Raw conv_transpose matches Python!");
    } else {
        println!("FAIL: Output differs from Python");

        // Calculate ratios
        println!("\nRatios (Rust/Python):");
        for i in 0..8 {
            if python_first_16[i].abs() > 1e-6 {
                println!("  [{}]: {:.4}", i, out_vals[i] / python_first_16[i]);
            }
        }
    }
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
    println!("  -e, --extended             Include extended (long paragraph) test phrases");
    println!("  --validation-output PATH   Output dir for validation (default: ./validation/rust_outputs)");
    println!("  --export-latents PATH      Export latents to .npy file (for debugging)");
    println!("  --load-latents PATH        Load pre-saved latents (.f32) and run through Mimi");
    println!("  --json-report PATH         Write JSON report to file");
    println!("  --temperature FLOAT        Sampling temperature (0.0-1.0, default: 0.7)");
    println!("  --top-p FLOAT              Top-P nucleus sampling (0.1-1.0, default: 0.9)");
    println!("  --consistency-steps INT    Consistency steps (1-4, default: 2)");
    println!("  --speed FLOAT              Speed multiplier (0.5-2.0, default: 1.0)");
    println!("  --seed INT                 Fixed random seed (enables deterministic mode)");
    println!("  -h, --help                 Show this help message");
    println!("\nThe model directory should contain:");
    println!("  - model.safetensors");
    println!("  - tokenizer.model");
    println!("  - voices/ directory with voice embeddings");
    println!("\nValidation mode outputs:");
    println!("  - phrase_XX_rust.wav       Audio files for each test phrase");
    println!("  - manifest.json            Statistics and metadata");
}
