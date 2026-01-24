#!/usr/bin/env python3
"""
Pocket TTS Validation Script

Three-layer validation of the Rust/Candle implementation against the Python reference.

Layer 1: Reference Comparison (latent similarity, audio correlation)
Layer 2: ASR Round-Trip (Whisper transcription, WER comparison)
Layer 3: Signal Health (NaN/Inf, amplitude, DC offset)

Usage:
    python validate.py --model-dir /path/to/model
    python validate.py --model-dir /path/to/model --rust-binary ../target/release/test-tts
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile

# Thresholds for pass/fail
LATENT_COSINE_THRESHOLD = 0.99
AUDIO_CORRELATION_THRESHOLD = 0.95
WER_DELTA_THRESHOLD = 0.05  # 5% WER difference allowed
AMPLITUDE_MIN = 0.01
AMPLITUDE_MAX = 1.0
DC_OFFSET_MAX = 0.05


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    value: float
    threshold: float
    message: str


@dataclass
class LayerResult:
    """Result of a validation layer."""
    name: str
    passed: bool
    checks: list = field(default_factory=list)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays."""
    # Ensure same length
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def validate_layer1_reference(
    rust_audio_path: Path,
    ref_audio_path: Path,
    rust_latents_path: Optional[Path] = None,
    ref_latents_path: Optional[Path] = None,
) -> LayerResult:
    """Layer 1: Compare Rust output to Python reference."""
    checks = []

    # Load audio files
    rust_rate, rust_audio = wavfile.read(str(rust_audio_path))
    ref_rate, ref_audio = wavfile.read(str(ref_audio_path))

    # Normalize to float if needed
    if rust_audio.dtype == np.int16:
        rust_audio = rust_audio.astype(np.float32) / 32767.0
    if ref_audio.dtype == np.int16:
        ref_audio = ref_audio.astype(np.float32) / 32767.0

    # Check sample count
    sample_diff = abs(len(rust_audio) - len(ref_audio))
    sample_match = sample_diff < 100  # Allow small tolerance
    checks.append(ValidationResult(
        name="Sample count",
        passed=sample_match,
        value=float(sample_diff),
        threshold=100.0,
        message=f"Rust: {len(rust_audio)}, Ref: {len(ref_audio)}, Diff: {sample_diff}"
    ))

    # Audio correlation
    audio_corr = correlation(rust_audio, ref_audio)
    corr_passed = audio_corr >= AUDIO_CORRELATION_THRESHOLD
    checks.append(ValidationResult(
        name="Audio correlation",
        passed=corr_passed,
        value=audio_corr,
        threshold=AUDIO_CORRELATION_THRESHOLD,
        message=f"Correlation: {audio_corr:.4f}"
    ))

    # Latent comparison (if available)
    if rust_latents_path and ref_latents_path:
        if rust_latents_path.exists() and ref_latents_path.exists():
            rust_latents = np.load(str(rust_latents_path))
            ref_latents = np.load(str(ref_latents_path))

            latent_sim = cosine_similarity(rust_latents, ref_latents)
            latent_passed = latent_sim >= LATENT_COSINE_THRESHOLD
            checks.append(ValidationResult(
                name="Latent cosine similarity",
                passed=latent_passed,
                value=latent_sim,
                threshold=LATENT_COSINE_THRESHOLD,
                message=f"Similarity: {latent_sim:.4f}"
            ))

    all_passed = all(c.passed for c in checks)
    return LayerResult(name="Layer 1: Reference Match", passed=all_passed, checks=checks)


def validate_layer2_asr(
    rust_audio_path: Path,
    ref_manifest: dict,
    phrase_id: str,
) -> LayerResult:
    """Layer 2: ASR round-trip validation."""
    checks = []

    try:
        import whisper
        from jiwer import wer as compute_wer, cer as compute_cer
    except ImportError:
        return LayerResult(
            name="Layer 2: ASR Round-Trip",
            passed=True,  # Skip if not available
            checks=[ValidationResult(
                name="ASR",
                passed=True,
                value=0.0,
                threshold=0.0,
                message="Whisper not installed, skipping ASR validation"
            )]
        )

    # Find the phrase info
    phrase_info = None
    for p in ref_manifest.get("phrases", []):
        if p["id"] == phrase_id:
            phrase_info = p
            break

    if not phrase_info:
        return LayerResult(
            name="Layer 2: ASR Round-Trip",
            passed=False,
            checks=[ValidationResult(
                name="ASR",
                passed=False,
                value=0.0,
                threshold=0.0,
                message=f"Phrase {phrase_id} not found in reference manifest"
            )]
        )

    original_text = phrase_info["text"]
    ref_wer = phrase_info.get("asr", {}).get("wer", None)

    # Load and transcribe Rust audio
    print("  Loading Whisper model...")
    whisper_model = whisper.load_model("base")

    print("  Transcribing Rust audio...")
    result = whisper_model.transcribe(str(rust_audio_path), language="en")
    rust_transcription = result["text"].strip()

    rust_wer = compute_wer(original_text.lower(), rust_transcription.lower())
    rust_cer = compute_cer(original_text.lower(), rust_transcription.lower())

    checks.append(ValidationResult(
        name="Rust WER",
        passed=True,  # Informational
        value=rust_wer,
        threshold=0.0,
        message=f"WER: {rust_wer:.1%}, Transcription: '{rust_transcription[:50]}...'"
    ))

    # Compare to reference if available
    if ref_wer is not None:
        wer_delta = abs(rust_wer - ref_wer)
        wer_delta_passed = wer_delta <= WER_DELTA_THRESHOLD
        checks.append(ValidationResult(
            name="WER delta vs reference",
            passed=wer_delta_passed,
            value=wer_delta,
            threshold=WER_DELTA_THRESHOLD,
            message=f"Rust WER: {rust_wer:.1%}, Ref WER: {ref_wer:.1%}, Delta: {wer_delta:.1%}"
        ))

    all_passed = all(c.passed for c in checks)
    return LayerResult(name="Layer 2: ASR Round-Trip", passed=all_passed, checks=checks)


def validate_layer3_signal(rust_audio_path: Path) -> LayerResult:
    """Layer 3: Signal health checks."""
    checks = []

    # Load audio
    rate, audio = wavfile.read(str(rust_audio_path))

    # Normalize to float if needed
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32767.0

    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(audio))
    inf_count = np.sum(np.isinf(audio))
    no_nan_inf = nan_count == 0 and inf_count == 0
    checks.append(ValidationResult(
        name="No NaN/Inf",
        passed=no_nan_inf,
        value=float(nan_count + inf_count),
        threshold=0.0,
        message=f"NaN: {nan_count}, Inf: {inf_count}"
    ))

    # Amplitude range
    max_amp = float(np.max(np.abs(audio)))
    amp_in_range = AMPLITUDE_MIN <= max_amp <= AMPLITUDE_MAX
    checks.append(ValidationResult(
        name="Amplitude range",
        passed=amp_in_range,
        value=max_amp,
        threshold=AMPLITUDE_MIN,
        message=f"Max amplitude: {max_amp:.4f} (expected {AMPLITUDE_MIN}-{AMPLITUDE_MAX})"
    ))

    # DC offset
    dc_offset = float(np.mean(audio))
    dc_ok = abs(dc_offset) <= DC_OFFSET_MAX
    checks.append(ValidationResult(
        name="DC offset",
        passed=dc_ok,
        value=abs(dc_offset),
        threshold=DC_OFFSET_MAX,
        message=f"DC offset: {dc_offset:.6f}"
    ))

    # RMS (informational)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    checks.append(ValidationResult(
        name="RMS level",
        passed=True,  # Informational
        value=rms,
        threshold=0.0,
        message=f"RMS: {rms:.4f}"
    ))

    all_passed = all(c.passed for c in checks)
    return LayerResult(name="Layer 3: Signal Health", passed=all_passed, checks=checks)


def run_rust_harness(
    rust_binary: Path,
    model_dir: Path,
    output_wav: Path,
    output_latents: Optional[Path] = None,
    text: str = "Hello, this is a test of the Pocket TTS system.",
) -> bool:
    """Run the Rust test harness to generate output."""
    cmd = [
        str(rust_binary),
        "--model-dir", str(model_dir),
        "--output", str(output_wav),
        "--text", text,
    ]

    if output_latents:
        cmd.extend(["--export-latents", str(output_latents)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Rust harness failed:")
        print(result.stderr)
        return False

    print(result.stdout)
    return True


def print_results(layers: list[LayerResult]) -> bool:
    """Print validation results and return overall pass/fail."""
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    all_passed = True

    for layer in layers:
        status = "PASS" if layer.passed else "FAIL"
        symbol = "✓" if layer.passed else "✗"
        print(f"\n{symbol} {layer.name}: {status}")

        for check in layer.checks:
            check_symbol = "✓" if check.passed else "✗"
            print(f"  {check_symbol} {check.name}: {check.message}")

        if not layer.passed:
            all_passed = False

    print("\n" + "=" * 60)
    final_status = "PASS" if all_passed else "FAIL"
    print(f"OVERALL RESULT: {final_status}")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate Pocket TTS Rust implementation against Python reference"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to Pocket TTS model directory"
    )
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=None,
        help="Path to Rust test-tts binary (default: auto-detect)"
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(__file__).parent / "reference_outputs",
        help="Directory containing reference outputs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "rust_outputs",
        help="Directory to save Rust outputs"
    )
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        help="Skip running Rust harness (use existing outputs)"
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="Skip ASR round-trip validation"
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--phrase-id",
        type=str,
        default="phrase_00",
        help="Which phrase to validate (default: phrase_00)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check reference outputs exist
    ref_manifest_path = args.reference_dir / "manifest.json"
    if not ref_manifest_path.exists():
        print(f"Reference outputs not found at {args.reference_dir}")
        print("Run: python reference_harness.py --with-whisper first")
        sys.exit(1)

    with open(ref_manifest_path) as f:
        ref_manifest = json.load(f)

    # Find Rust binary
    if args.rust_binary is None:
        # Try to find it
        possible_paths = [
            Path(__file__).parent.parent / "target" / "release" / "test-tts",
            Path(__file__).parent.parent / "target" / "debug" / "test-tts",
        ]
        for p in possible_paths:
            if p.exists():
                args.rust_binary = p
                break

    if args.rust_binary is None or not args.rust_binary.exists():
        print("Rust binary not found. Build with: cargo build --bin test-tts")
        if not args.skip_rust:
            sys.exit(1)

    # Get phrase info
    phrase_info = None
    for p in ref_manifest.get("phrases", []):
        if p["id"] == args.phrase_id:
            phrase_info = p
            break

    if not phrase_info:
        print(f"Phrase {args.phrase_id} not found in reference manifest")
        sys.exit(1)

    text = phrase_info["text"]
    ref_wav = args.reference_dir / phrase_info["wav_file"]
    ref_npy = args.reference_dir / phrase_info.get("npy_file", "")

    rust_wav = args.output_dir / f"{args.phrase_id}_rust.wav"
    rust_latents = args.output_dir / f"{args.phrase_id}_rust_latents.npy"

    print(f"Validating phrase: '{text}'")
    print(f"Reference: {ref_wav}")
    print(f"Rust output: {rust_wav}")
    print()

    # Run Rust harness
    if not args.skip_rust:
        print("Running Rust test harness...")
        if not run_rust_harness(
            args.rust_binary,
            args.model_dir,
            rust_wav,
            rust_latents,
            text,
        ):
            print("Rust harness failed")
            sys.exit(1)

    if not rust_wav.exists():
        print(f"Rust output not found: {rust_wav}")
        sys.exit(1)

    # Run validation layers
    layers = []

    print("\nLayer 1: Reference Comparison...")
    layer1 = validate_layer1_reference(
        rust_wav,
        ref_wav,
        rust_latents if rust_latents.exists() else None,
        ref_npy if ref_npy.exists() else None,
    )
    layers.append(layer1)

    if not args.skip_asr:
        print("\nLayer 2: ASR Round-Trip...")
        layer2 = validate_layer2_asr(rust_wav, ref_manifest, args.phrase_id)
        layers.append(layer2)

    print("\nLayer 3: Signal Health...")
    layer3 = validate_layer3_signal(rust_wav)
    layers.append(layer3)

    # Print results
    passed = print_results(layers)

    # Save JSON report if requested
    if args.json_report:
        report = {
            "passed": passed,
            "phrase": text,
            "layers": [
                {
                    "name": l.name,
                    "passed": l.passed,
                    "checks": [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "value": c.value,
                            "threshold": c.threshold,
                            "message": c.message,
                        }
                        for c in l.checks
                    ]
                }
                for l in layers
            ]
        }
        with open(args.json_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {args.json_report}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
