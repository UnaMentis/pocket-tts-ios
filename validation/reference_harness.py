#!/usr/bin/env python3
"""
Pocket TTS Reference Harness

Generates reference outputs using the official Kyutai Python implementation.
These outputs serve as ground truth for validating the Rust/Candle port.

Usage:
    python reference_harness.py --output-dir ./reference_outputs
    python reference_harness.py --output-dir ./reference_outputs --with-whisper
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

# Test phrases for validation
TEST_PHRASES = [
    "Hello, this is a test of the Pocket TTS system.",
    "The quick brown fox jumps over the lazy dog.",
    "One two three four five six seven eight nine ten.",
    "How are you doing today?",
]


def generate_reference_outputs(output_dir: Path, voice: str = "alba") -> dict:
    """Generate reference audio and latents using official Pocket TTS."""
    from pocket_tts import TTSModel

    print("Loading Pocket TTS model...")
    model = TTSModel.load_model()
    print(f"  Sample rate: {model.sample_rate} Hz")

    print(f"Getting voice state for '{voice}'...")
    voice_state = model.get_state_for_audio_prompt(voice)

    results = {
        "model_version": "official_pocket_tts",
        "sample_rate": model.sample_rate,
        "voice": voice,
        "phrases": []
    }

    for i, phrase in enumerate(tqdm(TEST_PHRASES, desc="Generating audio")):
        phrase_id = f"phrase_{i:02d}"

        # Generate audio
        audio = model.generate_audio(voice_state, phrase)
        audio_np = audio.numpy()

        # Save audio as WAV
        wav_path = output_dir / f"{phrase_id}.wav"
        wavfile.write(str(wav_path), model.sample_rate, audio_np)

        # Save audio as raw float32 for precise comparison
        npy_path = output_dir / f"{phrase_id}_audio.npy"
        np.save(str(npy_path), audio_np)

        # Compute audio statistics
        audio_stats = {
            "samples": len(audio_np),
            "duration_sec": len(audio_np) / model.sample_rate,
            "max_amplitude": float(np.max(np.abs(audio_np))),
            "mean_amplitude": float(np.mean(np.abs(audio_np))),
            "rms": float(np.sqrt(np.mean(audio_np ** 2))),
            "dc_offset": float(np.mean(audio_np)),
        }

        phrase_result = {
            "id": phrase_id,
            "text": phrase,
            "wav_file": str(wav_path.name),
            "npy_file": str(npy_path.name),
            "audio_stats": audio_stats,
        }

        results["phrases"].append(phrase_result)

        print(f"  {phrase_id}: {audio_stats['samples']} samples, "
              f"{audio_stats['duration_sec']:.2f}s, "
              f"max={audio_stats['max_amplitude']:.4f}")

    return results


def run_whisper_transcription(output_dir: Path, results: dict) -> dict:
    """Run Whisper ASR on generated audio to establish baseline WER."""
    try:
        import whisper
        from jiwer import wer, cer
    except ImportError:
        print("Warning: whisper or jiwer not installed, skipping ASR evaluation")
        return results

    print("\nLoading Whisper model...")
    whisper_model = whisper.load_model("base")

    for phrase_info in tqdm(results["phrases"], desc="Transcribing"):
        wav_path = output_dir / phrase_info["wav_file"]

        # Transcribe
        result = whisper_model.transcribe(str(wav_path), language="en")
        transcription = result["text"].strip()

        # Compute WER and CER
        reference = phrase_info["text"]
        word_error_rate = wer(reference.lower(), transcription.lower())
        char_error_rate = cer(reference.lower(), transcription.lower())

        phrase_info["asr"] = {
            "transcription": transcription,
            "wer": word_error_rate,
            "cer": char_error_rate,
        }

        print(f"  {phrase_info['id']}: WER={word_error_rate:.1%}, "
              f"'{transcription[:50]}...'")

    # Compute aggregate WER
    total_wer = np.mean([p["asr"]["wer"] for p in results["phrases"]])
    total_cer = np.mean([p["asr"]["cer"] for p in results["phrases"]])
    results["aggregate_wer"] = float(total_wer)
    results["aggregate_cer"] = float(total_cer)

    print(f"\nAggregate WER: {total_wer:.1%}")
    print(f"Aggregate CER: {total_cer:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pocket TTS reference outputs for validation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "reference_outputs",
        help="Directory to save reference outputs"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="alba",
        help="Voice to use (default: alba)"
    )
    parser.add_argument(
        "--with-whisper",
        action="store_true",
        help="Run Whisper transcription to establish baseline WER"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate outputs even if they exist"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if outputs already exist
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"Reference outputs already exist at {args.output_dir}")
        print("Use --force to regenerate")
        return

    print(f"Generating reference outputs to {args.output_dir}")
    print(f"Test phrases: {len(TEST_PHRASES)}")
    print()

    # Generate reference outputs
    results = generate_reference_outputs(args.output_dir, args.voice)

    # Optionally run Whisper
    if args.with_whisper:
        results = run_whisper_transcription(args.output_dir, results)

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nReference outputs saved to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
