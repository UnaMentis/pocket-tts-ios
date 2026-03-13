#!/usr/bin/env python3
"""
Pocket TTS Reference Harness

Generates reference outputs using the official Kyutai Python implementation.
These outputs serve as ground truth for validating the Rust/Candle port.

Usage:
    python reference_harness.py --output-dir ./reference_outputs
    python reference_harness.py --output-dir ./reference_outputs --with-whisper
    python reference_harness.py --output-dir ./reference_outputs --capture-noise --seed 42
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List

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


class NoiseCapture:
    """Captures noise tensors from torch.nn.init.normal_ during FlowNet generation.

    When capture_noise=True, this hooks into PyTorch's normal_ initialization
    to intercept the noise tensors used by FlowNet at each generation step.
    These captured tensors can be loaded into Rust to eliminate RNG differences
    and measure pure implementation correlation.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.captured: List[np.ndarray] = []
        self._original_normal_ = None

    def start(self):
        """Install the hook."""
        if not self.enabled:
            return
        import torch
        self._original_normal_ = torch.nn.init.normal_

        captured = self.captured

        def capturing_normal_(tensor, mean=0.0, std=1.0):
            result = self._original_normal_(tensor, mean=mean, std=std)
            # Only capture noise tensors that look like FlowNet noise
            # Shape is typically [1, 1, 32] or [1, seq, 32]
            if tensor.dim() == 3 and tensor.shape[-1] == 32:
                captured.append(tensor.detach().cpu().numpy().copy())
            return result

        torch.nn.init.normal_ = capturing_normal_

    def stop(self):
        """Remove the hook and restore original function."""
        if not self.enabled or self._original_normal_ is None:
            return
        import torch
        torch.nn.init.normal_ = self._original_normal_
        self._original_normal_ = None

    def save(self, output_dir: Path, phrase_id: str):
        """Save captured noise tensors to .npy files."""
        if not self.enabled or not self.captured:
            return 0

        noise_dir = output_dir / "noise"
        noise_dir.mkdir(parents=True, exist_ok=True)

        for step, noise in enumerate(self.captured):
            npy_path = noise_dir / f"{phrase_id}_noise_step_{step:03d}.npy"
            np.save(str(npy_path), noise)

        count = len(self.captured)
        self.captured.clear()
        return count


def generate_reference_outputs(output_dir: Path, voice: str = "alba",
                                seed: Optional[int] = None,
                                capture_noise: bool = False) -> dict:
    """Generate reference audio and latents using official Pocket TTS.

    Args:
        output_dir: Directory to save outputs
        voice: Voice name to use
        seed: If set, seed PyTorch RNG before each phrase for deterministic noise
        capture_noise: If True, capture FlowNet noise tensors as .npy files
    """
    import torch
    from pocket_tts import TTSModel

    print("Loading Pocket TTS model...")
    model = TTSModel.load_model()
    print(f"  Sample rate: {model.sample_rate} Hz")

    if seed is not None:
        print(f"  Using seed: {seed} (deterministic mode)")
    if capture_noise:
        print(f"  Capturing FlowNet noise tensors")

    print(f"Getting voice state for '{voice}'...")
    voice_state = model.get_state_for_audio_prompt(voice)

    noise_capture = NoiseCapture(enabled=capture_noise)

    results = {
        "model_version": "official_pocket_tts",
        "sample_rate": model.sample_rate,
        "voice": voice,
        "seed": seed,
        "noise_captured": capture_noise,
        "phrases": []
    }

    for i, phrase in enumerate(tqdm(TEST_PHRASES, desc="Generating audio")):
        phrase_id = f"phrase_{i:02d}"

        # Set seed before each phrase for deterministic generation
        if seed is not None:
            torch.manual_seed(seed + i)

        # Start noise capture
        noise_capture.start()

        # Generate audio
        audio = model.generate_audio(voice_state, phrase)
        audio_np = audio.numpy()

        # Stop noise capture and save
        noise_capture.stop()
        noise_count = noise_capture.save(output_dir, phrase_id)

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

        if noise_count > 0:
            phrase_result["noise_tensors"] = noise_count
            phrase_result["noise_dir"] = "noise"

        results["phrases"].append(phrase_result)

        noise_info = f", {noise_count} noise tensors" if noise_count > 0 else ""
        print(f"  {phrase_id}: {audio_stats['samples']} samples, "
              f"{audio_stats['duration_sec']:.2f}s, "
              f"max={audio_stats['max_amplitude']:.4f}{noise_info}")

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
        "--capture-noise",
        action="store_true",
        help="Capture FlowNet noise tensors as .npy files for Rust correlation testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic generation (e.g., 42). Required with --capture-noise."
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

    # Validate flags
    if args.capture_noise and args.seed is None:
        print("WARNING: --capture-noise without --seed means noise is non-deterministic.")
        print("         Use --seed 42 for reproducible noise tensors.")

    # Generate reference outputs
    results = generate_reference_outputs(
        args.output_dir, args.voice,
        seed=args.seed,
        capture_noise=args.capture_noise,
    )

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
