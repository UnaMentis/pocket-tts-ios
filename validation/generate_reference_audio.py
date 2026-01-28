#!/usr/bin/env python3
"""
Generate reference audio files for iOS app AB testing.

IMPORTANT: This uses the ACTUAL Python Pocket TTS implementation to create
REAL SPEECH audio. Never use random latents or synthetic data for testing.

Usage:
    cd validation
    source .venv/bin/activate  # or venv/bin/activate
    python generate_reference_audio.py
"""

import json
import struct
import wave
from pathlib import Path

import numpy as np

# Test phrases for iOS AB testing
REFERENCE_PHRASES = [
    {
        "id": "short",
        "text": "Hello world.",
        "description": "Short phrase for quick testing"
    },
    {
        "id": "medium",
        "text": "The quick brown fox jumps over the lazy dog.",
        "description": "Medium phrase with all letters"
    },
    {
        "id": "long",
        "text": "This is a longer test sentence to verify that the text to speech system handles extended input correctly and produces natural sounding audio.",
        "description": "Long phrase for extended testing"
    }
]

SAMPLE_RATE = 24000
VOICE = "alba"


def save_wav_int16(samples: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE):
    """Save audio as Int16 PCM WAV (iOS compatible)."""
    # Normalize to [-1, 1] if needed
    max_val = np.abs(samples).max()
    if max_val > 1.0:
        samples = samples / max_val * 0.95

    # Convert to int16
    int16_samples = (np.clip(samples, -1, 1) * 32767).astype(np.int16)

    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_samples.tobytes())

    return path


def main():
    # Import the actual Pocket TTS model
    try:
        from pocket_tts import TTSModel
    except ImportError:
        print("ERROR: pocket_tts module not found!")
        print("Make sure you have the Python Pocket TTS installed.")
        print("Try: pip install pocket-tts  OR  check your PYTHONPATH")
        return 1

    # Output directory for iOS app resources
    output_dir = Path(__file__).parent.parent / "tests/ios-harness/PocketTTSDemo/PocketTTSDemo/ReferenceAudio"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING REFERENCE AUDIO FOR iOS AB TESTING")
    print("Using ACTUAL TTS-generated speech (not synthetic data)")
    print("=" * 60)

    # Load the Python TTS model
    print(f"\nLoading Pocket TTS model...")
    model = TTSModel.load_model()
    print(f"  Sample rate: {model.sample_rate} Hz")

    # Get voice state
    print(f"\nGetting voice state for '{VOICE}'...")
    voice_state = model.get_state_for_audio_prompt(VOICE)

    # Generate manifest
    manifest = {
        "sample_rate": SAMPLE_RATE,
        "format": "Int16 PCM WAV (iOS compatible)",
        "voice": VOICE,
        "note": "Generated using actual Python TTS pipeline - REAL SPEECH",
        "phrases": []
    }

    for phrase in REFERENCE_PHRASES:
        phrase_id = phrase["id"]
        text = phrase["text"]

        print(f"\nGenerating '{phrase_id}': {text}")

        # Generate ACTUAL TTS audio
        audio = model.generate_audio(voice_state, text)
        audio_np = audio.numpy()

        print(f"  Samples: {len(audio_np)}")
        print(f"  Duration: {len(audio_np) / SAMPLE_RATE:.2f}s")
        print(f"  Max amplitude: {np.abs(audio_np).max():.4f}")

        # Save as Int16 WAV (iOS compatible)
        wav_path = output_dir / f"reference_{phrase_id}_int16.wav"
        save_wav_int16(audio_np, wav_path)
        print(f"  Saved: {wav_path.name}")

        # Add to manifest
        manifest["phrases"].append({
            "id": phrase_id,
            "text": text,
            "description": phrase["description"],
            "audio_file": f"reference_{phrase_id}_int16.wav",
            "samples": len(audio_np),
            "duration_sec": len(audio_np) / SAMPLE_RATE
        })

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("REFERENCE AUDIO GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.suffix in ['.wav', '.json']:
            print(f"  {f.name}")

    print("\nNext steps:")
    print("1. Verify files play correctly: afplay <file>.wav")
    print("2. Rebuild iOS app: ./scripts/build-ios.sh")
    print("3. Test in iOS simulator")

    return 0


if __name__ == "__main__":
    exit(main())
