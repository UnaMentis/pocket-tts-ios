#!/usr/bin/env python3
"""
Download Kyutai Pocket TTS model files for local use.

This script downloads the model files from HuggingFace and organizes them
in the expected directory structure for the Rust implementation.

Usage:
    python scripts/download-model.py
    python scripts/download-model.py --output-dir ./models/kyutai-pocket-ios
"""

import argparse
import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Run: pip install huggingface_hub")
    exit(1)


def download_model(output_dir: Path):
    """Download Pocket TTS model files from HuggingFace."""

    output_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(exist_ok=True)

    print(f"Downloading model to {output_dir}...")

    # Download main model (without voice cloning variant has the core files)
    print("\n1. Downloading main model weights...")
    model_file = hf_hub_download(
        repo_id="kyutai/pocket-tts-without-voice-cloning",
        filename="tts_b6369a24.safetensors",
    )
    # Copy to our directory with expected name
    shutil.copy(model_file, output_dir / "model.safetensors")
    print(f"   -> model.safetensors ({os.path.getsize(output_dir / 'model.safetensors') / 1024 / 1024:.1f} MB)")

    # Download tokenizer
    print("\n2. Downloading tokenizer...")
    tokenizer_file = hf_hub_download(
        repo_id="kyutai/pocket-tts-without-voice-cloning",
        filename="tokenizer.model",
    )
    shutil.copy(tokenizer_file, output_dir / "tokenizer.model")
    print(f"   -> tokenizer.model")

    # Download voice embeddings
    print("\n3. Downloading voice embeddings...")

    # Alba is in the main model repo
    alba_file = hf_hub_download(
        repo_id="kyutai/pocket-tts-without-voice-cloning",
        filename="embeddings/alba.safetensors",
    )
    shutil.copy(alba_file, voices_dir / "alba.safetensors")
    print("   -> alba.safetensors")

    # Additional voices from tts-voices repo
    additional_voices = [
        "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"
    ]

    for voice in additional_voices:
        try:
            voice_file = hf_hub_download(
                repo_id="kyutai/tts-voices",
                filename=f"{voice}.safetensors",
            )
            shutil.copy(voice_file, voices_dir / f"{voice}.safetensors")
            print(f"   -> {voice}.safetensors")
        except Exception as e:
            print(f"   -> {voice}.safetensors (not available: {e})")

    print(f"\n✓ Model downloaded successfully to {output_dir}")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"  ├── model.safetensors")
    print(f"  ├── tokenizer.model")
    print(f"  └── voices/")
    for v in sorted(voices_dir.iterdir()):
        print(f"      └── {v.name}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Kyutai Pocket TTS model for local use"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "kyutai-pocket-ios",
        help="Directory to save model files (default: ./models/kyutai-pocket-ios)"
    )

    args = parser.parse_args()

    if (args.output_dir / "model.safetensors").exists():
        print(f"Model already exists at {args.output_dir}")
        response = input("Re-download? [y/N] ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return

    download_model(args.output_dir)


if __name__ == "__main__":
    main()
