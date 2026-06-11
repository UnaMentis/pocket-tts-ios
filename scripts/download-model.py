#!/usr/bin/env python3
"""
Download Kyutai Pocket TTS model files for local use.

This script downloads the model files from HuggingFace and organizes them
in the expected directory structure for the Rust implementation.

Two model generations are supported:

  v1 (default)  english_2026-01 weights from the public repo
                kyutai/pocket-tts-without-voice-cloning. No login needed.
  v2            english_2026-04 weights from the GATED repo kyutai/pocket-tts.
                Requires accepting the gate at
                https://huggingface.co/kyutai/pocket-tts and an HF token
                (`huggingface-cli login` or the HF_TOKEN env var).
                v2 voice files are precomputed KV-state voices — a different
                format from v1 embeddings; the Rust engine detects either.

Usage:
    python scripts/download-model.py                  # v1 -> ./models/kyutai-pocket-ios
    python scripts/download-model.py --model v2       # v2 -> ./kyutai-pocket-ios-en2026-04
    python scripts/download-model.py --output-dir DIR
"""

import argparse
import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Run: pip install huggingface_hub")
    exit(1)

VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]

V2_REPO = "kyutai/pocket-tts"
V2_LANG = "english_2026-04"

GATE_HELP = f"""
ERROR: could not download from the gated repo '{V2_REPO}'.

The v2 models are gated on HuggingFace. To get access:
  1. Visit https://huggingface.co/{V2_REPO} and accept the terms (instant, gated=auto).
  2. Authenticate locally, either:
       huggingface-cli login
     or set a token for this run:
       HF_TOKEN=hf_... python scripts/download-model.py --model v2
  3. Re-run this command.

Note: per Kyutai's gate, these weights are NOT redistributed in this
project's release artifacts — every user downloads them directly.
"""


def download_v1(output_dir: Path):
    """Download Pocket TTS v1 (english_2026-01) from the public repo."""

    output_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(exist_ok=True)

    print(f"Downloading v1 model to {output_dir}...")

    print("\n1. Downloading main model weights...")
    model_file = hf_hub_download(
        repo_id="kyutai/pocket-tts-without-voice-cloning",
        filename="tts_b6369a24.safetensors",
    )
    shutil.copy(model_file, output_dir / "model.safetensors")
    print(f"   -> model.safetensors ({os.path.getsize(output_dir / 'model.safetensors') / 1024 / 1024:.1f} MB)")

    print("\n2. Downloading tokenizer...")
    tokenizer_file = hf_hub_download(
        repo_id="kyutai/pocket-tts-without-voice-cloning",
        filename="tokenizer.model",
    )
    shutil.copy(tokenizer_file, output_dir / "tokenizer.model")
    print("   -> tokenizer.model")

    print("\n3. Downloading voice embeddings...")
    alba_file = hf_hub_download(
        repo_id="kyutai/pocket-tts-without-voice-cloning",
        filename="embeddings/alba.safetensors",
    )
    shutil.copy(alba_file, voices_dir / "alba.safetensors")
    print("   -> alba.safetensors")

    for voice in VOICES[1:]:
        try:
            voice_file = hf_hub_download(
                repo_id="kyutai/tts-voices",
                filename=f"{voice}.safetensors",
            )
            shutil.copy(voice_file, voices_dir / f"{voice}.safetensors")
            print(f"   -> {voice}.safetensors")
        except Exception as e:
            print(f"   -> {voice}.safetensors (not available: {e})")

    print_summary(output_dir, voices_dir)


def download_v2(output_dir: Path):
    """Download Pocket TTS v2 (english_2026-04) from the gated repo."""

    token = os.environ.get("HF_TOKEN")  # falls back to the local HF login if unset

    output_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(exist_ok=True)

    print(f"Downloading v2 ({V2_LANG}) model to {output_dir}...")

    print("\n1. Downloading main model weights...")
    try:
        model_file = hf_hub_download(
            repo_id=V2_REPO,
            filename=f"languages/{V2_LANG}/model.safetensors",
            token=token,
        )
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("401", "403", "gated", "Gated", "restricted", "Access")):
            print(GATE_HELP)
            exit(1)
        raise
    shutil.copy(model_file, output_dir / "model.safetensors")
    print(f"   -> model.safetensors ({os.path.getsize(output_dir / 'model.safetensors') / 1024 / 1024:.1f} MB)")

    print("\n2. Downloading tokenizer...")
    tokenizer_file = hf_hub_download(
        repo_id=V2_REPO,
        filename=f"languages/{V2_LANG}/tokenizer.model",
        token=token,
    )
    shutil.copy(tokenizer_file, output_dir / "tokenizer.model")
    print("   -> tokenizer.model")

    print("\n3. Downloading voices (v2 KV-state format)...")
    for voice in VOICES:
        try:
            voice_file = hf_hub_download(
                repo_id=V2_REPO,
                filename=f"languages/{V2_LANG}/embeddings/{voice}.safetensors",
                token=token,
            )
            shutil.copy(voice_file, voices_dir / f"{voice}.safetensors")
            print(f"   -> {voice}.safetensors")
        except Exception as e:
            print(f"   -> {voice}.safetensors (not available: {e})")

    print_summary(output_dir, voices_dir)


def print_summary(output_dir: Path, voices_dir: Path):
    print(f"\n✓ Model downloaded successfully to {output_dir}")
    print("\nDirectory structure:")
    print(f"  {output_dir}/")
    print("  ├── model.safetensors")
    print("  ├── tokenizer.model")
    print("  └── voices/")
    for v in sorted(voices_dir.iterdir()):
        print(f"      └── {v.name}")


def main():
    repo_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Download Kyutai Pocket TTS model for local use")
    parser.add_argument(
        "--model",
        choices=["v1", "v2"],
        default="v1",
        help="Model generation: v1 = english_2026-01 (public), v2 = english_2026-04 (gated, needs HF token)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save model files (default: ./models/kyutai-pocket-ios for v1, "
        "./kyutai-pocket-ios-en2026-04 for v2)",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            repo_root / "models" / "kyutai-pocket-ios" if args.model == "v1" else repo_root / "kyutai-pocket-ios-en2026-04"
        )

    if (output_dir / "model.safetensors").exists():
        print(f"Model already exists at {output_dir}")
        response = input("Re-download? [y/N] ").strip().lower()
        if response != "y":
            print("Skipping download.")
            return

    if args.model == "v1":
        download_v1(output_dir)
    else:
        download_v2(output_dir)


if __name__ == "__main__":
    main()
