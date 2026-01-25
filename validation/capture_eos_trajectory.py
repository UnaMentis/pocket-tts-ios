#!/usr/bin/env python3
"""
Capture EOS logit trajectory from Python Pocket TTS for comparison with Rust.

This script logs the EOS logit at every generation step to help diagnose
EOS detection divergence between Rust and Python implementations.

Usage:
    python capture_eos_trajectory.py --text "Your test phrase here"
    python capture_eos_trajectory.py --text "Your test phrase here" --output eos_trajectory.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np


def hook_eos_logits(model, text: str, voice: str = "alba") -> Dict:
    """
    Generate audio while capturing EOS logits at every step.

    Returns dict with:
    - eos_logits: list of (step, eos_value) tuples
    - num_latents: total latent frames generated
    - audio_samples: number of audio samples produced
    """
    from pocket_tts import TTSModel

    # Track EOS logits
    eos_trajectory: List[tuple] = []
    step_counter = [0]  # Use list to allow modification in closure

    # Store original forward function
    original_out_eos_forward = None

    def hook_fn(module, input, output):
        """Capture EOS logit at each step."""
        eos_val = output.squeeze().item()
        step = step_counter[0]
        eos_trajectory.append((step, eos_val))

        # Log every 10 steps or near threshold
        if step % 10 == 0 or step == 0 or eos_val > -5.0:
            print(f"[EOS-TRAJ] step={step:3d}, eos_logit={eos_val:7.4f}, threshold=-4.0")

        step_counter[0] += 1
        return output

    # Load model
    print("Loading model...")
    tts_model = TTSModel.load_model()

    # Find the out_eos layer
    flow_lm = tts_model.flow_lm
    out_eos = flow_lm.out_eos

    # Register hook
    hook_handle = out_eos.register_forward_hook(hook_fn)

    try:
        # Get voice state
        print(f"Getting voice state for '{voice}'...")
        voice_state = tts_model.get_state_for_audio_prompt(voice)

        # Generate audio
        print(f"\nGenerating audio for: \"{text}\"")
        print("=" * 60)
        audio = tts_model.generate_audio(voice_state, text)
        print("=" * 60)

        audio_np = audio.numpy()

        # Summary
        print(f"\n[EOS-SUMMARY] Total steps: {len(eos_trajectory)}")
        if eos_trajectory:
            eos_values = [e[1] for e in eos_trajectory]
            print(f"[EOS-SUMMARY] min={min(eos_values):.4f}, max={max(eos_values):.4f}, mean={np.mean(eos_values):.4f}")

            # Find first step above threshold
            eos_steps = [e[0] for e in eos_trajectory if e[1] > -4.0]
            if eos_steps:
                print(f"[EOS-SUMMARY] First EOS trigger at step {eos_steps[0]}")

        print(f"\nAudio samples: {len(audio_np)}")
        print(f"Audio duration: {len(audio_np) / tts_model.sample_rate:.2f}s")

        return {
            "text": text,
            "voice": voice,
            "eos_trajectory": eos_trajectory,
            "num_latents": len(eos_trajectory),
            "audio_samples": len(audio_np),
            "eos_threshold": -4.0,
            "eos_min": float(min(e[1] for e in eos_trajectory)) if eos_trajectory else None,
            "eos_max": float(max(e[1] for e in eos_trajectory)) if eos_trajectory else None,
            "eos_mean": float(np.mean([e[1] for e in eos_trajectory])) if eos_trajectory else None,
        }
    finally:
        hook_handle.remove()


def main():
    parser = argparse.ArgumentParser(
        description="Capture EOS logit trajectory from Python Pocket TTS"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the Pocket TTS system.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="alba",
        help="Voice to use (default: alba)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for trajectory data"
    )
    parser.add_argument(
        "--long-test",
        action="store_true",
        help="Run multiple test phrases including longer ones"
    )

    args = parser.parse_args()

    if args.long_test:
        # Test multiple phrases of increasing length
        test_phrases = [
            "Hello, this is a test of the Pocket TTS system.",
            "The quick brown fox jumps over the lazy dog.",
            "The pharmaceutical company Pfizer and actor Arnold Schwarzenegger discussed mRNA vaccines at the café while listening to Tchaikovsky. Scientists believe this breakthrough will revolutionize medicine.",
            "Machine learning has transformed the way we approach complex problems. Neural networks, inspired by the human brain, can now recognize images, understand speech, and even generate creative content. The technology continues to evolve rapidly.",
        ]

        all_results = []
        for phrase in test_phrases:
            print(f"\n{'='*80}")
            print(f"Testing: \"{phrase[:60]}...\"" if len(phrase) > 60 else f"Testing: \"{phrase}\"")
            print(f"{'='*80}\n")

            result = hook_eos_logits(None, phrase, args.voice)
            all_results.append(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        result = hook_eos_logits(None, args.text, args.voice)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nTrajectory saved to {args.output}")


if __name__ == "__main__":
    main()
