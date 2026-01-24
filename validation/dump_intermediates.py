#!/usr/bin/env python3
"""
Dump intermediate tensors from Python Pocket TTS for comparison with Rust.

This script hooks into the model to capture intermediate values at key points:
1. Text embeddings (conditioner output)
2. Audio conditioning (speaker_proj output)
3. Transformer hidden states
4. FlowNet conditioning
5. Generated latents
6. Mimi decoder intermediates

Usage:
    python dump_intermediates.py --output-dir ./debug_outputs
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from pocket_tts import TTSModel

# Global dict to store intermediate tensors
INTERMEDIATES = {}


def hook_factory(name):
    """Create a forward hook that saves the output."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        INTERMEDIATES[name] = output.detach().cpu().numpy()
    return hook


def main():
    parser = argparse.ArgumentParser(description="Dump intermediate tensors from Pocket TTS")
    parser.add_argument("--output-dir", type=Path, default=Path("debug_outputs"))
    parser.add_argument("--text", type=str, default="Hello, this is a test of the Pocket TTS system.")
    parser.add_argument("--voice", type=str, default="alba")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = TTSModel.load_model()
    print(f"Sample rate: {model.sample_rate}")

    # Get voice state
    print(f"Getting voice state for '{args.voice}'...")
    voice_state = model.get_state_for_audio_prompt(args.voice)

    # Hook key modules to capture intermediates
    print("Setting up hooks...")

    # Hook conditioner (text embeddings)
    if hasattr(model.flow_lm, 'conditioner'):
        model.flow_lm.conditioner.register_forward_hook(hook_factory('conditioner_output'))

    # Hook flow_net (latent generation)
    if hasattr(model.flow_lm, 'flow_net'):
        model.flow_lm.flow_net.register_forward_hook(hook_factory('flownet_output'))

    # Hook out_norm (final hidden states before FlowNet)
    if hasattr(model.flow_lm, 'out_norm'):
        model.flow_lm.out_norm.register_forward_hook(hook_factory('out_norm'))

    # Hook input_linear (latent -> hidden projection)
    if hasattr(model.flow_lm, 'input_linear'):
        model.flow_lm.input_linear.register_forward_hook(hook_factory('input_linear'))

    # Generate with monitoring
    print(f"\nGenerating audio for: '{args.text}'")

    # Tokenize - use the conditioner's tokenization method
    tokenizer = model.flow_lm.conditioner.tokenizer
    tokens = tokenizer.sp.encode(args.text)
    print(f"Tokens ({len(tokens)}): {tokens}")

    # Save token info
    token_info = {
        "text": args.text,
        "tokens": tokens,
        "num_tokens": len(tokens),
    }

    # Generate audio
    audio = model.generate_audio(voice_state, args.text)
    audio_np = audio.numpy()

    print(f"\nGenerated {len(audio_np)} samples ({len(audio_np) / model.sample_rate:.2f}s)")
    print(f"Max amplitude: {np.max(np.abs(audio_np)):.4f}")

    # Save all intermediates
    print(f"\nSaving intermediates to {args.output_dir}...")

    # Save token info
    with open(args.output_dir / "token_info.json", "w") as f:
        json.dump(token_info, f, indent=2)

    # Save audio
    np.save(args.output_dir / "audio.npy", audio_np)

    # Save captured intermediates
    for name, tensor in INTERMEDIATES.items():
        print(f"  {name}: shape={tensor.shape}, mean={np.mean(tensor):.6f}, std={np.std(tensor):.6f}")
        np.save(args.output_dir / f"{name}.npy", tensor)

    # Also dump model state info
    print("\nModel component info:")
    print(f"  flow_lm.ldim (latent dim): {model.flow_lm.ldim}")
    print(f"  flow_lm.dim (hidden dim): {model.flow_lm.dim}")

    # Dump bos_emb and emb_mean/emb_std
    bos = model.flow_lm.bos_emb.detach().cpu().numpy()
    emb_mean = model.flow_lm.emb_mean.detach().cpu().numpy()
    emb_std = model.flow_lm.emb_std.detach().cpu().numpy()

    print(f"  bos_emb: shape={bos.shape}, first 8: {bos[:8]}")
    print(f"  emb_mean: shape={emb_mean.shape}, first 8: {emb_mean[:8]}")
    print(f"  emb_std: shape={emb_std.shape}, first 8: {emb_std[:8]}")

    np.save(args.output_dir / "bos_emb.npy", bos)
    np.save(args.output_dir / "emb_mean.npy", emb_mean)
    np.save(args.output_dir / "emb_std.npy", emb_std)

    # Dump audio conditioning from voice state
    if 'audio_conditioning' in voice_state:
        audio_cond = voice_state['audio_conditioning']
        if isinstance(audio_cond, torch.Tensor):
            audio_cond = audio_cond.detach().cpu().numpy()
        print(f"  audio_conditioning: shape={audio_cond.shape}, mean={np.mean(audio_cond):.6f}")
        np.save(args.output_dir / "audio_conditioning.npy", audio_cond)

    print(f"\nDone! Outputs saved to {args.output_dir}")
    print("\nCompare with Rust outputs to identify divergence point.")


if __name__ == "__main__":
    main()
