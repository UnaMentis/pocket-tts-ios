#!/usr/bin/env python3
"""Dump Layer 0 intermediate values from Python Pocket TTS."""

import torch
import numpy as np
from pocket_tts import TTSModel

def main():
    print("Loading model...")
    model = TTSModel.load_model()

    # Get transformer layer 0
    layer0 = model.flow_lm.transformer.layers[0]

    # We'll manually trace through the layer to capture intermediates
    call_count = [0]
    CAPTURED = {}

    # Hook norm1 output
    def norm1_hook(module, input, output):
        call_count[0] += 1
        seq_len = output.shape[1]
        # Text phase has 7 tokens, voice has 125
        if seq_len == 7 and 'norm1' not in CAPTURED:
            print(f"\n[Python-L0] norm1 output shape: {output.shape} (TEXT PHASE)")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-L0] norm1 first 8: {vals}")
            CAPTURED['norm1'] = vals

    # Hook self_attn output
    def attn_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 7 and 'attn' not in CAPTURED:
            print(f"[Python-L0] attn output shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-L0] attn output first 8: {vals}")
            CAPTURED['attn'] = vals

    # Hook norm2 output
    def norm2_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 7 and 'norm2' not in CAPTURED:
            print(f"[Python-L0] norm2 output shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-L0] norm2 first 8: {vals}")
            CAPTURED['norm2'] = vals

    # Hook linear1 and linear2 (MLP)
    def linear1_hook(module, input, output):
        # linear1 outputs to intermediate_size (4096), check input shape via input[0]
        if input[0].shape[1] == 7 and 'linear1' not in CAPTURED:
            print(f"[Python-L0] linear1 (MLP hidden) shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-L0] linear1 first 8: {vals}")
            CAPTURED['linear1'] = vals

    def linear2_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 7 and 'mlp' not in CAPTURED:
            print(f"[Python-L0] linear2 (MLP out) shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-L0] linear2 first 8: {vals}")
            CAPTURED['mlp'] = vals

    # Hook full layer output
    def layer_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 7 and 'layer_output' not in CAPTURED:
            print(f"\n[Python-L0] LAYER OUTPUT shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-L0] LAYER OUTPUT first 8: {vals}")
            CAPTURED['layer_output'] = vals

    # Register hooks
    h1 = layer0.norm1.register_forward_hook(norm1_hook)
    h2 = layer0.self_attn.register_forward_hook(attn_hook)
    h3 = layer0.norm2.register_forward_hook(norm2_hook)
    h4 = layer0.linear1.register_forward_hook(linear1_hook)
    h5 = layer0.linear2.register_forward_hook(linear2_hook)
    h6 = layer0.register_forward_hook(layer_hook)

    try:
        print("\nGetting voice state...")
        voice_state = model.get_state_for_audio_prompt("alba")

        print("\n" + "=" * 60)
        print("Generating: 'Hello, this is a test.'")
        print("=" * 60)

        audio = model.generate_audio(voice_state, "Hello, this is a test.")
        print(f"\nGenerated {len(audio)} samples")
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()
        h5.remove()
        h6.remove()

    print("\n" + "=" * 60)
    print("SUMMARY - Layer 0 Text Phase:")
    for key, val in CAPTURED.items():
        print(f"  {key}: {val}")

if __name__ == "__main__":
    main()
