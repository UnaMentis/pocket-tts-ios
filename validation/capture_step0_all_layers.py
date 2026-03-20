#!/usr/bin/env python3
"""
Capture hidden states after EACH of 6 transformer layers at step 0 (first latent generation).
Also captures input to layer 0 (after input_linear) and final output (after out_norm).
Saves all tensors to npz for comparison with Rust.

Python step 2 = Rust step 0 (offset by 2: voice init + text init).
"""

import torch
import numpy as np
from pocket_tts import TTSModel

# Python step 2 = Rust step 0
# voice phase = step 0 (125 tokens), text phase = step 1 (7 tokens), first latent = step 2
# We detect first latent by seq_len == 1 and track which call it is.

def main():
    print("=" * 70)
    print("ALL-LAYER HIDDEN STATE CAPTURE AT STEP 0 (first latent)")
    print("=" * 70)

    print("\nLoading model...")
    model = TTSModel.load_model()
    flow_lm = model.flow_lm
    transformer = flow_lm.transformer
    layers = transformer.layers

    print(f"Found {len(layers)} transformer layers")

    # Storage
    captured = {}
    # Track out_norm calls: voice(0), text(1), step0(2), step1(3), ...
    out_norm_calls = [0]
    input_linear_calls = [0]
    # Track per-layer calls to know which step we're in
    # Each step processes all 6 layers, so layer hook call / 6 = step
    layer_call_counts = [0]

    TARGET_OUTNORM_STEP = 2  # Python step 2 = Rust step 0

    def input_linear_hook(module, input, output):
        step = input_linear_calls[0]
        input_linear_calls[0] += 1
        # Guard against unexpected shapes during voice/text processing
        if output.dim() < 2 or output.shape[1] == 0:
            return
        # We want seq_len=1 calls (latent generation steps), first one is step 0
        if output.shape[1] == 1 and 'input_l0' not in captured:
            hidden = output[0, -1].float().detach().cpu().numpy()
            captured['input_l0'] = hidden
            print(f"\n[INPUT-L0] Python input_linear call={step} (first seq_len=1):")
            print(f"  Shape: {hidden.shape}")
            print(f"  mean={hidden.mean():.6f}, std={hidden.std():.6f}")
            print(f"  first 8: {hidden[:8].tolist()}")

    # Track per-layer seq_len=1 calls (first one per layer = step 0)
    layer_seq1_seen = [False] * len(layers)

    def make_layer_hook(layer_idx):
        def hook(module, input, output):
            # Only capture first seq_len=1 call per layer (= step 0)
            if output.dim() < 2 or output.shape[1] != 1:
                return
            if layer_seq1_seen[layer_idx]:
                return
            layer_seq1_seen[layer_idx] = True

            hidden = output[0, -1].float().detach().cpu().numpy()
            captured[f'layer_{layer_idx}'] = hidden
            print(f"\n[Layer {layer_idx}] Step 0:")
            print(f"  mean={hidden.mean():.6f}, std={hidden.std():.6f}")
            print(f"  min={hidden.min():.6f}, max={hidden.max():.6f}")
            print(f"  first 8: {hidden[:8].tolist()}")

            # Also capture input to layer 0
            if layer_idx == 0:
                inp = input[0][0, -1].float().detach().cpu().numpy()
                captured['input_to_l0'] = inp
                print(f"\n[INPUT to Layer 0] Step 0:")
                print(f"  mean={inp.mean():.6f}, std={inp.std():.6f}")
                print(f"  first 8: {inp[:8].tolist()}")

        return hook

    out_norm_seq1_seen = [False]

    def out_norm_hook(module, input, output):
        # Capture first seq_len=1 out_norm call = step 0 final hidden
        if output.dim() < 2 or output.shape[1] != 1:
            return
        if out_norm_seq1_seen[0]:
            return
        out_norm_seq1_seen[0] = True
        hidden = output[0, -1].float().detach().cpu().numpy()
        captured['final'] = hidden
        print(f"\n[FINAL after out_norm] Step 0:")
        print(f"  mean={hidden.mean():.6f}, std={hidden.std():.6f}")
        print(f"  min={hidden.min():.6f}, max={hidden.max():.6f}")
        print(f"  first 8: {hidden[:8].tolist()}")

    # Register hooks
    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_layer_hook(i)))
    hooks.append(flow_lm.out_norm.register_forward_hook(out_norm_hook))
    hooks.append(flow_lm.input_linear.register_forward_hook(input_linear_hook))

    try:
        print("\nGetting voice state...")
        voice_state = model.get_state_for_audio_prompt("alba")

        text = "Hello, this is a test of the Pocket TTS system."
        print(f'\nGenerating audio for: "{text}"')
        print("=" * 70)
        audio = model.generate_audio(voice_state, text)
        print("=" * 70)

        print(f"\nTotal out_norm calls: {out_norm_calls[0]}")
        print(f"Total layer calls: {layer_call_counts[0]}")
        print(f"Audio samples: {len(audio)}")

        if captured:
            print("\n" + "=" * 70)
            print("SUMMARY - All layers at Step 0 (first latent generation)")
            print("=" * 70)

            for key in sorted(captured.keys()):
                arr = captured[key]
                print(f"\n{key}:")
                print(f"  Shape: {arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}")
                print(f"  first 8: {[f'{v:.6f}' for v in arr[:8]]}")

            # Save for comparison
            outpath = 'validation/python_step0_all_layers.npz'
            np.savez(outpath, **captured)
            print(f"\nSaved {len(captured)} arrays to {outpath}")

    finally:
        for h in hooks:
            h.remove()


if __name__ == "__main__":
    main()
