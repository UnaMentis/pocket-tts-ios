#!/usr/bin/env python3
"""Dump all latents from Python generation for comparison with Rust."""

import torch
import numpy as np
from pathlib import Path

# Monkey-patch to use zeros
_orig_normal = torch.nn.init.normal_
_orig_trunc = torch.nn.init.trunc_normal_
def zeros_normal_(t, mean=0, std=1):
    if not t.requires_grad:
        with torch.no_grad():
            t.zero_()
    else:
        _orig_normal(t, mean, std)
    return t
def zeros_trunc_(t, mean=0, std=1, a=-2, b=2):
    if not t.requires_grad:
        with torch.no_grad():
            t.zero_()
    else:
        _orig_trunc(t, mean, std, a, b)
    return t
torch.nn.init.normal_ = zeros_normal_
torch.nn.init.trunc_normal_ = zeros_trunc_

from pocket_tts import TTSModel

ALL_LATENTS = []

def main():
    print("Loading model...")
    model = TTSModel.load_model()

    # Hook to capture all latents (FlowNet outputs)
    flow_net = model.flow_lm.flow_net
    call_count = [0]

    def flow_net_hook(module, args, output):
        call_count[0] += 1
        c, s, t, x = args
        latent = output.flatten().detach().cpu().numpy()
        ALL_LATENTS.append(latent.copy())

        if call_count[0] <= 5 or call_count[0] % 10 == 0:
            print(f"Latent {call_count[0]}: first 4 = {latent[:4].tolist()}")

    handle = flow_net.register_forward_hook(flow_net_hook)

    print("\nGetting voice state...")
    voice_state = model.get_state_for_audio_prompt("alba")

    text = "Hello, this is a test of the Pocket TTS system."
    print(f"\nGenerating audio for: '{text}'")

    try:
        audio = model.generate_audio(voice_state, text)
    finally:
        handle.remove()

    print(f"\nTotal latents captured: {len(ALL_LATENTS)}")

    # Save latents
    output_dir = Path(__file__).parent / "reference_zeros"
    latents_array = np.array(ALL_LATENTS)  # Shape: [num_latents, 32]
    np.save(output_dir / "python_latents.npy", latents_array)
    print(f"Saved latents to {output_dir / 'python_latents.npy'}")
    print(f"Latents shape: {latents_array.shape}")

    # Show first 5 latents for comparison
    print("\n=== First 5 latents (first 8 values each) ===")
    for i in range(min(5, len(ALL_LATENTS))):
        print(f"Latent {i+1}: {ALL_LATENTS[i][:8].tolist()}")

if __name__ == "__main__":
    main()
