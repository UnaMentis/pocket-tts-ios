#!/usr/bin/env python3
"""
Capture hidden states after EACH transformer layer at step 36 (Python step 38).
This diagnoses where divergence first appears between Rust and Python.

Based on Research Advisor recommendation #1:
> Add logging in Python to dump hidden state after EACH of the 6 transformer layers at step 36

Python step 38 = Rust step 36 (offset of 2 due to voice init)
"""

import torch
import numpy as np
from pocket_tts import TTSModel

# Target step: Python 38 = Rust 36 (where divergence begins)
TARGET_PYTHON_STEP = 38

# Also capture steps 35 and 36 for latent comparison
# Python steps 2-7 = Rust steps 0-5 (early steps)
# Python steps 37-38 = Rust steps 35-36 (divergence area)
LATENT_CAPTURE_STEPS = [2, 3, 4, 5, 6, 7, 37, 38]

def main():
    print("=" * 70)
    print("LAYER-BY-LAYER HIDDEN STATE CAPTURE AT STEP 36")
    print("=" * 70)

    print("\nLoading model...")
    model = TTSModel.load_model()
    flow_lm = model.flow_lm

    # Access the transformer layers
    transformer = flow_lm.transformer
    layers = transformer.layers  # nn.ModuleList of 6 StreamingTransformerLayer

    print(f"Found {len(layers)} transformer layers")

    # Storage for per-layer hidden states
    layer_outputs = {}
    step_counter = [0]  # Mutable counter for hook

    # We need to track which call to out_norm corresponds to step 38
    # The out_norm is called once per forward pass through the transformer
    out_norm_call_counter = [0]
    input_linear_call_counter = [0]

    def input_linear_hook(module, input, output):
        """Capture the raw latent (input to input_linear) at target steps."""
        step = input_linear_call_counter[0]
        if step in LATENT_CAPTURE_STEPS:
            # Input is the raw latent [B, 1, 32]
            latent = input[0][0, -1].float().detach().cpu().numpy()
            layer_outputs[f'latent_step{step}'] = latent

            print(f"\n[LATENT] Python step={step} (Rust step={step-2}):")
            print(f"  Shape: {latent.shape}")
            print(f"  mean={latent.mean():.6f}, std={latent.std():.6f}")
            print(f"  first 8: {latent[:8].tolist()}")

        input_linear_call_counter[0] += 1

    def make_layer_hook(layer_idx):
        """Create a hook for a specific layer."""
        def hook(module, input, output):
            step = out_norm_call_counter[0]
            if step == TARGET_PYTHON_STEP:
                # Capture INPUT to layer 0
                if layer_idx == 0:
                    inp = input[0][0, -1].float().detach().cpu().numpy()
                    layer_outputs['input_l0'] = inp
                    print(f"\n[INPUT-L0] Python step={step} (Rust step={step-2}):")
                    print(f"  mean={inp.mean():.6f}, std={inp.std():.6f}")
                    print(f"  first 8: {inp[:8].tolist()}")

                # Get last position hidden state (output)
                hidden = output[0, -1].float().detach().cpu().numpy()
                layer_outputs[layer_idx] = hidden

                print(f"\n[Layer {layer_idx}] Python step={step} (Rust step={step-2}):")
                print(f"  mean={hidden.mean():.6f}, std={hidden.std():.6f}")
                print(f"  min={hidden.min():.6f}, max={hidden.max():.6f}")
                print(f"  first 8: {hidden[:8].tolist()}")

        return hook

    def out_norm_hook(module, input, output):
        """Track which step we're on via out_norm calls."""
        step = out_norm_call_counter[0]

        if step == TARGET_PYTHON_STEP:
            # Capture final hidden state after all layers + out_norm
            hidden = output[0, -1].float().detach().cpu().numpy()
            layer_outputs['final'] = hidden

            print(f"\n[FINAL after out_norm] Python step={step} (Rust step={step-2}):")
            print(f"  mean={hidden.mean():.6f}, std={hidden.std():.6f}")
            print(f"  min={hidden.min():.6f}, max={hidden.max():.6f}")
            print(f"  first 8: {hidden[:8].tolist()}")

            # Compute EOS logit
            with torch.no_grad():
                hidden_tensor = torch.from_numpy(hidden).unsqueeze(0).to(output.device)
                eos_logit = flow_lm.out_eos(hidden_tensor)
                print(f"  EOS logit: {eos_logit.item():.6f}")

        out_norm_call_counter[0] += 1

    # Register hooks on each transformer layer
    layer_hooks = []
    for i, layer in enumerate(layers):
        hook = layer.register_forward_hook(make_layer_hook(i))
        layer_hooks.append(hook)

    # Register hook on out_norm
    out_norm_hook_handle = flow_lm.out_norm.register_forward_hook(out_norm_hook)

    # Register hook on input_linear to capture raw latents
    input_linear_hook_handle = flow_lm.input_linear.register_forward_hook(input_linear_hook)

    try:
        # Get voice state
        print("\nGetting voice state...")
        voice_state = model.get_state_for_audio_prompt("alba")

        # Generate audio with the test phrase
        text = "Hello, this is a test of the Pocket TTS system."
        print(f'\nGenerating audio for: "{text}"')
        print("=" * 70)
        audio = model.generate_audio(voice_state, text)
        print("=" * 70)

        print(f"\nTotal out_norm calls (steps): {out_norm_call_counter[0]}")
        print(f"Audio samples: {len(audio)}")

        # Save layer outputs for comparison
        if layer_outputs:
            print("\n" + "=" * 70)
            print("SUMMARY - All layer outputs at step 36 (Python 38)")
            print("=" * 70)

            for key in sorted(layer_outputs.keys(), key=lambda x: (isinstance(x, str), x)):
                hidden = layer_outputs[key]
                label = f"Layer {key}" if isinstance(key, int) else key.upper()
                print(f"\n{label}:")
                print(f"  Shape: {hidden.shape}")
                print(f"  mean={hidden.mean():.6f}, std={hidden.std():.6f}")
                print(f"  first 8: {[f'{v:.6f}' for v in hidden[:8]]}")

            # Save to file for Rust comparison
            np.savez(
                'validation/python_layer_outputs_step36.npz',
                **{f'layer_{k}' if isinstance(k, int) else k: v for k, v in layer_outputs.items()}
            )
            print("\nSaved to validation/python_layer_outputs_step36.npz")

    finally:
        # Remove all hooks
        for hook in layer_hooks:
            hook.remove()
        out_norm_hook_handle.remove()
        input_linear_hook_handle.remove()


if __name__ == "__main__":
    main()
