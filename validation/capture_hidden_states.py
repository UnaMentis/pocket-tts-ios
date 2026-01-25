#!/usr/bin/env python3
"""
Capture hidden states at specific generation steps from Python Pocket TTS.
This helps diagnose EOS divergence between Rust and Python implementations.
"""

import torch
from pocket_tts import TTSModel

# Steps to capture (Python steps, Rust equivalent in comments)
# Python step = Rust step + 2 due to voice init offset
CAPTURE_STEPS = [38, 39, 40, 41, 42, 43, 44]  # Python steps (Rust 36, 37, 38, 39, 40, 41, 42)

def main():
    print("Loading model...")
    model = TTSModel.load_model()
    flow_lm = model.flow_lm

    # Storage for hidden states
    hidden_states = {}
    step_counter = [0]  # Use list to allow modification in closure

    def out_norm_hook(module, input, output):
        """Capture hidden state after out_norm (before EOS computation)."""
        step = step_counter[0]
        if step in CAPTURE_STEPS:
            # Get last position hidden state
            hidden = output[0, -1].float().detach().cpu()
            hidden_states[step] = hidden

            # Compute statistics
            h_np = hidden.numpy()
            print(f"\n[HIDDEN] Python step={step} (Rust step={step-2}):")
            print(f"  mean={h_np.mean():.4f}, std={h_np.std():.4f}")
            print(f"  min={h_np.min():.4f}, max={h_np.max():.4f}")
            print(f"  first 8: {h_np[:8].tolist()}")

            # Also compute what EOS would be
            with torch.no_grad():
                eos_logit = flow_lm.out_eos(hidden.unsqueeze(0).to(output.device))
                print(f"  EOS logit: {eos_logit.item():.4f}")

        step_counter[0] += 1
        return output

    # Register hook on out_norm
    hook_handle = flow_lm.out_norm.register_forward_hook(out_norm_hook)

    try:
        # Get voice state
        print("Getting voice state...")
        voice_state = model.get_state_for_audio_prompt("alba")

        # Generate audio
        text = "Hello, this is a test of the Pocket TTS system."
        print(f'\nGenerating audio for: "{text}"')
        print("=" * 60)
        audio = model.generate_audio(voice_state, text)
        print("=" * 60)

        print(f"\nTotal steps: {step_counter[0]}")
        print(f"Audio samples: {len(audio)}")

    finally:
        hook_handle.remove()


if __name__ == "__main__":
    main()
