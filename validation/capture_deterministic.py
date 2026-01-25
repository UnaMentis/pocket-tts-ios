#!/usr/bin/env python3
"""
Capture latents and hidden states with deterministic (zero) noise.
This matches Rust's DEBUG mode which uses zeros instead of random noise.
"""

import torch
import numpy as np
from pocket_tts import TTSModel

def main():
    print("=" * 70)
    print("DETERMINISTIC COMPARISON (zeros instead of random noise)")
    print("=" * 70)

    print("\nLoading model...")
    model = TTSModel.load_model()
    flow_lm = model.flow_lm

    # Storage
    latent_outputs = {}
    step_counter = [0]

    # Patch the flow_lm.forward to use zeros instead of random noise
    original_forward = flow_lm.forward

    def patched_forward(
        sequence, text_embeddings, model_state, lsd_decode_steps, temp, noise_clamp, eos_threshold
    ):
        # NaN values signal a BOS position.
        sequence = torch.where(torch.isnan(sequence), flow_lm.bos_emb, sequence)
        input_ = flow_lm.input_linear(sequence)

        transformer_out = flow_lm.backbone(input_, text_embeddings, sequence, model_state=model_state)
        transformer_out = transformer_out.to(torch.float32)
        assert lsd_decode_steps > 0

        transformer_out = transformer_out[:, -1]
        out_eos = flow_lm.out_eos(transformer_out) > eos_threshold

        # USE ZEROS instead of random noise!
        noise_shape = transformer_out.shape[:-1] + (flow_lm.ldim,)
        noise = torch.zeros(noise_shape, dtype=transformer_out.dtype, device=transformer_out.device)

        from functools import partial
        from pocket_tts.models.flow_lm import lsd_decode
        conditioned_flow = partial(flow_lm.flow_net, transformer_out)
        return lsd_decode(conditioned_flow, noise, lsd_decode_steps), out_eos

    flow_lm.forward = patched_forward

    # Hook to capture latents
    def input_linear_hook(module, input, output):
        step = step_counter[0]
        if step <= 5 or step == 37 or step == 38:
            latent = input[0][0, -1].float().detach().cpu().numpy()
            latent_outputs[f'step{step}'] = latent
            print(f"\n[LATENT] Python step={step} (Rust step={step-2}):")
            print(f"  mean={latent.mean():.6f}, std={latent.std():.6f}")
            print(f"  first 8: {latent[:8].tolist()}")
        step_counter[0] += 1

    hook_handle = flow_lm.input_linear.register_forward_hook(input_linear_hook)

    try:
        # Get voice state
        print("\nGetting voice state...")
        voice_state = model.get_state_for_audio_prompt("alba")

        # Generate audio
        text = "Hello, this is a test of the Pocket TTS system."
        print(f'\nGenerating audio for: "{text}"')
        print("=" * 70)
        audio = model.generate_audio(voice_state, text)
        print("=" * 70)

        print(f"\nTotal steps: {step_counter[0]}")
        print(f"Audio samples: {len(audio)}")

        # Save for comparison
        if latent_outputs:
            np.savez('/tmp/python_deterministic_latents.npz', **latent_outputs)
            print("\nSaved to /tmp/python_deterministic_latents.npz")

    finally:
        hook_handle.remove()
        flow_lm.forward = original_forward


if __name__ == "__main__":
    main()
