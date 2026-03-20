#!/usr/bin/env python3
"""
Capture FlowNet intermediates with seed=42, matching the reference run.
This gives us Python FlowNet values that correspond to the same noise as Rust.
"""

import torch
import numpy as np
from pocket_tts import TTSModel

def get_vals(tensor):
    if tensor.dim() == 1:
        return tensor.float().detach().cpu().numpy()
    elif tensor.dim() == 2:
        return tensor[0].float().detach().cpu().numpy()
    return tensor[0, -1].float().detach().cpu().numpy()

def main():
    model = TTSModel.load_model()
    flow_net = model.flow_lm.flow_net

    # Process voice FIRST (before seed/hooks)
    voice_state = model.get_state_for_audio_prompt("alba")

    # Set seed to match reference run
    torch.manual_seed(42)

    captured = {}
    call_count = [0]

    # Track first latent generation FlowNet call
    # Skip text processing calls by counting
    # We need to determine which call corresponds to latent step 0
    # During generate_audio, FlowNet is called for text+latent steps

    # Use input_proj to identify calls and capture all FlowNet calls
    all_calls = []

    def input_proj_hook(module, input, output):
        noise = get_vals(input[0])
        proj = get_vals(output)
        all_calls.append({
            'noise': noise.copy(),
            'input_proj': proj.copy(),
        })

    def cond_embed_hook(module, input, output):
        if len(all_calls) > 0 and 'cond_embed' not in all_calls[-1]:
            # cond_embed fires during same forward pass
            pass
        vals = get_vals(output)
        if len(all_calls) > 0:
            all_calls[-1]['cond_embed'] = vals.copy()

    resblock_idx = [0]
    def make_resblock_hook(idx):
        def hook(module, input, output):
            if len(all_calls) > 0:
                all_calls[-1][f'resblock_{idx}'] = get_vals(output).copy()
        return hook

    def final_layer_hook(module, input, output):
        if len(all_calls) > 0:
            all_calls[-1]['velocity'] = get_vals(output).copy()

    hooks = []
    hooks.append(flow_net.input_proj.register_forward_hook(input_proj_hook))
    hooks.append(flow_net.cond_embed.register_forward_hook(cond_embed_hook))
    for i, block in enumerate(flow_net.res_blocks):
        hooks.append(block.register_forward_hook(make_resblock_hook(i)))
    hooks.append(flow_net.final_layer.register_forward_hook(final_layer_hook))

    try:
        audio = model.generate_audio(voice_state,
            "Hello, this is a test of the Pocket TTS system.")
        print(f"Audio samples: {len(audio)}")
        print(f"Total FlowNet calls: {len(all_calls)}")

        # Find which call uses noise matching reference step 0
        ref_noise_0 = np.load('validation/reference_outputs/noise/phrase_00_noise_step_000.npy').flatten()
        print(f"\nRef noise step 0 first 4: {ref_noise_0[:4].tolist()}")

        for i, call in enumerate(all_calls):
            if np.allclose(call['noise'], ref_noise_0, atol=1e-5):
                print(f"\n*** FlowNet call {i} matches reference noise step 0 ***")

                for key in sorted(call.keys()):
                    arr = call[key]
                    if len(arr) <= 32:
                        print(f"  {key}: ALL {len(arr)} = {arr.tolist()}")
                    else:
                        print(f"  {key}: mean={arr.mean():.6f}, std={arr.std():.6f}, first 8={arr[:8].tolist()}")

                # Save this call's data
                np.savez('validation/python_flownet_step0_seeded.npz', **call)
                print(f"\n  Saved to python_flownet_step0_seeded.npz")
                break
        else:
            print("\nWARNING: No FlowNet call matched reference noise!")
            # Print all calls' noise for debugging
            for i, call in enumerate(all_calls[:5]):
                print(f"  call {i} noise first 4: {call['noise'][:4].tolist()}")

    finally:
        for h in hooks:
            h.remove()

if __name__ == "__main__":
    main()
