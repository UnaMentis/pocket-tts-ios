#!/usr/bin/env python3
"""
Capture FlowNet intermediate values at the first LATENT GENERATION step.
Identifies the correct call by matching noise tensor against captured reference.
"""

import numpy as np
from pocket_tts import TTSModel

def get_last_pos(tensor):
    """Extract last position from tensor, handling 2D and 3D."""
    if tensor.dim() == 2:
        return tensor[0].float().detach().cpu().numpy()
    elif tensor.dim() == 3:
        return tensor[0, -1].float().detach().cpu().numpy()
    return tensor.flatten().float().detach().cpu().numpy()

def main():
    print("=" * 70)
    print("FLOWNET INTERMEDIATE CAPTURE — MATCHING BY NOISE TENSOR")
    print("=" * 70)

    # Load the reference noise for step 0
    ref_noise = np.load('validation/reference_outputs/noise/phrase_00_noise_step_000.npy').flatten()
    print(f"Reference noise step 0 first 8: {ref_noise[:8].tolist()}")

    model = TTSModel.load_model()
    flow_lm = model.flow_lm
    flow_net = flow_lm.flow_net

    captured = {}
    found_target = [False]
    call_count = [0]

    # We'll use input_proj hook to detect when noise matches, then capture everything
    # on that same forward pass using a flag
    capture_active = [False]

    def input_proj_hook(module, input, output):
        call_count[0] += 1
        inp = get_last_pos(input[0])
        # Check if this noise matches our reference
        if not found_target[0] and np.allclose(inp, ref_noise[:len(inp)], atol=0.01):
            found_target[0] = True
            capture_active[0] = True
            captured['noise_input'] = inp
            captured['input_proj'] = get_last_pos(output)
            print(f"\n*** FOUND target FlowNet call at call #{call_count[0]} ***")
            print(f"[input_proj] noise first 8: {inp[:8].tolist()}")
            print(f"[input_proj] output first 8: {captured['input_proj'][:8].tolist()}")

    def cond_embed_hook(module, input, output):
        if capture_active[0] and 'cond_embed' not in captured:
            vals = get_last_pos(output)
            captured['cond_embed'] = vals
            print(f"\n[cond_embed] mean={vals.mean():.6f}, std={vals.std():.6f}")
            print(f"  first 8: {vals[:8].tolist()}")

    def time_embed_0_hook(module, input, output):
        if capture_active[0] and 'time_embed_s' not in captured:
            vals = get_last_pos(output)
            captured['time_embed_s'] = vals
            print(f"\n[time_embed_0] first 8: {vals[:8].tolist()}")

    def time_embed_1_hook(module, input, output):
        if capture_active[0] and 'time_embed_t' not in captured:
            vals = get_last_pos(output)
            captured['time_embed_t'] = vals
            print(f"\n[time_embed_1] first 8: {vals[:8].tolist()}")

    def make_resblock_hook(idx):
        def hook(module, input, output):
            key = f'resblock_{idx}'
            if capture_active[0] and key not in captured:
                vals = get_last_pos(output)
                captured[key] = vals
                print(f"\n[ResBlock {idx}] mean={vals.mean():.6f}, std={vals.std():.6f}")
                print(f"  first 8: {vals[:8].tolist()}")
        return hook

    def final_layer_hook(module, input, output):
        if capture_active[0] and 'velocity' not in captured:
            vals = get_last_pos(output)
            captured['velocity'] = vals
            print(f"\n[FinalLayer/velocity] ALL 32: {vals.tolist()}")

    def flownet_hook(module, input, output):
        if capture_active[0] and 'final_latent' not in captured:
            vals = get_last_pos(output)
            captured['final_latent'] = vals
            capture_active[0] = False  # Done capturing
            print(f"\n[FlowNet output] ALL 32: {vals.tolist()}")

    hooks = []
    hooks.append(flow_net.input_proj.register_forward_hook(input_proj_hook))
    hooks.append(flow_net.cond_embed.register_forward_hook(cond_embed_hook))
    hooks.append(flow_net.time_embed[0].register_forward_hook(time_embed_0_hook))
    hooks.append(flow_net.time_embed[1].register_forward_hook(time_embed_1_hook))
    for i, block in enumerate(flow_net.res_blocks):
        hooks.append(block.register_forward_hook(make_resblock_hook(i)))
    hooks.append(flow_net.final_layer.register_forward_hook(final_layer_hook))
    hooks.append(flow_net.register_forward_hook(flownet_hook))

    try:
        voice_state = model.get_state_for_audio_prompt("alba")
        text = "Hello, this is a test of the Pocket TTS system."
        print(f'\nGenerating: "{text}"')
        print("=" * 70)
        audio = model.generate_audio(voice_state, text)
        print("=" * 70)
        print(f"Audio samples: {len(audio)}")
        print(f"Total FlowNet input_proj calls: {call_count[0]}")
        print(f"Found target: {found_target[0]}")

        if captured:
            outpath = 'validation/python_flownet_step0.npz'
            np.savez(outpath, **captured)
            print(f"\nSaved {len(captured)} arrays to {outpath}")
    finally:
        for h in hooks:
            h.remove()

if __name__ == "__main__":
    main()
