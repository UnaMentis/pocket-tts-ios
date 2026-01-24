#!/usr/bin/env python3
"""
Dump FlowNet velocity values using zeros instead of random noise.
This allows direct comparison with Rust which is also using zeros.

Patches torch.nn.init.normal_ to return zeros instead.
"""

import torch
import sys

# Monkey-patch torch.nn.init functions to return zeros
_original_normal_ = torch.nn.init.normal_
_original_trunc_normal_ = torch.nn.init.trunc_normal_

def zeros_normal_(tensor, mean=0.0, std=1.0):
    """Replace normal_ with zeros for deterministic comparison."""
    with torch.no_grad():
        tensor.zero_()
    return tensor

def zeros_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Replace trunc_normal_ with zeros for deterministic comparison."""
    with torch.no_grad():
        tensor.zero_()
    return tensor

# Apply patches
torch.nn.init.normal_ = zeros_normal_
torch.nn.init.trunc_normal_ = zeros_trunc_normal_

# Now import pocket_tts after patching
from pocket_tts import TTSModel

# Storage for captured values
CAPTURED = {}

def main():
    print("Loading model...")
    model = TTSModel.load_model()

    # Get the flow_net
    flow_net = model.flow_lm.flow_net
    print(f"FlowNet type: {type(flow_net)}")

    # Hook to capture velocity (flow_net output) at step 0
    # The flow_net is called via: conditioned_flow = partial(self.flow_net, transformer_out)
    # Then lsd_decode(conditioned_flow, noise, ...)
    # Inside lsd_decode: flow_dir = v_t(s * ones, t * ones, current)
    # So flow_net gets: (c=transformer_out, s=tensor, t=tensor, x=current)

    call_count = [0]

    def flow_net_hook(module, args, output):
        call_count[0] += 1
        c, s, t, x = args

        # s and t are [batch, 1] tensors with the time values
        s_val = s[0, 0].item() if s.dim() > 1 else s[0].item()
        t_val = t[0, 0].item() if t.dim() > 1 else t[0].item()

        print(f"\n[Python-FlowNet] call #{call_count[0]}: s={s_val:.3f}, t={t_val:.3f}")
        print(f"  x shape: {x.shape}, x first 8: {x.flatten()[:8].tolist()}")
        print(f"  c shape (BEFORE cond_embed): {c.shape}")

        # Show raw c stats
        c_flat = c.flatten().detach().cpu()
        c_mean = c_flat.mean().item()
        c_std = c_flat.std().item()
        print(f"  c (raw) first 8: {c_flat[:8].tolist()}")
        print(f"  c (raw) mean={c_mean:.6f}, std={c_std:.4f}")

        # Project c to see what cond_embed produces
        with torch.no_grad():
            c_proj = module.cond_embed(c)
            c_proj_flat = c_proj.flatten().detach().cpu()
            c_proj_mean = c_proj_flat.mean().item()
            c_proj_std = c_proj_flat.std().item()
            print(f"  c (after cond_embed) first 8: {c_proj_flat[:8].tolist()}")
            print(f"  c (after cond_embed) mean={c_proj_mean:.6f}, std={c_proj_std:.4f}")

        print(f"  output shape: {output.shape}")

        out_flat = output.flatten().detach().cpu()
        print(f"  velocity first 8: {out_flat[:8].tolist()}")

        v_mean = out_flat.mean().item()
        v_std = out_flat.std().item()
        print(f"  velocity mean={v_mean:.6f}, std={v_std:.4f}")

        # Capture step 0 (s=0, t=1 for 1-step) - detailed breakdown
        if s_val < 0.01 and 'step0_velocity' not in CAPTURED:
            with torch.no_grad():
                # 1. input_proj(x)
                h = module.input_proj(x)
                h_flat = h.flatten().detach().cpu()
                print(f"  [BREAKDOWN] after input_proj first 8: {h_flat[:8].tolist()}")

                # 2. Time embeddings
                ts = [s, t]
                t_emb_s = module.time_embed[0](ts[0])
                t_emb_t = module.time_embed[1](ts[1])
                te_s_flat = t_emb_s.flatten().detach().cpu()
                te_t_flat = t_emb_t.flatten().detach().cpu()
                print(f"  [BREAKDOWN] time_embed_s (s=0) first 8: {te_s_flat[:8].tolist()}")
                print(f"  [BREAKDOWN] time_embed_t (t=1) first 8: {te_t_flat[:8].tolist()}")

                # 3. Average time embeddings
                t_combined = (t_emb_s + t_emb_t) / 2
                tc_flat = t_combined.flatten().detach().cpu()
                print(f"  [BREAKDOWN] avg_time_emb first 8: {tc_flat[:8].tolist()}")

                # 4. y = t_combined + c (after cond_embed)
                y = t_combined + c_proj
                y_flat = y.flatten().detach().cpu()
                print(f"  [BREAKDOWN] cond_combined (c + time) first 8: {y_flat[:8].tolist()}")

            CAPTURED['step0_velocity'] = out_flat[:8].tolist()
            CAPTURED['step0_c_raw_first8'] = c_flat[:8].tolist()
            CAPTURED['step0_c_proj_first8'] = c_proj_flat[:8].tolist()
            CAPTURED['step0_x_first8'] = x.flatten()[:8].detach().cpu().tolist()

    # Register hook
    handle = flow_net.register_forward_hook(flow_net_hook)

    try:
        print("\nGetting voice state...")
        voice_state = model.get_state_for_audio_prompt("alba")

        print("\n" + "=" * 60)
        print("Generating: 'Hello, this is a test.' (with ZEROS noise)")
        print("=" * 60)

        audio = model.generate_audio(voice_state, "Hello, this is a test.")
        print(f"\nGenerated {len(audio)} samples")
    finally:
        handle.remove()

    print("\n" + "=" * 60)
    print("SUMMARY - FlowNet with zeros (for comparison with Rust):")
    print("=" * 60)
    for key, val in sorted(CAPTURED.items()):
        print(f"  {key}: {val}")

    print("\n" + "=" * 60)
    print("RUST VALUES (for reference):")
    print("  velocity first 8: [-0.6430681, 0.23889166, 0.2160035, 1.347405, -0.71227896, 3.2762485, -0.24154162, 0.1518566]")
    print("=" * 60)

if __name__ == "__main__":
    main()
