#!/usr/bin/env python3
"""
Compare FlowNet step 0 intermediate values between Python and Rust.
Specifically targets the call where c_raw matches Rust's step 0 values.
"""

import torch

# Monkey-patch to use zeros (only for noise, not for model init)
_orig_normal = torch.nn.init.normal_
_orig_trunc = torch.nn.init.trunc_normal_
def zeros_normal_(t, mean=0, std=1):
    # Only patch tensors that don't require grad (noise tensors)
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

def main():
    print("Loading model...")
    model = TTSModel.load_model()
    flow_net = model.flow_lm.flow_net

    # Target: Rust step0 c_raw first 8
    TARGET_C = [-0.2274, -0.3591, 0.0514, 0.5092, 0.2089, 0.5238, 0.6198, 0.2637]

    call_count = [0]
    found = [False]

    def flow_net_hook(module, args, output):
        call_count[0] += 1
        c, s, t, x = args

        s_val = s[0, 0].item() if s.dim() > 1 else s[0].item()
        t_val = t[0, 0].item() if t.dim() > 1 else t[0].item()

        c_flat = c.flatten()[:8].detach().cpu().tolist()

        # Check if this matches the target conditioning
        if not found[0] and abs(c_flat[0] - TARGET_C[0]) < 0.01:
            found[0] = True
            print(f"\n{'='*60}")
            print(f"FOUND MATCHING CALL #{call_count[0]} (s={s_val:.3f}, t={t_val:.3f})")
            print(f"{'='*60}")

            # Show raw inputs
            print(f"\n[INPUT] x first 8: {x.flatten()[:8].tolist()}")
            print(f"[INPUT] c (raw) first 8: {c_flat}")
            print(f"[INPUT] c (raw) mean={c.mean().item():.6f}, std={c.std().item():.4f}")

            with torch.no_grad():
                # 1. input_proj(x) - x is zeros
                h = module.input_proj(x)
                print(f"\n[1] after input_proj first 8: {h.flatten()[:8].tolist()}")

                # 2. Time embeddings
                t_emb_s = module.time_embed[0](s)
                t_emb_t = module.time_embed[1](t)
                print(f"[2] time_embed_s (s=0) first 8: {t_emb_s.flatten()[:8].tolist()}")
                print(f"[2] time_embed_t (t=1) first 8: {t_emb_t.flatten()[:8].tolist()}")

                # 3. Average time embeddings
                t_combined = (t_emb_s + t_emb_t) / 2
                print(f"[3] avg_time_emb first 8: {t_combined.flatten()[:8].tolist()}")

                # 4. c = cond_embed(c)
                c_proj = module.cond_embed(c)
                print(f"[4] c (after cond_embed) first 8: {c_proj.flatten()[:8].tolist()}")

                # 5. y = t_combined + c
                y = t_combined + c_proj
                print(f"[5] cond_combined (c + time) first 8: {y.flatten()[:8].tolist()}")

                # 6. Process through res_blocks
                x_cur = h
                for i, block in enumerate(module.res_blocks):
                    x_cur = block(x_cur, y)
                    if i == 0:
                        print(f"[6] after ResBlock 0 first 8: {x_cur.flatten()[:8].tolist()}")
                        print(f"    ResBlock 0 mean={x_cur.mean().item():.6f}")

                # 7. Final layer
                out = module.final_layer(x_cur, y)
                print(f"\n[7] VELOCITY (final_layer output) first 8: {out.flatten()[:8].tolist()}")
                print(f"    VELOCITY mean={out.mean().item():.6f}, std={out.std().item():.4f}")

            print(f"\n[ACTUAL OUTPUT] velocity first 8: {output.flatten()[:8].tolist()}")

            print(f"\n{'='*60}")
            print("RUST VALUES (for comparison):")
            print(f"  input_proj first 8: [0.046143, 0.017700, -0.001411, -0.029175, -0.013367, -0.043457, 0.011108, 0.002121]")
            print(f"  time_embed_s first 8: [0.041368, -0.021100, -0.003899, 0.007114, 0.004067, -0.002125, -0.003817, -0.001274]")
            print(f"  time_embed_t first 8: [-0.000530, 0.003988, 0.001339, -0.000053, 0.000170, 0.001264, 0.000787, 0.000782]")
            print(f"  avg_time_emb first 8: [0.020419, -0.008556, -0.001280, 0.003531, 0.002119, -0.000430, -0.001515, -0.000246]")
            print(f"  cond_combined first 8: [-0.786096, -0.551375, 0.069754, 1.053466, 0.444372, -0.522660, 0.942159, 0.758426]")
            print(f"  velocity first 8: [-0.660062, 0.169975, 0.264507, 1.333054, -0.690123, 3.197480, -0.264743, 0.094307]")
            print(f"{'='*60}")

    handle = flow_net.register_forward_hook(flow_net_hook)

    try:
        print("\nGetting voice state...")
        voice_state = model.get_state_for_audio_prompt("alba")
        print("\nGenerating audio...")
        audio = model.generate_audio(voice_state, "Hello, this is a test.")
        print(f"\nGenerated {len(audio)} samples")
    finally:
        handle.remove()

    if not found[0]:
        print("WARNING: Target conditioning not found!")

if __name__ == "__main__":
    main()
