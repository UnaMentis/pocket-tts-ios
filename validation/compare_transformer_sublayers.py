#!/usr/bin/env python3
"""
Compare Rust vs Python decoder transformer sub-layer outputs.

Uses the Rust-dumped denormalized latents as input to ensure identical inputs.
Runs the Python Mimi decoder transformer in BATCH mode (model_state=None)
and compares each sub-layer output.

Usage:
    python validation/compare_transformer_sublayers.py
"""

import numpy as np
import torch
from pocket_tts import TTSModel


def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    ml = min(len(a), len(b))
    a, b = a[:ml], b[:ml]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def max_abs_error(a, b):
    a, b = a.flatten(), b.flatten()
    ml = min(len(a), len(b))
    return float(np.max(np.abs(a[:ml] - b[:ml])))


def rmse(a, b):
    a, b = a.flatten(), b.flatten()
    ml = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:ml] - b[:ml]) ** 2)))


def compare(name, rs_arr, py_arr):
    cos = cosine_similarity(rs_arr, py_arr)
    mae = max_abs_error(rs_arr, py_arr)
    rms = rmse(rs_arr, py_arr)
    status = "OK" if cos > 0.999 else ("WARN" if cos > 0.99 else "FAIL")
    print(f"  [{status}] {name:40s}: cos={cos:.6f}, max_err={mae:.2e}, rmse={rms:.2e}, "
          f"rs_shape={rs_arr.shape}, py_shape={py_arr.shape}")
    return cos


def main():
    dump_dir = "/tmp/mimi_blocks"

    print("Loading model...")
    model = TTSModel.load_model()
    mimi = model.mimi

    # Load Rust's denormalized latents (the exact input to the Mimi decoder)
    rs_denorm = np.load(f"{dump_dir}/rs_denorm_latents.npy")
    print(f"Rust denorm latents: shape={rs_denorm.shape}")
    latents = torch.from_numpy(rs_denorm)
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)  # [1, seq, 32]

    num_frames = latents.shape[1]
    print(f"Total frames: {num_frames}")

    with torch.no_grad():
        # Step 1: output_proj (same as Rust)
        transposed = latents.transpose(1, 2)  # [1, 32, seq]
        quantized = mimi.quantizer(transposed)  # [1, 512, seq]
        print(f"output_proj: {quantized.shape}")

        # Step 2: batch upsample
        # In Python streaming mode, upsample is done per-frame.
        # But Rust does batch upsample in the dump path. Let's just batch it.
        # Actually, mimi._to_encoder_framerate needs streaming state.
        # The Rust dump path does streaming upsample. Let's match that.
        from pocket_tts.modules.stateful_module import init_states, increment_steps
        mimi_context = model.config.mimi.transformer.context
        up_state = init_states(mimi, batch_size=1, sequence_length=mimi_context)
        up_chunks = []
        for f in range(num_frames):
            frame_q = quantized[:, :, f:f+1]
            up = mimi._to_encoder_framerate(frame_q, up_state)
            up_chunks.append(up)
            increment_steps(mimi, up_state, increment=16)
        all_upsampled = torch.cat(up_chunks, dim=-1)  # [1, 512, seq*16]
        print(f"upsampled: {all_upsampled.shape}")

        # Compare upsample with Rust
        rs_up_f0 = np.load(f"{dump_dir}/rs_f0_upsample.npy")
        py_up_f0 = all_upsampled[:, :, :16].numpy()
        compare("upsample_frame0", rs_up_f0, py_up_f0)

        # Step 3: Now run through the decoder transformer in BATCH mode
        # The decoder_transformer is a ProjectedTransformer
        # Its forward: x.T(1,2) -> input_proj -> transformer(x, model_state=None) -> output_proj -> y.T(1,2)
        # For the Mimi decoder_transformer: input_dim=512, d_model=512, so input_proj=None
        # output_proj is Identity (since d_model == output_dim)
        dec_tr = mimi.decoder_transformer

        # Rust transposes to [B, T, C] before calling transformer
        # Python ProjectedTransformer also transposes: x.T(1,2) -> transformer -> y.T(1,2)
        # So input to ProjectedTransformer is [B, C, T], matching all_upsampled

        # First, let's get the transformer input (after transpose)
        tr_input = all_upsampled.transpose(1, 2)  # [1, T, 512]

        # Compare transformer input
        rs_tr_input = np.load(f"{dump_dir}/rs_tr_input.npy")
        compare("transformer_input", rs_tr_input, tr_input.numpy())

        # Now run through layer 0 manually with dumps
        layer0 = dec_tr.transformer.layers[0]
        rope = dec_tr.transformer.rope
        x = tr_input  # [1, T, 512]

        # norm1
        h = layer0.norm1(x)
        rs_norm1 = np.load(f"{dump_dir}/rs_tr_L0_norm1.npy")
        compare("L0_norm1", rs_norm1, h.numpy())

        # in_proj
        projected = layer0.self_attn.in_proj(h)  # [1, T, 1536]
        rs_in_proj = np.load(f"{dump_dir}/rs_tr_L0_in_proj.npy")
        compare("L0_in_proj", rs_in_proj, projected.numpy())

        # Split QKV - using einops rearrange like Python does
        from einops import rearrange
        num_heads = layer0.self_attn.num_heads
        q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=num_heads)

        # Compare q, k, v pre-rope
        # Rust has [batch, heads, seq, head_dim] from permute
        rs_q_pre = np.load(f"{dump_dir}/rs_tr_L0_q_pre_rope.npy")
        rs_k_pre = np.load(f"{dump_dir}/rs_tr_L0_k_pre_rope.npy")
        rs_v = np.load(f"{dump_dir}/rs_tr_L0_v.npy")
        compare("L0_q_pre_rope", rs_q_pre, q.numpy())
        compare("L0_k_pre_rope", rs_k_pre, k.numpy())
        compare("L0_v", rs_v, v.numpy())

        # Apply RoPE
        # Python: permutes to [B, T, H, D] for rope, then back
        q_for_rope = q.permute(0, 2, 1, 3)  # [B, T, H, D]
        k_for_rope = k.permute(0, 2, 1, 3)  # [B, T, H, D]

        # Python MimiStreamingMultiheadAttention uses offset from state
        # In batch mode (model_state=None), offset=0
        offset = torch.zeros(1, device=q.device, dtype=torch.long)
        q_rope, k_rope = rope(q_for_rope, k_for_rope, offset)

        # Permute back to [B, H, T, D]
        q_rope = q_rope.permute(0, 2, 1, 3)
        k_rope = k_rope.permute(0, 2, 1, 3)

        rs_q_rope = np.load(f"{dump_dir}/rs_tr_L0_q_rope.npy")
        rs_k_rope = np.load(f"{dump_dir}/rs_tr_L0_k_rope.npy")
        compare("L0_q_rope", rs_q_rope, q_rope.numpy())
        compare("L0_k_rope", rs_k_rope, k_rope.numpy())

        # Attention
        # Python uses F.scaled_dot_product_attention with attn_bias
        # In batch mode (model_state=None), KVCacheResult.from_kv is used
        # which just returns positions = arange(T)
        T = q.shape[2]
        pos_k = torch.arange(T, device=q.device, dtype=torch.long).unsqueeze(0)  # [1, T]
        pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)  # [T, 1]
        delta = pos_q - pos_k[:, None]  # broadcasting to [1, T, T] via Python's shapes
        # Actually let me match the Python code exactly:
        pos_k_exp = pos_k[:, None]  # [1, 1, T]
        pos_q_exp = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)  # [T, 1]
        delta = pos_q_exp - pos_k_exp  # [1, T, T]
        context = layer0.self_attn.context
        attn_bias = (pos_k_exp >= 0) & (delta >= 0) & (delta < context)
        attn_bias = attn_bias[:, None]  # [1, 1, T, T]
        print(f"\n  Python attention mask: context={context}, mask_shape={attn_bias.shape}")
        print(f"  Mask[0,0,:5,:5]:\n{attn_bias[0,0,:5,:5]}")
        print(f"  Mask all True? {attn_bias.all().item()}")
        # How many False entries?
        n_false = (~attn_bias).sum().item()
        n_total = attn_bias.numel()
        print(f"  False entries: {n_false}/{n_total} ({100*n_false/n_total:.1f}%)")

        # Now compute manual attention scores (without sdpa) for comparison
        head_dim = q.shape[-1]
        scale = head_dim ** 0.5
        attn_scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / scale
        rs_attn_scores = np.load(f"{dump_dir}/rs_tr_L0_attn_scores.npy")
        compare("L0_attn_scores (pre-mask)", rs_attn_scores, attn_scores.numpy())

        # Apply mask (Rust doesn't apply any mask - it's full bidirectional)
        # Python applies context-window mask via attn_bias
        if n_false > 0:
            print(f"\n  *** CRITICAL: Python uses context window mask (context={context}) ***")
            print(f"  *** Rust uses NO mask (full bidirectional attention) ***")
            print(f"  *** This is likely the source of divergence! ***\n")

            # Also compute what Python's sdpa would produce
            attn_out_py = torch.nn.functional.scaled_dot_product_attention(
                q_rope, k_rope, v, attn_bias.float().masked_fill(~attn_bias, float('-inf')).masked_fill(attn_bias, 0.0),
                dropout_p=0.0
            )
            # Actually, sdpa with boolean mask treats True as "attend" and False as "don't attend"
            # Let me use it correctly
            attn_out_py = torch.nn.functional.scaled_dot_product_attention(
                q_rope, k_rope, v, attn_bias, dropout_p=0.0
            )
            print(f"  Python sdpa output: {attn_out_py.shape}")

            # Manual softmax with mask for comparison with Rust
            attn_masked = attn_scores.clone()
            # Where mask is False, set to -inf
            attn_masked = attn_masked.masked_fill(~attn_bias, float('-inf'))
            attn_probs_masked = torch.softmax(attn_masked, dim=-1)
            attn_out_masked = torch.matmul(attn_probs_masked, v)

            # Rust softmax (no mask)
            attn_probs_nomask = torch.softmax(attn_scores, dim=-1)
            rs_attn_probs = np.load(f"{dump_dir}/rs_tr_L0_attn_probs.npy")
            compare("L0_attn_probs (Rust no-mask vs Python masked)", rs_attn_probs, attn_probs_masked.numpy())
            compare("L0_attn_probs (Rust no-mask vs Rust)", rs_attn_probs, attn_probs_nomask.numpy())

            rs_attn_out = np.load(f"{dump_dir}/rs_tr_L0_attn_out_raw.npy")
            compare("L0_attn_out_raw (Rust vs Python masked)", rs_attn_out, attn_out_masked.permute(0,2,1,3).numpy())
        else:
            # No masking difference
            attn_probs = torch.softmax(attn_scores, dim=-1)
            rs_attn_probs = np.load(f"{dump_dir}/rs_tr_L0_attn_probs.npy")
            compare("L0_attn_probs", rs_attn_probs, attn_probs.numpy())

            attn_out = torch.matmul(attn_probs, v)
            rs_attn_out = np.load(f"{dump_dir}/rs_tr_L0_attn_out_raw.npy")
            compare("L0_attn_out_raw", rs_attn_out, attn_out.numpy())


if __name__ == "__main__":
    main()
