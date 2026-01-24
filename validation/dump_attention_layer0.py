#!/usr/bin/env python3
"""
Dump Layer 0 attention values from Python Pocket TTS for comparison with Rust.

We need to capture:
1. K cache at position 0 (from voice processing)
2. K at position 125 (first text token after voice)
3. V at position 0 (from voice)
4. Raw attention scores
5. Attention output after weighted sum
"""

import torch
import numpy as np
from pocket_tts import TTSModel

# Global storage for captured values
CAPTURED = {}

def main():
    print("Loading model...")
    model = TTSModel.load_model()

    # The transformer is at flow_lm.transformer
    transformer = model.flow_lm.transformer
    layer0 = transformer.layers[0]
    attn = layer0.self_attn

    print(f"Layer 0 attention type: {type(attn)}")

    # Hook to capture QKV values from layer 0
    call_count = [0]  # Use list to allow mutation in closure
    text_phase_call = [None]  # Track which call is text phase

    def attn_forward_hook(module, args, output):
        """Hook the entire attention forward to capture intermediate values."""
        call_count[0] += 1
        call_idx = call_count[0]

        # Input is the first positional arg
        x = args[0] if args else None
        if x is not None:
            seq_len = x.shape[1]
            print(f"[Python] Attention L0 call #{call_idx}: input shape {x.shape}")

            # Voice phase: seq_len = 125
            # Text phase: seq_len = 7 (for "Hello, this is a test.")
            if seq_len == 7 and text_phase_call[0] is None:
                text_phase_call[0] = call_idx
                print(f"  ^^^ This is likely TEXT PHASE (seq_len={seq_len})")

                # Capture the Q, K, V from in_proj
                hidden_size = x.shape[-1]
                qkv = module.in_proj(x)  # [batch, seq, 3*hidden]

                q = qkv[..., :hidden_size]
                k = qkv[..., hidden_size:2*hidden_size]
                v = qkv[..., 2*hidden_size:]

                # Reshape to heads: [batch, seq, num_heads, head_dim]
                num_heads = 16
                head_dim = hidden_size // num_heads
                q_heads = q.reshape(q.shape[0], q.shape[1], num_heads, head_dim)
                k_heads = k.reshape(k.shape[0], k.shape[1], num_heads, head_dim)
                v_heads = v.reshape(v.shape[0], v.shape[1], num_heads, head_dim)

                print(f"\n[Python-L0] Q shape: {q_heads.shape}")
                print(f"[Python-L0] Q head0 first 8 before RoPE: {q_heads[0, 0, 0, :8].tolist()}")
                print(f"[Python-L0] K head0 first 8 before RoPE: {k_heads[0, 0, 0, :8].tolist()}")
                print(f"[Python-L0] V head0 first 8: {v_heads[0, 0, 0, :8].tolist()}")

                # Store for later
                CAPTURED['q_before_rope'] = q_heads[0, 0, 0, :8].detach().cpu().numpy()
                CAPTURED['k_before_rope'] = k_heads[0, 0, 0, :8].detach().cpu().numpy()
                CAPTURED['v'] = v_heads[0, 0, 0, :8].detach().cpu().numpy()

        # Output is the attention result
        if hasattr(output, 'shape'):
            print(f"  Output shape: {output.shape}")
            if text_phase_call[0] == call_idx:
                # This is text phase output - capture attention output
                # Output shape: [batch, seq, hidden]
                out_first8 = output[0, 0, :8].detach().cpu().numpy()
                print(f"\n[Python-L0] Attention OUTPUT first 8: {out_first8.tolist()}")
                CAPTURED['attn_output'] = out_first8

    # Get voice state first (this populates KV cache)
    print("\n" + "=" * 60)
    print("Getting voice state for 'alba'...")
    voice_state = model.get_state_for_audio_prompt("alba")
    print(f"Voice state keys: {voice_state.keys()}")

    # Examine the KV cache from voice processing
    for key, cache in voice_state.items():
        if 'layers.0' in key:  # Focus on layer 0
            print(f"\n{key}:")
            print(f"  Cache type: {type(cache)}")
            if hasattr(cache, '__dict__'):
                print(f"  Cache attrs: {list(cache.__dict__.keys())}")
            if hasattr(cache, 'state'):
                state = cache.state
                print(f"  State type: {type(state)}")
                if isinstance(state, tuple) and len(state) == 2:
                    k, v = state
                    print(f"  K shape: {k.shape}")
                    print(f"  V shape: {v.shape}")
                    # KV cache is [batch, num_heads, seq_len, head_dim]
                    if k.dim() == 4:
                        print(f"  K[pos=0] head0 first 8: {k[0, 0, 0, :8].tolist()}")
                        print(f"  V[pos=0] head0 first 8: {v[0, 0, 0, :8].tolist()}")
                        CAPTURED['voice_k_pos0'] = k[0, 0, 0, :8].detach().cpu().numpy()
                        CAPTURED['voice_v_pos0'] = v[0, 0, 0, :8].detach().cpu().numpy()

    # Now register hooks and generate
    print("\n" + "=" * 60)
    print("Generating with text: 'Hello, this is a test.'")
    print("=" * 60)

    handle = attn.register_forward_hook(attn_forward_hook)

    try:
        # Generate audio
        text = "Hello, this is a test."
        audio = model.generate_audio(voice_state, text)
        print(f"\nGenerated {len(audio)} samples")
    finally:
        handle.remove()

    print(f"\nTotal attention L0 calls: {call_count[0]}")
    print(f"Text phase was call #{text_phase_call[0]}")

    # Print captured values
    print("\n" + "=" * 60)
    print("CAPTURED VALUES (for comparison with Rust):")
    for key, val in CAPTURED.items():
        if isinstance(val, np.ndarray):
            print(f"\n{key}: {val}")

if __name__ == "__main__":
    main()
