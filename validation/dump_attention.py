#!/usr/bin/env python3
"""Dump attention layer values from Python Pocket TTS for comparison with Rust."""

import sys
import torch
import numpy as np

# Add pocket-tts to path
sys.path.insert(0, "/Users/ramerman/dev/unamentis/models/kyutai-pocket-ios")

from pocket_tts import PocketTTS

def main():
    model_dir = "/Users/ramerman/dev/unamentis/models/kyutai-pocket-ios"

    print("Loading model...")
    tts = PocketTTS(model_dir)

    # Hook to capture attention values
    attention_values = {}

    def make_attn_hook(layer_idx, name):
        def hook(module, input, output):
            if layer_idx == 0:  # Only capture layer 0
                key = f"layer{layer_idx}_{name}"
                if key not in attention_values:
                    attention_values[key] = []
                attention_values[key].append({
                    'input': [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in input],
                    'output': output.detach().cpu() if isinstance(output, torch.Tensor) else output
                })
        return hook

    # Register hooks on attention layers
    for i, layer in enumerate(tts.model.flow_lm.layers):
        # Hook the attention forward to see Q, K, V
        layer.self_attn.register_forward_hook(make_attn_hook(i, "attn"))

    # Also hook the Q, K, V projections specifically
    def qkv_hook(module, input, output):
        x = input[0]
        # Get Q, K, V from the combined projection
        qkv = output
        hidden_size = qkv.shape[-1] // 3
        q = qkv[..., :hidden_size]
        k = qkv[..., hidden_size:2*hidden_size]
        v = qkv[..., 2*hidden_size:]

        print(f"\n[Python] QKV projection input shape: {x.shape}")
        print(f"[Python] Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

        # For first text token (after voice prompting)
        if x.shape[1] > 1:  # Multi-token input (likely text phase)
            # Q for first text token, first head
            num_heads = 16
            head_dim = hidden_size // num_heads
            q_heads = q.view(q.shape[0], q.shape[1], num_heads, head_dim)
            k_heads = k.view(k.shape[0], k.shape[1], num_heads, head_dim)
            v_heads = v.view(v.shape[0], v.shape[1], num_heads, head_dim)

            print(f"[Python-L0] Q head0 first 8 before RoPE: {q_heads[0, 0, 0, :8].tolist()}")
            print(f"[Python-L0] K head0 first 8 before RoPE: {k_heads[0, 0, 0, :8].tolist()}")
            print(f"[Python-L0] V head0 first 8: {v_heads[0, 0, 0, :8].tolist()}")

    # Hook the in_proj of layer 0
    tts.model.flow_lm.layers[0].self_attn.in_proj.register_forward_hook(qkv_hook)

    # We need to manually trace through to get KV cache values
    # This is harder since the cache is internal to the attention

    text = "Hello, this is a test."
    print(f"\nSynthesizing: {text}")
    print("=" * 60)

    # Tokenize
    tokens = tts.tokenizer.encode(text)
    print(f"Tokens: {tokens}")

    # Generate with verbose output
    try:
        audio = tts.synthesize(text, voice="alba")
    except Exception as e:
        print(f"Error during synthesis: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Attention values captured:")
    for key, values in attention_values.items():
        print(f"  {key}: {len(values)} captures")

if __name__ == "__main__":
    main()
