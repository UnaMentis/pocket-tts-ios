#!/usr/bin/env python3
"""Dump Step 0 (first latent generation) intermediate values from Python Pocket TTS."""

from pocket_tts import TTSModel

def main():
    print("Loading model...")
    model = TTSModel.load_model()

    # Get transformer layer 0
    layer0 = model.flow_lm.transformer.layers[0]

    # Track call counts to detect step 0
    # Voice: 125 tokens, Text: 7 tokens, Step 0: 1 token
    CAPTURED = {}

    # Hook norm1 output - step 0 has seq_len=1
    def norm1_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 1 and 'step0_norm1' not in CAPTURED:
            print(f"\n[Python-Step0] norm1 output shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-Step0] norm1 first 8: {vals}")
            CAPTURED['step0_norm1'] = vals

    # Hook self_attn output
    def attn_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 1 and 'step0_attn' not in CAPTURED:
            print(f"[Python-Step0] attn output shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-Step0] attn output first 8: {vals}")
            CAPTURED['step0_attn'] = vals

    # Hook norm2 output
    def norm2_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 1 and 'step0_norm2' not in CAPTURED:
            print(f"[Python-Step0] norm2 output shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-Step0] norm2 first 8: {vals}")
            CAPTURED['step0_norm2'] = vals

    # Hook linear2 (MLP output)
    def linear2_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 1 and 'step0_mlp' not in CAPTURED:
            print(f"[Python-Step0] MLP output shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-Step0] MLP first 8: {vals}")
            CAPTURED['step0_mlp'] = vals

    # Hook full layer output
    def layer_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 1 and 'step0_layer_output' not in CAPTURED:
            print(f"\n[Python-Step0] LAYER OUTPUT shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-Step0] LAYER OUTPUT first 8: {vals}")
            CAPTURED['step0_layer_output'] = vals

    # Hook out_norm to capture final hidden state for step 0
    def out_norm_hook(module, input, output):
        seq_len = output.shape[1]
        if seq_len == 1 and 'step0_final_hidden' not in CAPTURED:
            print(f"\n[Python-Step0] FINAL HIDDEN (after out_norm) shape: {output.shape}")
            vals = output[0, 0, :8].tolist()
            print(f"[Python-Step0] FINAL HIDDEN first 8: {vals}")
            CAPTURED['step0_final_hidden'] = vals

    # Register hooks
    h1 = layer0.norm1.register_forward_hook(norm1_hook)
    h2 = layer0.self_attn.register_forward_hook(attn_hook)
    h3 = layer0.norm2.register_forward_hook(norm2_hook)
    h4 = layer0.linear2.register_forward_hook(linear2_hook)
    h5 = layer0.register_forward_hook(layer_hook)
    h6 = model.flow_lm.out_norm.register_forward_hook(out_norm_hook)

    try:
        print("\nGetting voice state...")
        voice_state = model.get_state_for_audio_prompt("alba")

        print("\n" + "=" * 60)
        print("Generating: 'Hello, this is a test.'")
        print("=" * 60)

        audio = model.generate_audio(voice_state, "Hello, this is a test.")
        print(f"\nGenerated {len(audio)} samples")
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()
        h5.remove()
        h6.remove()

    print("\n" + "=" * 60)
    print("SUMMARY - Step 0 (first latent generation):")
    for key, val in sorted(CAPTURED.items()):
        print(f"  {key}: {val}")

if __name__ == "__main__":
    main()
