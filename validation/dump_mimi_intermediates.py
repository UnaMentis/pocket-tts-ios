#!/usr/bin/env python3
"""
Dump Mimi decoder intermediate tensors for comparison with Rust.

This script captures outputs at each stage of the Mimi decoder:
1. quantizer.output_proj (32 -> 512)
2. upsample (16x temporal)
3. decoder_transformer
4. SEANet stages

Usage:
    python dump_mimi_intermediates.py --output-dir ./mimi_debug
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps


def main():
    parser = argparse.ArgumentParser(description="Dump Mimi decoder intermediates")
    parser.add_argument("--output-dir", type=Path, default=Path("mimi_debug"))
    parser.add_argument("--text", type=str, default="Hello, this is a test of the Pocket TTS system.")
    parser.add_argument("--voice", type=str, default="alba")
    parser.add_argument("--use-saved-latents", type=Path, default=None,
                       help="Use previously saved latents instead of generating new ones")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = TTSModel.load_model()
    mimi = model.mimi

    print("\n=== Mimi Model Architecture ===")
    print(f"Sample rate: {mimi.sample_rate}")
    print(f"Frame rate: {mimi.frame_rate}")
    print(f"Dimension: {mimi.dimension}")
    print(f"Encoder frame rate: {mimi.encoder_frame_rate}")

    # Print quantizer info
    print("\n=== Quantizer ===")
    quantizer = mimi.quantizer
    print(f"output_proj: {quantizer.output_proj}")

    # Print decoder transformer info
    print("\n=== Decoder Transformer ===")
    dec_tr = mimi.decoder_transformer
    print(f"Layers: {len(dec_tr.transformer.layers)}")
    print(f"Input projection: {dec_tr.input_proj}")

    # Print upsample info
    print("\n=== Upsampler ===")
    upsample = mimi.upsample
    print(f"Upsample: {upsample}")
    print(f"  convtr: {upsample.convtr}")

    # Print SEANet decoder info
    print("\n=== SEANet Decoder ===")
    decoder = mimi.decoder
    print(f"Dimension: {decoder.dimension}")
    print(f"Channels: {decoder.channels}")
    print(f"N filters: {decoder.n_filters}")
    print(f"Ratios: {decoder.ratios}")
    print(f"Hop length: {decoder.hop_length}")
    print(f"N blocks: {decoder.n_blocks}")
    print(f"\nModel layers ({len(decoder.model)}):")
    for i, layer in enumerate(decoder.model):
        print(f"  [{i}] {layer}")

    # Get denormalization parameters from flow_lm
    emb_mean = model.flow_lm.emb_mean.detach()
    emb_std = model.flow_lm.emb_std.detach()
    print(f"\n=== Denormalization Params ===")
    print(f"emb_mean: shape={emb_mean.shape}, first 8: {emb_mean[:8].numpy()}")
    print(f"emb_std: shape={emb_std.shape}, first 8: {emb_std[:8].numpy()}")

    # Get latents
    if args.use_saved_latents:
        print(f"\nLoading latents from {args.use_saved_latents}...")
        latents = np.load(args.use_saved_latents)
        latents = torch.from_numpy(latents)
        print(f"Loaded normalized latents shape: {latents.shape}")

        # Denormalize: mimi_decoding_input = latent * emb_std + emb_mean
        latents = latents * emb_std + emb_mean
        print(f"After denormalization: mean={latents.mean():.4f}, std={latents.std():.4f}")
    else:
        print(f"\nGenerating latents for: '{args.text}'")
        voice_state = model.get_state_for_audio_prompt(args.voice)

        # Generate using flow_lm only
        with torch.no_grad():
            tokenizer = model.flow_lm.conditioner.tokenizer
            tokens = tokenizer.sp.encode(args.text)
            print(f"Tokens ({len(tokens)}): {tokens}")

            # Run flow_lm to get latents
            audio = model.generate_audio(voice_state, args.text)

        # Can't easily get just latents from the API, so let's hook the decode call
        # For now, let's load from a saved file if available
        print("Note: To get raw latents, we need to hook the model or use saved latents")
        return

    # Process through Mimi decoder step by step
    print("\n=== Processing through Mimi decoder ===")

    # Initialize streaming state for Mimi
    mimi_state = init_states(mimi, batch_size=1, sequence_length=1000)
    print(f"Initialized mimi_state with {len(mimi_state)} modules")

    intermediates = {}

    with torch.no_grad():
        # Ensure latents have batch dimension
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)  # [seq, latent_dim] -> [1, seq, latent_dim]

        # latents: [batch, seq, latent_dim=32]
        print(f"\nInput latents: {latents.shape}")
        intermediates['input_latents'] = latents.numpy()

        # Step 1: quantizer.output_proj (latent -> mimi_dim)
        # First transpose to [batch, latent_dim, seq]
        latent_t = latents.transpose(1, 2)
        print(f"After transpose: {latent_t.shape}")

        # Pass through quantizer (which includes output_proj)
        emb = quantizer(latent_t)
        print(f"After quantizer: {emb.shape}")
        intermediates['after_quantizer'] = emb.numpy()

        # Step 2: upsample (16x temporal) using _to_encoder_framerate
        upsampled = mimi._to_encoder_framerate(emb, mimi_state)
        print(f"After upsample (16x): {upsampled.shape}")
        intermediates['after_upsample'] = upsampled.numpy()

        # Step 3: decoder_transformer
        dec_tr_out = mimi.decoder_transformer(upsampled, mimi_state)
        dec_tr_out = dec_tr_out[0]  # Unpack tuple
        print(f"After decoder_transformer: {dec_tr_out.shape}")
        intermediates['after_decoder_transformer'] = dec_tr_out.numpy()

        # Step 4: SEANet decoder (all at once with streaming state)
        audio = mimi.decoder(dec_tr_out, mimi_state)
        print(f"After SEANet decoder: {audio.shape}")
        intermediates['final_audio'] = audio.numpy()

        # Also run through SEANet layer by layer for detailed comparison
        print("\n=== SEANet layer-by-layer ===")
        x = dec_tr_out
        for i, layer in enumerate(decoder.model):
            if isinstance(layer, torch.nn.ELU):
                x = layer(x)
            else:
                # StreamingConv1d, StreamingConvTranspose1d, SEANetResnetBlock
                x = layer(x, mimi_state)

            layer_name = type(layer).__name__
            print(f"  SEANet[{i}] {layer_name}: {x.shape}, max={torch.max(torch.abs(x)).item():.6f}")
            intermediates[f'seanet_{i:02d}_{layer_name}'] = x.numpy()

    # Save all intermediates
    print(f"\n=== Saving intermediates to {args.output_dir} ===")
    for name, tensor in intermediates.items():
        path = args.output_dir / f"{name}.npy"
        np.save(path, tensor)
        print(f"  {name}: shape={tensor.shape}, mean={np.mean(tensor):.6f}, std={np.std(tensor):.6f}, max={np.max(np.abs(tensor)):.6f}")

    # Save stats summary
    stats = {name: {
        'shape': list(tensor.shape),
        'mean': float(np.mean(tensor)),
        'std': float(np.std(tensor)),
        'min': float(np.min(tensor)),
        'max': float(np.max(tensor)),
    } for name, tensor in intermediates.items()}

    import json
    with open(args.output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
