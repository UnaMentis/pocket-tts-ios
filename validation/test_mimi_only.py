#!/usr/bin/env python3
"""
Test Mimi decoder in isolation with saved latents.
This helps debug differences between Python and Rust Mimi implementations.
"""

import subprocess
import numpy as np
import torch
from pathlib import Path
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states


def main():
    print("Loading model...")
    model = TTSModel.load_model()
    mimi = model.mimi
    flow_lm = model.flow_lm

    # Load normalized latents
    latents_path = Path("reference_zeros/python_latents.npy")
    print(f"\nLoading latents from {latents_path}...")
    latents = np.load(latents_path)
    print(f"Normalized latents shape: {latents.shape}")

    # Save first 8 values for comparison
    print(f"First 8 latent values (frame 0): {latents[0, :8]}")
    print(f"Last 8 latent values (frame 0): {latents[0, -8:]}")

    # Get denormalization params
    emb_mean = flow_lm.emb_mean.detach().numpy()
    emb_std = flow_lm.emb_std.detach().numpy()
    print(f"\nemb_mean first 8: {emb_mean[:8]}")
    print(f"emb_std first 8: {emb_std[:8]}")

    # Denormalize
    latents_denorm = latents * emb_std + emb_mean
    print(f"\nDenormalized first 8 (frame 0): {latents_denorm[0, :8]}")

    # Convert to tensor and add batch dim
    latents_t = torch.from_numpy(latents_denorm).unsqueeze(0)  # [1, 44, 32]
    print(f"Latent tensor shape: {latents_t.shape}")

    # Initialize streaming state
    mimi_state = init_states(mimi, batch_size=1, sequence_length=1000)

    # Run through Mimi decoder
    with torch.no_grad():
        # transpose to [B, latent_dim, seq]
        x = latents_t.transpose(1, 2)
        print(f"\nAfter transpose: {x.shape}")

        # quantizer output_proj
        x = mimi.quantizer.output_proj(x)
        print(f"After output_proj: shape={x.shape}, first 8: {x[0, :8, 0].numpy()}")

        # upsample
        x = mimi._to_encoder_framerate(x, mimi_state)
        print(f"After upsample: shape={x.shape}, first 8: {x[0, :8, 0].numpy()}")

        # decoder_transformer
        dec_out = mimi.decoder_transformer(x, mimi_state)
        dec_out = dec_out[0]
        print(f"After decoder_transformer: shape={dec_out.shape}, first 8: {dec_out[0, :8, 0].numpy()}")

        # SEANet decoder
        audio = mimi.decoder(dec_out, mimi_state)
        audio = audio.squeeze()
        print(f"Final audio: shape={audio.shape}, max={torch.max(torch.abs(audio)):.4f}")

        # Save audio
        np.save("python_mimi_output.npy", audio.numpy())

    # Also save the denormalized latents for Rust to use
    np.save("denormalized_latents.npy", latents_denorm.astype(np.float32))
    print("\nSaved denormalized_latents.npy and python_mimi_output.npy")


if __name__ == "__main__":
    main()
