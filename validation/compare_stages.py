#!/usr/bin/env python3
"""
Compare intermediate stages between Python and expected Rust output.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states


def main():
    parser = argparse.ArgumentParser(description="Compare stages")
    parser.add_argument("--latents", type=Path, default=Path("reference_zeros/python_latents.npy"))
    args = parser.parse_args()

    print("Loading model...")
    model = TTSModel.load_model()
    mimi = model.mimi
    flow_lm = model.flow_lm

    # Load and denormalize latents
    print(f"\nLoading latents from {args.latents}...")
    latents = np.load(args.latents)
    latents = torch.from_numpy(latents)
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)

    emb_mean = flow_lm.emb_mean.detach()
    emb_std = flow_lm.emb_std.detach()
    latents = latents * emb_std + emb_mean
    print(f"Denormalized latents: shape={latents.shape}, mean={latents.mean():.4f}, std={latents.std():.4f}")

    # Process through stages WITHOUT streaming state for fairer comparison
    print("\n=== Processing without streaming state ===")

    with torch.no_grad():
        # Stage 1: Transpose to [batch, latent_dim, seq]
        x = latents.transpose(1, 2)
        print(f"After transpose: {x.shape}")

        # Stage 2: output_proj
        x = mimi.quantizer.output_proj(x)
        print(f"After output_proj: {x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")

        # Save for comparison
        np.save("stage_output_proj.npy", x.numpy())

        # Stage 3: upsample (without streaming state)
        # Use the underlying ConvTranspose1d directly instead of streaming wrapper
        upsample_weight = mimi.upsample.convtr.convtr.weight
        print(f"Upsample weight shape: {upsample_weight.shape}")  # [512, 1, 32] for depthwise

        # Manual batch ConvTranspose1d
        # For depthwise with groups=512: weight is [512, 1, 32]
        # Python's F.conv_transpose1d with no padding
        x_upsample = torch.nn.functional.conv_transpose1d(
            x,
            upsample_weight,
            bias=None,
            stride=16,
            padding=0,
            output_padding=0,
            groups=512,
            dilation=1,
        )
        # This produces (T - 1) * stride + kernel = (44-1) * 16 + 32 = 43*16 + 32 = 720
        print(f"After raw upsample (no padding): {x_upsample.shape}, mean={x_upsample.mean():.4f}, std={x_upsample.std():.4f}")

        # Python's StreamingConvTranspose1d uses overlap-add
        # For batch mode, let's trim to expected size
        # Expected: T * stride = 44 * 16 = 704 for Python's 44 frames
        # But we have 44 frames, so output should be (44-1)*16 + 32 = 720 without trim

        np.save("stage_upsample_raw.npy", x_upsample.numpy())

        # Now use Python's streaming upsample for comparison
        mimi_state = init_states(mimi, batch_size=1, sequence_length=1000)
        x_streaming = x.clone()
        x_streaming_up = mimi._to_encoder_framerate(x_streaming, mimi_state)
        print(f"After streaming upsample: {x_streaming_up.shape}, mean={x_streaming_up.mean():.4f}, std={x_streaming_up.std():.4f}")

        np.save("stage_upsample_streaming.npy", x_streaming_up.numpy())

        # Compare raw vs streaming
        # They should have different lengths due to overlap-add trimming
        print(f"\nRaw upsample length: {x_upsample.shape[2]}")
        print(f"Streaming upsample length: {x_streaming_up.shape[2]}")

        # Continue with decoder_transformer using streaming upsample output
        dec_out = mimi.decoder_transformer(x_streaming_up, mimi_state)
        dec_out = dec_out[0]
        print(f"After decoder_transformer: {dec_out.shape}, mean={dec_out.mean():.4f}, std={dec_out.std():.4f}")

        np.save("stage_decoder_transformer.npy", dec_out.numpy())

        # Final audio
        audio = mimi.decoder(dec_out, mimi_state)
        print(f"Final audio: {audio.shape}, mean={audio.mean():.4f}, max={torch.max(torch.abs(audio)):.4f}")

        np.save("stage_final_audio.npy", audio.numpy())

    print("\nStage outputs saved to stage_*.npy files")


if __name__ == "__main__":
    main()
