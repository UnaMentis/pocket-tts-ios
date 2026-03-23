#!/usr/bin/env python3
"""
Dump Mimi decoder per-block intermediate tensors for comparison with Rust.

This runs the SAME streaming pipeline as generate_audio, processing one latent
frame at a time through the full Mimi decoder, and dumps intermediates at each
stage for the first N frames.

Stages dumped per frame:
  1. after_output_proj  - quantizer.output_proj: [1, 512, 1]
  2. after_upsample     - ConvTrUpsample1d: [1, 512, 16]
  3. after_dec_transformer - decoder_transformer: [1, 512, 16]
  4. after_seanet       - SEANet decoder: [1, 1, samples]

Also dumps per-layer SEANet intermediates for frame 0.

Usage:
    python validation/dump_mimi_per_block.py --output-dir /tmp/mimi_blocks
"""

import argparse
from pathlib import Path
import json
import numpy as np
import torch
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps


def dump_tensor(t, name, output_dir):
    """Save tensor as .npy and return stats."""
    arr = t.detach().cpu().numpy()
    np.save(output_dir / f"{name}.npy", arr)
    return {
        'shape': list(arr.shape),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'abs_max': float(np.max(np.abs(arr))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/mimi_blocks"))
    parser.add_argument("--latents-file", type=Path,
                       default=Path("validation/reference_outputs/phrase_00_latents.npy"))
    parser.add_argument("--num-frames", type=int, default=5,
                       help="Number of frames to dump intermediates for")
    parser.add_argument("--seanet-detail-frame", type=int, default=0,
                       help="Frame index to dump detailed SEANet per-layer outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = TTSModel.load_model()
    mimi = model.mimi

    # Load and prepare latents
    latents_np = np.load(args.latents_file)
    print(f"Loaded latents: shape={latents_np.shape}, dtype={latents_np.dtype}")

    latents = torch.from_numpy(latents_np)  # [seq, 32]
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)  # [1, seq, 32]
    num_frames = latents.shape[1]
    print(f"Total frames: {num_frames}")

    # Get denormalization parameters
    emb_mean = model.flow_lm.emb_mean.detach()
    emb_std = model.flow_lm.emb_std.detach()
    print(f"emb_mean shape: {emb_mean.shape}, emb_std shape: {emb_std.shape}")

    # Initialize streaming state (same as _decode_audio_worker)
    mimi_context = model.config.mimi.transformer.context
    mimi_state = init_states(mimi, batch_size=1, sequence_length=mimi_context)

    all_stats = {}

    with torch.no_grad():
        all_audio_chunks = []

        for frame_idx in range(num_frames):
            frame_latent = latents[:, frame_idx:frame_idx+1, :]  # [1, 1, 32]

            # Step 1: Denormalize (same as _decode_audio_worker)
            denorm = frame_latent * emb_std + emb_mean  # [1, 1, 32]

            if frame_idx < args.num_frames:
                stats = dump_tensor(denorm, f"frame_{frame_idx:03d}_00_denorm", args.output_dir)
                all_stats[f"frame_{frame_idx:03d}_00_denorm"] = stats
                print(f"  frame {frame_idx} denorm: {stats['shape']}, abs_max={stats['abs_max']:.6f}")

            # Step 2: Transpose and quantizer output_proj
            transposed = denorm.transpose(-1, -2)  # [1, 32, 1]
            quantized = mimi.quantizer(transposed)  # [1, 512, 1]

            if frame_idx < args.num_frames:
                stats = dump_tensor(quantized, f"frame_{frame_idx:03d}_01_output_proj", args.output_dir)
                all_stats[f"frame_{frame_idx:03d}_01_output_proj"] = stats
                print(f"  frame {frame_idx} output_proj: {stats['shape']}, abs_max={stats['abs_max']:.6f}")

            # Step 3: Upsample (16x temporal)
            upsampled = mimi._to_encoder_framerate(quantized, mimi_state)  # [1, 512, 16]

            if frame_idx < args.num_frames:
                stats = dump_tensor(upsampled, f"frame_{frame_idx:03d}_02_upsample", args.output_dir)
                all_stats[f"frame_{frame_idx:03d}_02_upsample"] = stats
                print(f"  frame {frame_idx} upsample: {stats['shape']}, abs_max={stats['abs_max']:.6f}")

            # Step 4: Decoder transformer
            dec_tr_out = mimi.decoder_transformer(upsampled, mimi_state)
            dec_tr_out = dec_tr_out[0]  # Unpack tuple

            if frame_idx < args.num_frames:
                stats = dump_tensor(dec_tr_out, f"frame_{frame_idx:03d}_03_dec_transformer", args.output_dir)
                all_stats[f"frame_{frame_idx:03d}_03_dec_transformer"] = stats
                print(f"  frame {frame_idx} dec_transformer: {stats['shape']}, abs_max={stats['abs_max']:.6f}")

            # Step 5: SEANet decoder
            # For the detail frame, dump per-layer outputs
            if frame_idx == args.seanet_detail_frame:
                x = dec_tr_out
                decoder = mimi.decoder
                for layer_idx, layer in enumerate(decoder.model):
                    if isinstance(layer, torch.nn.ELU):
                        x = layer(x)
                        layer_name = "ELU"
                    else:
                        # StreamingConv1d, StreamingConvTranspose1d, SEANetResnetBlock
                        x = layer(x, mimi_state)
                        layer_name = type(layer).__name__

                    stats = dump_tensor(x,
                        f"frame_{frame_idx:03d}_04_seanet_layer_{layer_idx:02d}_{layer_name}",
                        args.output_dir)
                    all_stats[f"frame_{frame_idx:03d}_04_seanet_layer_{layer_idx:02d}_{layer_name}"] = stats
                    print(f"  frame {frame_idx} SEANet[{layer_idx}] {layer_name}: {stats['shape']}, abs_max={stats['abs_max']:.6f}")

                audio_frame = x
            else:
                audio_frame = mimi.decoder(dec_tr_out, mimi_state)

            if frame_idx < args.num_frames:
                stats = dump_tensor(audio_frame, f"frame_{frame_idx:03d}_05_audio", args.output_dir)
                all_stats[f"frame_{frame_idx:03d}_05_audio"] = stats
                print(f"  frame {frame_idx} audio: {stats['shape']}, abs_max={stats['abs_max']:.6f}")

            all_audio_chunks.append(audio_frame)

            # Increment streaming step (same as _decode_audio_worker)
            increment_steps(mimi, mimi_state, increment=16)

        # Concatenate all audio
        full_audio = torch.cat(all_audio_chunks, dim=-1)
        stats = dump_tensor(full_audio, "full_audio", args.output_dir)
        all_stats["full_audio"] = stats
        print(f"\nFull audio: {stats['shape']}, abs_max={stats['abs_max']:.6f}")

    # Save stats
    with open(args.output_dir / "stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nSaved {len(all_stats)} tensors to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
