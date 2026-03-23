#!/usr/bin/env python3
"""
Compare Mimi decoder: streaming transformer vs batch transformer.

The key architectural difference between Python and Rust:
  - Python: processes 1 frame at a time (upsample->transformer->seanet per frame)
  - Rust: streaming upsample, BATCH transformer, streaming SEANet

This diagnostic tests whether the batch vs streaming transformer difference
explains the Rust correlation gap (0.839 -> 1.0).

Usage:
    python validation/compare_mimi_modes.py
"""

import numpy as np
import torch
from scipy.io import wavfile
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps


def correlation(a, b):
    """Pearson correlation between two numpy arrays."""
    a, b = a.flatten(), b.flatten()
    ml = min(len(a), len(b))
    a, b = a[:ml], b[:ml]
    return float(np.corrcoef(a, b)[0, 1])


def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    print("Loading model...")
    model = TTSModel.load_model()
    mimi = model.mimi

    latents_np = np.load("validation/reference_outputs/phrase_00_latents.npy")
    latents = torch.from_numpy(latents_np).unsqueeze(0)  # [1, 47, 32]
    num_frames = latents.shape[1]
    print(f"Latents: {latents.shape}")

    emb_mean = model.flow_lm.emb_mean.detach()
    emb_std = model.flow_lm.emb_std.detach()

    # ==================================================================
    # MODE 1: Full Python streaming (reference behavior)
    # ==================================================================
    print("\n=== MODE 1: Full streaming ===")
    mimi_context = model.config.mimi.transformer.context
    state1 = init_states(mimi, batch_size=1, sequence_length=mimi_context)

    streaming_chunks = []
    # Also capture per-frame transformer outputs for comparison
    streaming_tr_outputs = []

    with torch.no_grad():
        for f in range(num_frames):
            frame = latents[:, f:f+1, :]  # [1, 1, 32]
            denorm = frame * emb_std + emb_mean
            transposed = denorm.transpose(-1, -2)  # [1, 32, 1]
            quantized = mimi.quantizer(transposed)  # [1, 512, 1]
            upsampled = mimi._to_encoder_framerate(quantized, state1)
            dec_tr_out = mimi.decoder_transformer(upsampled, state1)
            dec_tr_out = dec_tr_out[0]
            streaming_tr_outputs.append(dec_tr_out.clone())
            audio = mimi.decoder(dec_tr_out, state1)
            streaming_chunks.append(audio)
            increment_steps(mimi, state1, increment=16)

    streaming_audio = torch.cat(streaming_chunks, dim=-1)[0, 0].numpy()
    streaming_tr_all = torch.cat(streaming_tr_outputs, dim=-1)  # [1, 512, 47*16]
    print(f"Streaming audio: {len(streaming_audio)} samples, max={np.max(np.abs(streaming_audio)):.4f}")
    print(f"Streaming transformer output: {streaming_tr_all.shape}")

    # ==================================================================
    # MODE 2: Streaming upsample + BATCH transformer + streaming SEANet
    # ==================================================================
    print("\n=== MODE 2: Batch transformer (Rust-like) ===")
    state2_up = init_states(mimi, batch_size=1, sequence_length=mimi_context)

    with torch.no_grad():
        # Step 1: Batch output_proj
        all_denorm = latents * emb_std + emb_mean
        all_transposed = all_denorm.transpose(-1, -2)
        all_quantized = mimi.quantizer(all_transposed)
        print(f"  output_proj: {all_quantized.shape}")

        # Step 2: Streaming upsample (one frame at a time)
        up_chunks = []
        for f in range(num_frames):
            frame_q = all_quantized[:, :, f:f+1]
            up = mimi._to_encoder_framerate(frame_q, state2_up)
            up_chunks.append(up)
            increment_steps(mimi, state2_up, increment=16)
        all_upsampled = torch.cat(up_chunks, dim=-1)
        print(f"  upsample: {all_upsampled.shape}")

        # Compare upsample outputs with Mode 1
        # Mode 1 upsample is embedded in the full streaming path,
        # so we collect it separately
        state1_up = init_states(mimi, batch_size=1, sequence_length=mimi_context)
        streaming_up_chunks = []
        for f in range(num_frames):
            frame = latents[:, f:f+1, :]
            denorm = frame * emb_std + emb_mean
            transposed = denorm.transpose(-1, -2)
            quantized = mimi.quantizer(transposed)
            up = mimi._to_encoder_framerate(quantized, state1_up)
            streaming_up_chunks.append(up)
            increment_steps(mimi, state1_up, increment=16)
        streaming_upsampled = torch.cat(streaming_up_chunks, dim=-1)

        up_cos = cosine_similarity(all_upsampled.numpy(), streaming_upsampled.numpy())
        up_rmse = np.sqrt(np.mean((all_upsampled.numpy() - streaming_upsampled.numpy()) ** 2))
        print(f"  Upsample match: cos_sim={up_cos:.6f}, rmse={up_rmse:.6f}")

        # Step 3: BATCH transformer (full sequence at once)
        state2_tr = init_states(mimi, batch_size=1, sequence_length=all_upsampled.shape[-1])
        batch_tr_out = mimi.decoder_transformer(all_upsampled, state2_tr)
        batch_tr_out = batch_tr_out[0]
        print(f"  batch transformer: {batch_tr_out.shape}")

        # Compare transformer outputs
        tr_cos = cosine_similarity(batch_tr_out.numpy(), streaming_tr_all.numpy())
        tr_rmse = np.sqrt(np.mean((batch_tr_out.numpy() - streaming_tr_all.numpy()) ** 2))
        print(f"  Transformer match: cos_sim={tr_cos:.6f}, rmse={tr_rmse:.6f}")

        # Per-frame transformer comparison
        print("\n  Per-frame transformer comparison:")
        for f in range(min(10, num_frames)):
            s_start = f * 16
            s_end = (f + 1) * 16
            s_frame = streaming_tr_all[:, :, s_start:s_end].numpy()
            b_frame = batch_tr_out[:, :, s_start:s_end].numpy()
            cos = cosine_similarity(s_frame, b_frame)
            rmse = np.sqrt(np.mean((s_frame - b_frame) ** 2))
            print(f"    Frame {f:2d}: cos_sim={cos:.6f}, rmse={rmse:.6f}")

        # Step 4: Streaming SEANet (chunks of 16)
        state2_sn = init_states(mimi, batch_size=1, sequence_length=mimi_context)
        sn_chunks = []
        for f in range(num_frames):
            chunk = batch_tr_out[:, :, f*16:(f+1)*16]
            audio = mimi.decoder(chunk, state2_sn)
            sn_chunks.append(audio)
            increment_steps(mimi, state2_sn, increment=16)

        hybrid_audio = torch.cat(sn_chunks, dim=-1)[0, 0].numpy()
        print(f"\n  Hybrid audio: {len(hybrid_audio)} samples, max={np.max(np.abs(hybrid_audio)):.4f}")

    # ==================================================================
    # CORRELATIONS
    # ==================================================================
    print("\n" + "=" * 60)
    print("CORRELATION RESULTS")
    print("=" * 60)

    # Load reference
    _, ref_wav = wavfile.read("validation/reference_outputs/phrase_00.wav")
    if ref_wav.dtype != np.float32:
        ref_wav = ref_wav.astype(np.float32) / 32768.0

    # Load Rust if available
    try:
        _, rust_wav = wavfile.read("/tmp/optimize-eval/phrase_00_rust.wav")
        if rust_wav.dtype != np.float32:
            rust_wav = rust_wav.astype(np.float32) / 32768.0
        has_rust = True
    except:
        has_rust = False

    print(f"\nStreaming vs Reference:      r = {correlation(streaming_audio, ref_wav):.6f}")
    print(f"Hybrid vs Reference:        r = {correlation(hybrid_audio, ref_wav):.6f}")
    print(f"Streaming vs Hybrid:        r = {correlation(streaming_audio, hybrid_audio):.6f}")
    if has_rust:
        print(f"Rust vs Reference:          r = {correlation(rust_wav, ref_wav):.6f}")
        print(f"Rust vs Hybrid:             r = {correlation(rust_wav, hybrid_audio):.6f}")
        print(f"Rust vs Streaming:          r = {correlation(rust_wav, streaming_audio):.6f}")

    # ==================================================================
    # DIAGNOSIS
    # ==================================================================
    print("\n--- DIAGNOSIS ---")
    corr_hybrid_ref = correlation(hybrid_audio, ref_wav)

    if corr_hybrid_ref < 0.95:
        gap = 1.0 - corr_hybrid_ref
        print(f"BATCH TRANSFORMER DIVERGES: r={corr_hybrid_ref:.4f}, gap={gap:.4f}")
        print("  -> The batch-vs-streaming transformer difference CAUSES correlation loss!")
        print("  -> FIX: Change Rust to process transformer one frame at a time with KV cache")

        # How much does the transformer explain?
        if has_rust:
            rust_gap = 1.0 - correlation(rust_wav, ref_wav)
            print(f"  -> Rust gap: {rust_gap:.4f}")
            print(f"  -> Transformer gap explains: {gap/rust_gap*100:.1f}% of Rust gap")
    else:
        print(f"Batch transformer matches streaming: r={corr_hybrid_ref:.4f}")
        print("  -> Transformer mode is NOT the issue")
        print("  -> Problem is in upsample overlap-add or SEANet streaming logic")


if __name__ == "__main__":
    main()
