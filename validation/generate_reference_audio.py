#!/usr/bin/env python3
"""
Generate reference audio files for iOS app AB testing.
Uses the Python Pocket TTS implementation to create known-good reference audio.
"""

import os
import sys
import torch
import numpy as np
from safetensors.torch import load_file
import scipy.io.wavfile as wavfile
from pathlib import Path

# Model path
MODEL_PATH = "/Users/ramerman/dev/unamentis/models/kyutai-pocket-ios/model.safetensors"

# Reference test phrases (matching various lengths)
REFERENCE_PHRASES = [
    {
        "id": "short",
        "text": "Hello world.",
        "description": "Short phrase for quick testing"
    },
    {
        "id": "medium",
        "text": "The quick brown fox jumps over the lazy dog.",
        "description": "Medium phrase with all letters"
    },
    {
        "id": "long",
        "text": "This is a longer test sentence to verify that the text to speech system handles extended input correctly and produces natural sounding audio.",
        "description": "Long phrase for extended testing"
    }
]

# Load model weights
print(f"Loading model from {MODEL_PATH}...")
weights = load_file(MODEL_PATH)

def layer_norm(x, weight, bias, eps=1e-5):
    """Standard layer norm"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return weight * (x - mean) / torch.sqrt(var + eps) + bias

def rotary_embedding(seq_len, head_dim, base=10000.0):
    position = torch.arange(seq_len).unsqueeze(1)
    freq_indices = torch.arange(0, head_dim, 2)
    freqs = 1.0 / (base ** (freq_indices / head_dim))
    angles = position * freqs
    return torch.sin(angles), torch.cos(angles)

def apply_rope(q, k, sin, cos):
    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot

def transformer_layer(x, layer_idx):
    """Decoder transformer layer"""
    prefix = f'mimi.decoder_transformer.transformer.layers.{layer_idx}'
    batch, seq, dim = x.shape
    num_heads, head_dim = 8, 64

    h = layer_norm(x,
        weights[f'{prefix}.norm1.weight'].float(),
        weights[f'{prefix}.norm1.bias'].float())

    qkv = torch.nn.functional.linear(h, weights[f'{prefix}.self_attn.in_proj.weight'].float())
    qkv = qkv.reshape(batch, seq, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    sin, cos = rotary_embedding(seq, head_dim)
    q, k = apply_rope(q, k, sin, cos)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    scale = head_dim ** 0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn = torch.softmax(attn, dim=-1)
    attn_out = torch.matmul(attn, v)

    attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch, seq, dim)
    attn_out = torch.nn.functional.linear(attn_out, weights[f'{prefix}.self_attn.out_proj.weight'].float())

    layer_scale_1 = weights[f'{prefix}.layer_scale_1.scale'].float()
    x = x + attn_out * layer_scale_1

    h = layer_norm(x,
        weights[f'{prefix}.norm2.weight'].float(),
        weights[f'{prefix}.norm2.bias'].float())
    h = torch.nn.functional.linear(h, weights[f'{prefix}.linear1.weight'].float())
    h = torch.nn.functional.gelu(h)
    h = torch.nn.functional.linear(h, weights[f'{prefix}.linear2.weight'].float())

    layer_scale_2 = weights[f'{prefix}.layer_scale_2.scale'].float()
    x = x + h * layer_scale_2

    return x

def causal_conv(x, weight, bias=None, dilation=1):
    """Causal conv: pad left only"""
    kernel_size = weight.shape[2]
    pad_size = (kernel_size - 1) * dilation
    x_padded = torch.nn.functional.pad(x, (pad_size, 0))
    return torch.nn.functional.conv1d(x_padded, weight, bias=bias, dilation=dilation)

def conv_transpose(x, weight, bias, stride):
    """Standard conv transpose"""
    return torch.nn.functional.conv_transpose1d(x, weight, bias=bias, stride=stride)

def mimi_decode(latents):
    """
    Full Mimi decoder: latents [batch, frames, 32] -> audio [batch, samples]
    """
    # Step 1: Transpose to [batch, 32, frames]
    x = latents.permute(0, 2, 1)

    # Step 2: output_proj - Conv1d(32, 512, kernel=1)
    output_proj_w = weights['mimi.quantizer.output_proj.weight'].float()
    output_proj_b = weights.get('mimi.quantizer.output_proj.bias')
    x = torch.nn.functional.conv1d(x, output_proj_w, bias=output_proj_b.float() if output_proj_b else None)

    # Step 3: Streaming upsample - ConvTranspose1d(512, 512, k=32, s=16, groups=512)
    upsample_w = weights['mimi.upsample.convtr.convtr.weight'].float()
    batch, channels, seq = x.shape
    stride, kernel = 16, 32
    overlap = kernel - stride

    output_chunks = []
    buffer = torch.zeros(batch, channels, overlap)
    for frame_idx in range(seq):
        frame = x[:, :, frame_idx:frame_idx+1]
        out = torch.nn.functional.conv_transpose1d(frame, upsample_w, stride=stride, padding=0, groups=512)
        out[:, :, :overlap] += buffer
        buffer = out[:, :, -overlap:].clone()
        output_chunks.append(out[:, :, :stride])
    x = torch.cat(output_chunks, dim=2)

    # Step 4: Transpose for transformer [batch, seq, dim]
    x = x.permute(0, 2, 1)

    # Step 5: Decoder Transformer (2 layers, NON-CAUSAL)
    for i in range(2):
        x = transformer_layer(x, i)

    # Step 6: Transpose for SEANet [batch, dim, seq]
    x = x.permute(0, 2, 1)

    # Step 7: SEANet decoder (full implementation)
    # Input conv
    x = causal_conv(x,
        weights['mimi.decoder.model.0.conv.weight'].float(),
        weights['mimi.decoder.model.0.conv.bias'].float())
    x = torch.nn.functional.elu(x, 1.0)

    # Block 1: ConvTr 512->256, ResBlock, ELU
    x = conv_transpose(x,
        weights['mimi.decoder.model.2.convtr.weight'].float(),
        weights['mimi.decoder.model.2.convtr.bias'].float(),
        stride=6)
    res = x
    h = torch.nn.functional.elu(res, 1.0)
    h = causal_conv(h,
        weights['mimi.decoder.model.3.block.1.conv.weight'].float(),
        weights['mimi.decoder.model.3.block.1.conv.bias'].float())
    h = torch.nn.functional.elu(h, 1.0)
    h = torch.nn.functional.conv1d(h,
        weights['mimi.decoder.model.3.block.3.conv.weight'].float(),
        weights['mimi.decoder.model.3.block.3.conv.bias'].float())
    x = res + h
    x = torch.nn.functional.elu(x, 1.0)

    # Block 2: ConvTr 256->128, ResBlock, ELU
    x = conv_transpose(x,
        weights['mimi.decoder.model.5.convtr.weight'].float(),
        weights['mimi.decoder.model.5.convtr.bias'].float(),
        stride=5)
    res = x
    h = torch.nn.functional.elu(res, 1.0)
    h = causal_conv(h,
        weights['mimi.decoder.model.6.block.1.conv.weight'].float(),
        weights['mimi.decoder.model.6.block.1.conv.bias'].float())
    h = torch.nn.functional.elu(h, 1.0)
    h = torch.nn.functional.conv1d(h,
        weights['mimi.decoder.model.6.block.3.conv.weight'].float(),
        weights['mimi.decoder.model.6.block.3.conv.bias'].float())
    x = res + h
    x = torch.nn.functional.elu(x, 1.0)

    # Block 3: ConvTr 128->64, ResBlock, ELU
    x = conv_transpose(x,
        weights['mimi.decoder.model.8.convtr.weight'].float(),
        weights['mimi.decoder.model.8.convtr.bias'].float(),
        stride=4)
    res = x
    h = torch.nn.functional.elu(res, 1.0)
    h = causal_conv(h,
        weights['mimi.decoder.model.9.block.1.conv.weight'].float(),
        weights['mimi.decoder.model.9.block.1.conv.bias'].float())
    h = torch.nn.functional.elu(h, 1.0)
    h = torch.nn.functional.conv1d(h,
        weights['mimi.decoder.model.9.block.3.conv.weight'].float(),
        weights['mimi.decoder.model.9.block.3.conv.bias'].float())
    x = res + h
    x = torch.nn.functional.elu(x, 1.0)

    # Output conv
    x = causal_conv(x,
        weights['mimi.decoder.model.11.conv.weight'].float(),
        weights['mimi.decoder.model.11.conv.bias'].float())

    return x.squeeze(1)  # [batch, samples]

def generate_reference_audio(phrase_id, text, output_dir):
    """Generate reference audio for a phrase using deterministic latents"""
    print(f"\nGenerating reference for '{phrase_id}': {text}")

    # For reference testing, we use deterministic random latents
    # Same seed as Rust tests: 42
    # Frame count based on approximate text length (rough estimate: 1 frame per 2 chars)
    num_frames = max(4, len(text) // 2)

    torch.manual_seed(42)
    latents = torch.randn(1, num_frames, 32, dtype=torch.float32)

    print(f"  Latents shape: {latents.shape}")
    print(f"  Latents first 4: {latents[0, 0, :4].tolist()}")

    # Decode with Mimi
    with torch.no_grad():
        audio = mimi_decode(latents)

    audio_np = audio.squeeze().numpy()
    print(f"  Output samples: {len(audio_np)}")
    print(f"  Max amplitude: {np.abs(audio_np).max():.4f}")

    # Save as WAV
    output_path = output_dir / f"reference_{phrase_id}.wav"
    wavfile.write(str(output_path), 24000, audio_np.astype(np.float32))
    print(f"  Saved to: {output_path}")

    # Also save the latents for Rust comparison
    latents_path = output_dir / f"reference_{phrase_id}_latents.f32"
    latents_np = latents.squeeze(0).numpy()  # [frames, 32]
    with open(latents_path, 'wb') as f:
        f.write(latents_np.astype(np.float32).tobytes())
    print(f"  Latents saved to: {latents_path}")

    return output_path

def main():
    # Output directory for iOS app resources
    output_dir = Path("/Users/ramerman/dev/pocket-tts/tests/ios-harness/PocketTTSDemo/PocketTTSDemo/ReferenceAudio")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING REFERENCE AUDIO FOR iOS AB TESTING")
    print("=" * 60)

    # Generate manifest file
    manifest = {
        "sample_rate": 24000,
        "phrases": []
    }

    for phrase in REFERENCE_PHRASES:
        output_path = generate_reference_audio(phrase["id"], phrase["text"], output_dir)
        manifest["phrases"].append({
            "id": phrase["id"],
            "text": phrase["text"],
            "description": phrase["description"],
            "audio_file": f"reference_{phrase['id']}.wav",
            "latents_file": f"reference_{phrase['id']}_latents.f32"
        })

    # Write manifest
    import json
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")

    print("\n" + "=" * 60)
    print("REFERENCE AUDIO GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
