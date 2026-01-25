#!/usr/bin/env python3
"""Generate Python reference output using zeros (same as Rust for comparison)."""

import torch
import numpy as np
from pathlib import Path

# Monkey-patch to use zeros (only for noise, not for model init)
_orig_normal = torch.nn.init.normal_
_orig_trunc = torch.nn.init.trunc_normal_
def zeros_normal_(t, mean=0, std=1):
    if not t.requires_grad:
        with torch.no_grad():
            t.zero_()
    else:
        _orig_normal(t, mean, std)
    return t
def zeros_trunc_(t, mean=0, std=1, a=-2, b=2):
    if not t.requires_grad:
        with torch.no_grad():
            t.zero_()
    else:
        _orig_trunc(t, mean, std, a, b)
    return t
torch.nn.init.normal_ = zeros_normal_
torch.nn.init.trunc_normal_ = zeros_trunc_

from pocket_tts import TTSModel

def main():
    output_dir = Path(__file__).parent / "reference_zeros"
    output_dir.mkdir(exist_ok=True)

    print("Loading model...")
    model = TTSModel.load_model()

    print("\nGetting voice state...")
    voice_state = model.get_state_for_audio_prompt("alba")

    text = "Hello, this is a test of the Pocket TTS system."
    print(f"\nGenerating audio for: '{text}'")
    audio = model.generate_audio(voice_state, text)
    audio_np = np.array(audio, dtype=np.float32)

    print(f"\nGenerated {len(audio_np)} samples")
    print(f"Stats: min={audio_np.min():.6f}, max={audio_np.max():.6f}, mean={audio_np.mean():.6f}")
    print(f"First 10: {audio_np[:10].tolist()}")

    # Save as numpy
    np.save(output_dir / "python_zeros.npy", audio_np)
    print(f"\nSaved to {output_dir / 'python_zeros.npy'}")

    # Also save as WAV
    import wave
    wav_path = output_dir / "python_zeros.wav"
    with wave.open(str(wav_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(24000)
        # Convert float32 to int16
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    print(f"Saved to {wav_path}")

if __name__ == "__main__":
    main()
