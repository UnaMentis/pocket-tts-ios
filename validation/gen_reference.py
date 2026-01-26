#!/usr/bin/env python3
"""Generate Python reference output for verification."""

import numpy as np
from scipy.io import wavfile
from pocket_tts import TTSModel

def main():
    print("Loading model...")
    model = TTSModel.load_model()

    print("Getting voice state...")
    voice_state = model.get_state_for_audio_prompt("alba")

    text = "Hello, this is a test."
    print(f'Generating audio for: "{text}"')

    audio = model.generate_audio(voice_state, text)

    # Convert to numpy if it's a torch tensor
    if hasattr(audio, 'numpy'):
        audio = audio.numpy()
    audio = np.array(audio).astype(np.float32)

    print(f"Audio samples: {len(audio)}")
    print(f"Max amplitude: {np.abs(audio).max():.4f}")
    print(f"RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    # Save as WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write('/tmp/python_reference.wav', 24000, audio_int16)
    print("Saved to /tmp/python_reference.wav")

    # Also save raw numpy array for exact comparison
    np.save('/tmp/python_reference.npy', audio)
    print("Saved to /tmp/python_reference.npy")

if __name__ == "__main__":
    main()
