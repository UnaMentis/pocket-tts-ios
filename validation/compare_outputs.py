#!/usr/bin/env python3
"""Compare Rust and Python outputs for verification."""

import numpy as np
from scipy.io import wavfile
from scipy import signal

def main():
    # Load Python reference
    python_audio = np.load('/tmp/python_reference.npy')
    print(f"Python audio: {len(python_audio)} samples, max={np.abs(python_audio).max():.4f}")

    # Load Rust output
    _, rust_audio = wavfile.read('/tmp/rust_output.wav')
    rust_audio = rust_audio.astype(np.float32) / 32768.0
    print(f"Rust audio: {len(rust_audio)} samples, max={np.abs(rust_audio).max():.4f}")

    # Direct correlation (unaligned)
    min_len = min(len(python_audio), len(rust_audio))
    direct_corr = np.corrcoef(python_audio[:min_len], rust_audio[:min_len])[0, 1]
    print(f"\nDirect correlation: {direct_corr:.6f}")

    # Find best alignment using cross-correlation
    if len(python_audio) > 1000 and len(rust_audio) > 1000:
        correlation = signal.correlate(rust_audio, python_audio, mode='full')
        lags = signal.correlation_lags(len(rust_audio), len(python_audio), mode='full')
        best_lag = lags[np.argmax(correlation)]

        # Align and compute correlation
        if best_lag > 0:
            aligned_python = python_audio[best_lag:]
            aligned_rust = rust_audio
        else:
            aligned_python = python_audio
            aligned_rust = rust_audio[-best_lag:]

        aligned_len = min(len(aligned_python), len(aligned_rust))
        aligned_corr = np.corrcoef(aligned_python[:aligned_len], aligned_rust[:aligned_len])[0, 1]

        print(f"Best alignment shift: {best_lag} samples")
        print(f"Aligned correlation: {aligned_corr:.6f}")
    else:
        aligned_corr = direct_corr
        best_lag = 0

    # Compute RMS levels
    python_rms = np.sqrt(np.mean(python_audio**2))
    rust_rms = np.sqrt(np.mean(rust_audio**2))
    print(f"\nPython RMS: {python_rms:.4f}")
    print(f"Rust RMS: {rust_rms:.4f}")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Sample count - Python: {len(python_audio)}, Rust: {len(rust_audio)}")
    print(f"Max amplitude - Python: {np.abs(python_audio).max():.4f}, Rust: {np.abs(rust_audio).max():.4f}")
    print(f"RMS - Python: {python_rms:.4f}, Rust: {rust_rms:.4f}")
    print(f"Direct correlation: {direct_corr:.6f}")
    print(f"Aligned correlation: {aligned_corr:.6f}")
    print(f"Alignment shift: {best_lag} samples")

if __name__ == "__main__":
    main()
