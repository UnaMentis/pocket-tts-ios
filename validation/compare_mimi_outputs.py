#!/usr/bin/env python3
"""
Compare Mimi decoder outputs between Python and Rust.

Loads the saved Python intermediates and compares with Rust audio output.
"""

import argparse
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal


def load_wav(path):
    """Load WAV file and return normalized float samples."""
    sample_rate, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    return sample_rate, audio


def compute_correlation(a, b):
    """Compute Pearson correlation between two signals."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def find_best_alignment(ref, test, max_shift=1000):
    """Find the best alignment between two signals using cross-correlation."""
    # Use smaller portion for correlation
    ref_portion = ref[:min(len(ref), 10000)]
    test_portion = test[:min(len(test), 10000)]

    correlation = signal.correlate(test_portion, ref_portion, mode='full')
    shift = np.argmax(correlation) - len(ref_portion) + 1
    return shift


def main():
    parser = argparse.ArgumentParser(description="Compare Mimi outputs")
    parser.add_argument("--python-audio", type=Path, default=Path("mimi_debug/final_audio.npy"))
    parser.add_argument("--rust-audio", type=Path, required=True)
    parser.add_argument("--detailed", action="store_true", help="Show detailed stats")
    args = parser.parse_args()

    print("=== Loading Python output ===")
    if args.python_audio.suffix == '.npy':
        python_audio = np.load(args.python_audio).flatten()
    else:
        _, python_audio = load_wav(args.python_audio)
    print(f"Python audio: {len(python_audio)} samples, max={np.max(np.abs(python_audio)):.4f}")

    print("\n=== Loading Rust output ===")
    if args.rust_audio.suffix == '.npy':
        rust_audio = np.load(args.rust_audio).flatten()
    else:
        _, rust_audio = load_wav(args.rust_audio)
    print(f"Rust audio: {len(rust_audio)} samples, max={np.max(np.abs(rust_audio)):.4f}")

    # Normalize both to same amplitude for fair comparison
    python_normalized = python_audio / (np.max(np.abs(python_audio)) + 1e-10)
    rust_normalized = rust_audio / (np.max(np.abs(rust_audio)) + 1e-10)

    # Truncate to same length
    min_len = min(len(python_normalized), len(rust_normalized))
    python_normalized = python_normalized[:min_len]
    rust_normalized = rust_normalized[:min_len]

    print(f"\n=== Comparison (first {min_len} samples) ===")

    # Direct correlation
    direct_corr = compute_correlation(python_normalized, rust_normalized)
    print(f"Direct correlation: {direct_corr:.4f}")

    # Find best alignment
    shift = find_best_alignment(python_normalized, rust_normalized)
    print(f"Best alignment shift: {shift} samples")

    # Aligned correlation
    if shift > 0:
        aligned_python = python_normalized[:-shift] if shift < len(python_normalized) else python_normalized
        aligned_rust = rust_normalized[shift:shift+len(aligned_python)]
    elif shift < 0:
        aligned_rust = rust_normalized[:-abs(shift)] if abs(shift) < len(rust_normalized) else rust_normalized
        aligned_python = python_normalized[abs(shift):abs(shift)+len(aligned_rust)]
    else:
        aligned_python = python_normalized
        aligned_rust = rust_normalized

    min_aligned_len = min(len(aligned_python), len(aligned_rust))
    aligned_python = aligned_python[:min_aligned_len]
    aligned_rust = aligned_rust[:min_aligned_len]

    aligned_corr = compute_correlation(aligned_python, aligned_rust)
    print(f"Aligned correlation: {aligned_corr:.4f}")

    # Compute MSE
    mse = np.mean((aligned_python - aligned_rust) ** 2)
    print(f"MSE (normalized): {mse:.6f}")

    # Sample-by-sample difference
    diff = np.abs(aligned_python - aligned_rust)
    print(f"Max abs difference: {np.max(diff):.4f}")
    print(f"Mean abs difference: {np.mean(diff):.4f}")

    if args.detailed:
        print("\n=== Sample-level comparison (first 100) ===")
        for i in range(min(100, len(aligned_python))):
            print(f"[{i:4d}] Python: {aligned_python[i]:+.6f}  Rust: {aligned_rust[i]:+.6f}  Diff: {diff[i]:.6f}")

    # Summary
    print("\n=== Summary ===")
    if aligned_corr > 0.95:
        print("✓ Waveforms match well (correlation > 0.95)")
    elif aligned_corr > 0.5:
        print("⚠ Waveforms partially match (correlation 0.5-0.95)")
    else:
        print("✗ Waveforms do not match (correlation < 0.5)")


if __name__ == "__main__":
    main()
