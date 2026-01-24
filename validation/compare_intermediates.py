#!/usr/bin/env python3
"""
Compare intermediate outputs between Rust and Python implementations.

This script helps debug where the Rust implementation diverges from Python
by comparing intermediate tensors at key points in the model.

Comparison points:
1. Input embeddings (text + voice)
2. FlowLM attention outputs (per layer)
3. FlowNet outputs
4. Mimi decoder inputs/outputs
5. Final audio waveform

Usage:
    python compare_intermediates.py rust_outputs/ python_outputs/
    python compare_intermediates.py --rust latents.npy --python ref_latents.npy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Thresholds for comparison
RTOL = 1e-3  # Relative tolerance
ATOL = 1e-4  # Absolute tolerance
COSINE_THRESHOLD = 0.99


def load_npy(path: Path) -> np.ndarray:
    """Load a numpy array from file."""
    return np.load(str(path))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute root mean squared error."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    # Handle different lengths
    min_len = min(len(a_flat), len(b_flat))
    a_flat = a_flat[:min_len]
    b_flat = b_flat[:min_len]

    return float(np.sqrt(np.mean((a_flat - b_flat) ** 2)))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute maximum absolute error."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    min_len = min(len(a_flat), len(b_flat))
    return float(np.max(np.abs(a_flat[:min_len] - b_flat[:min_len])))


def describe_array(arr: np.ndarray, name: str = "Array") -> None:
    """Print descriptive statistics for an array."""
    print(f"  {name}:")
    print(f"    Shape: {arr.shape}")
    print(f"    dtype: {arr.dtype}")
    print(f"    min: {arr.min():.6f}")
    print(f"    max: {arr.max():.6f}")
    print(f"    mean: {arr.mean():.6f}")
    print(f"    std: {arr.std():.6f}")

    nan_count = np.sum(np.isnan(arr))
    inf_count = np.sum(np.isinf(arr))
    if nan_count > 0:
        print(f"    NaN count: {nan_count}")
    if inf_count > 0:
        print(f"    Inf count: {inf_count}")


def compare_arrays(
    rust_arr: np.ndarray,
    python_arr: np.ndarray,
    name: str = "Tensor",
    verbose: bool = True
) -> dict:
    """Compare two arrays and return metrics."""

    result = {
        "name": name,
        "rust_shape": list(rust_arr.shape),
        "python_shape": list(python_arr.shape),
        "shape_match": rust_arr.shape == python_arr.shape,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing: {name}")
        print('='*60)

        describe_array(rust_arr, "Rust")
        describe_array(python_arr, "Python")

    # Check for NaN/Inf
    rust_has_nan = np.any(np.isnan(rust_arr))
    python_has_nan = np.any(np.isnan(python_arr))
    result["rust_has_nan"] = bool(rust_has_nan)
    result["python_has_nan"] = bool(python_has_nan)

    if rust_has_nan or python_has_nan:
        if verbose:
            print(f"\n  WARNING: NaN values detected!")
            print(f"    Rust has NaN: {rust_has_nan}")
            print(f"    Python has NaN: {python_has_nan}")

    # Compute metrics
    cos_sim = cosine_similarity(rust_arr, python_arr)
    rms_err = rmse(rust_arr, python_arr)
    max_err = max_abs_error(rust_arr, python_arr)

    result["cosine_similarity"] = cos_sim
    result["rmse"] = rms_err
    result["max_abs_error"] = max_err
    result["passed"] = cos_sim >= COSINE_THRESHOLD

    if verbose:
        print(f"\n  Comparison Metrics:")
        status = "PASS" if result["passed"] else "FAIL"
        symbol = "✓" if result["passed"] else "✗"
        print(f"    {symbol} Cosine similarity: {cos_sim:.6f} (threshold: {COSINE_THRESHOLD}) [{status}]")
        print(f"    RMSE: {rms_err:.6f}")
        print(f"    Max absolute error: {max_err:.6f}")

        # Show where the biggest differences are
        rust_flat = rust_arr.flatten()
        python_flat = python_arr.flatten()
        min_len = min(len(rust_flat), len(python_flat))

        diff = np.abs(rust_flat[:min_len] - python_flat[:min_len])
        top_indices = np.argsort(diff)[-5:][::-1]

        if max_err > ATOL:
            print(f"\n  Top 5 differences (index: rust vs python):")
            for idx in top_indices:
                print(f"    [{idx}]: {rust_flat[idx]:.6f} vs {python_flat[idx]:.6f} (diff: {diff[idx]:.6f})")

    return result


def compare_directories(
    rust_dir: Path,
    python_dir: Path,
    verbose: bool = True
) -> list:
    """Compare all matching .npy files in two directories."""

    results = []

    # Find common files
    rust_files = {f.name for f in rust_dir.glob("*.npy")}
    python_files = {f.name for f in python_dir.glob("*.npy")}

    common_files = rust_files & python_files

    if not common_files:
        print(f"No matching .npy files found between directories")
        print(f"  Rust files: {sorted(rust_files)}")
        print(f"  Python files: {sorted(python_files)}")
        return results

    print(f"Found {len(common_files)} matching files to compare")

    for filename in sorted(common_files):
        rust_arr = load_npy(rust_dir / filename)
        python_arr = load_npy(python_dir / filename)

        name = filename.replace(".npy", "")
        result = compare_arrays(rust_arr, python_arr, name, verbose)
        results.append(result)

    return results


def print_summary(results: list) -> bool:
    """Print summary of comparison results."""

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)

    print(f"\nPassed: {passed_count}/{total_count}")
    print("")

    # Sort by cosine similarity (worst first)
    sorted_results = sorted(results, key=lambda r: r["cosine_similarity"])

    for r in sorted_results:
        status = "✓" if r["passed"] else "✗"
        print(f"  {status} {r['name']}: cos_sim={r['cosine_similarity']:.4f}, rmse={r['rmse']:.6f}")

    all_passed = passed_count == total_count

    print("")
    if all_passed:
        print("All comparisons PASSED!")
    else:
        print("Some comparisons FAILED - see details above")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Compare intermediate outputs between Rust and Python implementations"
    )

    # Mode 1: Compare directories
    parser.add_argument(
        "rust_dir",
        type=Path,
        nargs="?",
        help="Directory containing Rust .npy outputs"
    )
    parser.add_argument(
        "python_dir",
        type=Path,
        nargs="?",
        help="Directory containing Python .npy outputs"
    )

    # Mode 2: Compare specific files
    parser.add_argument(
        "--rust",
        type=Path,
        help="Specific Rust .npy file to compare"
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Specific Python .npy file to compare"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary, not detailed comparisons"
    )

    parser.add_argument(
        "--json",
        type=Path,
        help="Output results to JSON file"
    )

    args = parser.parse_args()

    verbose = not args.quiet
    results = []

    if args.rust and args.python:
        # Mode 2: Compare specific files
        if not args.rust.exists():
            print(f"Error: Rust file not found: {args.rust}")
            sys.exit(1)
        if not args.python.exists():
            print(f"Error: Python file not found: {args.python}")
            sys.exit(1)

        rust_arr = load_npy(args.rust)
        python_arr = load_npy(args.python)

        result = compare_arrays(rust_arr, python_arr, "Comparison", verbose)
        results.append(result)

    elif args.rust_dir and args.python_dir:
        # Mode 1: Compare directories
        if not args.rust_dir.exists():
            print(f"Error: Rust directory not found: {args.rust_dir}")
            sys.exit(1)
        if not args.python_dir.exists():
            print(f"Error: Python directory not found: {args.python_dir}")
            sys.exit(1)

        results = compare_directories(args.rust_dir, args.python_dir, verbose)

    else:
        # Default: compare debug_outputs/ with itself or show help
        debug_dir = Path(__file__).parent / "debug_outputs"
        ref_dir = Path(__file__).parent / "reference_outputs"

        if debug_dir.exists() and ref_dir.exists():
            print("Comparing debug_outputs/ with reference_outputs/")
            results = compare_directories(debug_dir, ref_dir, verbose)
        else:
            parser.print_help()
            print("\n\nExample usage:")
            print("  python compare_intermediates.py rust_outputs/ python_outputs/")
            print("  python compare_intermediates.py --rust latents.npy --python ref_latents.npy")
            sys.exit(1)

    if not results:
        print("No results to summarize")
        sys.exit(1)

    all_passed = print_summary(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.json}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
