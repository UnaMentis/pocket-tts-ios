#!/usr/bin/env python3
"""
Compare quality metrics across multiple runs.

Used for Run 3 of iterative validation to check metric stability.

Usage:
    python compare_runs.py run1.json run2.json run3.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np


def load_results(json_files: List[str]) -> List[Dict]:
    """Load quality results from JSON files."""
    results = []
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)
            sys.exit(1)
    return results


def extract_metric(results: List[Dict], path: str, default=0.0) -> List[float]:
    """Extract a metric from nested dictionary using dot notation."""
    values = []
    for result in results:
        value = result
        for key in path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                value = default
                break
        values.append(float(value) if value != default else default)
    return values


def print_metric_stats(name: str, values: List[float], unit: str = "",
                      threshold_std: float = None, lower_better: bool = False):
    """Print statistics for a metric."""
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)

    print(f"\n{name}:")
    print(f"  Values: {[f'{v:.2f}{unit}' for v in values]}")
    print(f"  Mean:   {mean:.2f}{unit}")
    print(f"  Std:    {std:.2f}{unit}")
    print(f"  Range:  {min_val:.2f}{unit} - {max_val:.2f}{unit}")

    # Check stability
    if threshold_std is not None:
        if std < threshold_std:
            print(f"  ✅ Stable (std < {threshold_std}{unit})")
            return True
        else:
            print(f"  ❌ High variance (std >= {threshold_std}{unit})")
            return False
    return None


def main():
    parser = argparse.ArgumentParser(description='Compare quality metrics across multiple runs')
    parser.add_argument('json_files', nargs='+', help='Quality result JSON files')
    parser.add_argument('--verbose', action='store_true', help='Show all metrics')

    args = parser.parse_args()

    if len(args.json_files) < 2:
        print("Error: Need at least 2 result files to compare", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"COMPARING {len(args.json_files)} RUNS")
    print("=" * 70)

    results = load_results(args.json_files)

    # Track stability checks
    checks_passed = 0
    checks_failed = 0

    # Extract and compare key metrics
    print("\n" + "=" * 70)
    print("INTELLIGIBILITY")
    print("=" * 70)

    # WER (multiply by 100 for percentage)
    wer_values = [v * 100 for v in extract_metric(results, "wer.wer")]
    if any(v > 0 for v in wer_values):
        stable = print_metric_stats("WER", wer_values, "%", threshold_std=5.0)
        if stable:
            checks_passed += 1
        elif stable is False:
            checks_failed += 1

    print("\n" + "=" * 70)
    print("SIGNAL HEALTH")
    print("=" * 70)

    # SNR
    snr_values = extract_metric(results, "snr.snr_db")
    if any(v > 0 for v in snr_values):
        stable = print_metric_stats("SNR", snr_values, " dB", threshold_std=5.0)
        if stable:
            checks_passed += 1
        elif stable is False:
            checks_failed += 1

    # THD
    thd_values = extract_metric(results, "thd.thd_percent")
    if any(v > 0 for v in thd_values):
        stable = print_metric_stats("THD", thd_values, "%", threshold_std=1.0)
        if stable:
            checks_passed += 1
        elif stable is False:
            checks_failed += 1

    print("\n" + "=" * 70)
    print("BASIC METRICS")
    print("=" * 70)

    # Amplitude
    amp_values = extract_metric(results, "amplitude_max")
    if any(v > 0 for v in amp_values):
        stable = print_metric_stats("Max Amplitude", amp_values, "", threshold_std=0.05)
        if stable:
            checks_passed += 1
        elif stable is False:
            checks_failed += 1

    # RMS
    rms_values = extract_metric(results, "rms")
    if any(v > 0 for v in rms_values):
        stable = print_metric_stats("RMS", rms_values, "", threshold_std=0.02)
        if stable:
            checks_passed += 1
        elif stable is False:
            checks_failed += 1

    # Duration
    duration_values = extract_metric(results, "duration_sec")
    if any(v > 0 for v in duration_values):
        print_metric_stats("Duration", duration_values, " sec", threshold_std=0.1)

    if args.verbose:
        print("\n" + "=" * 70)
        print("SPECTRAL FEATURES")
        print("=" * 70)

        # Spectral features
        centroid_values = extract_metric(results, "spectral.spectral_centroid_hz")
        if any(v > 0 for v in centroid_values):
            print_metric_stats("Spectral Centroid", centroid_values, " Hz", threshold_std=200.0)

        rolloff_values = extract_metric(results, "spectral.spectral_rolloff_hz")
        if any(v > 0 for v in rolloff_values):
            print_metric_stats("Spectral Rolloff", rolloff_values, " Hz", threshold_std=500.0)

        flatness_values = extract_metric(results, "spectral.spectral_flatness")
        if any(v > 0 for v in flatness_values):
            print_metric_stats("Spectral Flatness", flatness_values, "", threshold_std=0.05)

    # Summary
    print("\n" + "=" * 70)
    print("STABILITY SUMMARY")
    print("=" * 70)
    print(f"Checks passed: {checks_passed}")
    print(f"Checks failed: {checks_failed}")
    print()

    if checks_failed == 0:
        print("✅ All metrics are stable across runs")
        print()
        print("Next step: Establish baseline with confidence")
        print("  ./validation/establish_baseline.sh")
        return 0
    else:
        print("❌ High variance detected in some metrics")
        print()
        print("Investigate before establishing baseline:")
        print("  1. Listen to all audio outputs")
        print("  2. Check for TTS instability")
        print("  3. Verify metric implementation")
        print("  4. Consider if variance is acceptable for this metric")
        return 1


if __name__ == '__main__':
    sys.exit(main())
