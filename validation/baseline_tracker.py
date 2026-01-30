#!/usr/bin/env python3
"""
Baseline Tracking and Regression Detection for Quality Metrics

Stores quality metric baselines and detects regressions across versions.

Usage:
    # Save current metrics as baseline
    python baseline_tracker.py --save baseline_v0.4.1.json --metrics results.json

    # Compare current metrics against baseline
    python baseline_tracker.py --compare --baseline baseline_v0.4.1.json --metrics results.json

    # Check for regressions (exits with code 1 if found)
    python baseline_tracker.py --check-regression --baseline baseline_v0.4.1.json --metrics results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BaselineTracker:
    """Tracks quality metric baselines and detects regressions."""

    # Threshold configuration
    # Format: {metric_path: (warning_threshold_pct, error_threshold_pct)}
    # Positive = worse is higher, Negative = worse is lower
    THRESHOLDS = {
        "wer.wer": (10.0, 20.0),  # WER increase of 10%/20% is warning/error
        "mcd.mcd": (10.0, 20.0),  # MCD increase of 10%/20%
        "snr.snr_db": (-10.0, -20.0),  # SNR decrease of 10%/20%
        "thd.thd_percent": (50.0, 100.0),  # THD increase of 50%/100%
        "amplitude_max": (-20.0, -40.0),  # Amplitude decrease of 20%/40%
        "rms": (-20.0, -40.0),  # RMS decrease of 20%/40%
        "spectral.spectral_centroid_hz": (20.0, 40.0),  # Centroid shift
        "spectral.spectral_flatness": (20.0, 40.0),  # Flatness change
    }

    def __init__(self, baselines_dir: Path = Path("validation/baselines")):
        """
        Initialize baseline tracker.

        Args:
            baselines_dir: Directory to store baseline files
        """
        self.baselines_dir = baselines_dir
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def save_baseline(self, metrics: Dict, baseline_path: Path,
                     version: Optional[str] = None,
                     git_commit: Optional[str] = None):
        """
        Save metrics as a baseline.

        Args:
            metrics: Quality metrics dictionary
            baseline_path: Path to save baseline
            version: Version string (e.g., "v0.4.1")
            git_commit: Git commit hash
        """
        baseline = {
            "version": version or "unknown",
            "git_commit": git_commit or "unknown",
            "date": datetime.now().isoformat(),
            "metrics": self._extract_comparable_metrics(metrics)
        }

        with open(baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)

        print(f"Baseline saved to: {baseline_path}")

    def load_baseline(self, baseline_path: Path) -> Dict:
        """Load baseline from file."""
        with open(baseline_path, 'r') as f:
            return json.load(f)

    def _extract_comparable_metrics(self, metrics: Dict) -> Dict:
        """Extract comparable metrics from full results."""
        comparable = {}

        # Basic metrics
        for key in ["amplitude_max", "rms", "duration_sec"]:
            if key in metrics:
                comparable[key] = metrics[key]

        # WER
        if "wer" in metrics and "error" not in metrics["wer"]:
            comparable["wer"] = {
                "wer": metrics["wer"]["wer"]
            }

        # MCD
        if "mcd" in metrics and "error" not in metrics["mcd"]:
            comparable["mcd"] = {
                "mcd": metrics["mcd"]["mcd"]
            }

        # SNR
        if "snr" in metrics and "error" not in metrics["snr"]:
            comparable["snr"] = {
                "snr_db": metrics["snr"]["snr_db"],
                "noise_floor": metrics["snr"]["noise_floor"]
            }

        # THD
        if "thd" in metrics and "error" not in metrics["thd"]:
            comparable["thd"] = {
                "thd_percent": metrics["thd"]["thd_percent"]
            }

        # Spectral
        if "spectral" in metrics and "error" not in metrics["spectral"]:
            comparable["spectral"] = metrics["spectral"]

        return comparable

    def _get_nested_value(self, data: Dict, path: str) -> Optional[float]:
        """Get value from nested dictionary using dot notation."""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def compare_metrics(self, baseline: Dict, current: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Compare current metrics against baseline.

        Returns:
            Tuple of (improvements, warnings, errors)
        """
        improvements = []
        warnings = []
        errors = []

        baseline_metrics = baseline.get("metrics", {})
        current_metrics = self._extract_comparable_metrics(current)

        for metric_path, (warn_threshold, error_threshold) in self.THRESHOLDS.items():
            baseline_val = self._get_nested_value(baseline_metrics, metric_path)
            current_val = self._get_nested_value(current_metrics, metric_path)

            if baseline_val is None or current_val is None:
                continue

            # Compute percent change
            if baseline_val == 0:
                if current_val == 0:
                    continue
                pct_change = 100.0  # Arbitrarily large
            else:
                pct_change = ((current_val - baseline_val) / abs(baseline_val)) * 100.0

            # Determine status
            # For negative thresholds (lower is worse), flip the comparison
            if warn_threshold < 0:
                # Lower is worse (e.g., SNR)
                if pct_change < error_threshold:
                    errors.append({
                        "metric": metric_path,
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_pct": pct_change,
                        "severity": "error"
                    })
                elif pct_change < warn_threshold:
                    warnings.append({
                        "metric": metric_path,
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_pct": pct_change,
                        "severity": "warning"
                    })
                elif pct_change > 0:
                    improvements.append({
                        "metric": metric_path,
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_pct": pct_change,
                        "severity": "improvement"
                    })
            else:
                # Higher is worse (e.g., WER, THD)
                if pct_change > error_threshold:
                    errors.append({
                        "metric": metric_path,
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_pct": pct_change,
                        "severity": "error"
                    })
                elif pct_change > warn_threshold:
                    warnings.append({
                        "metric": metric_path,
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_pct": pct_change,
                        "severity": "warning"
                    })
                elif pct_change < 0:
                    improvements.append({
                        "metric": metric_path,
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_pct": pct_change,
                        "severity": "improvement"
                    })

        return improvements, warnings, errors

    def print_comparison_report(self, baseline: Dict, current: Dict):
        """Print formatted comparison report."""
        improvements, warnings, errors = self.compare_metrics(baseline, current)

        print("=" * 70)
        print("BASELINE COMPARISON REPORT")
        print("=" * 70)
        print()

        print(f"Baseline Version: {baseline.get('version', 'unknown')}")
        print(f"Baseline Date:    {baseline.get('date', 'unknown')}")
        print(f"Baseline Commit:  {baseline.get('git_commit', 'unknown')}")
        print()

        if errors:
            print("🚨 ERRORS (Significant Regressions):")
            for item in errors:
                sign = "+" if item["change_pct"] > 0 else ""
                print(f"  ❌ {item['metric']}: {item['baseline']:.4f} → {item['current']:.4f} "
                      f"({sign}{item['change_pct']:.1f}%)")
            print()

        if warnings:
            print("⚠️  WARNINGS (Minor Regressions):")
            for item in warnings:
                sign = "+" if item["change_pct"] > 0 else ""
                print(f"  ⚠️  {item['metric']}: {item['baseline']:.4f} → {item['current']:.4f} "
                      f"({sign}{item['change_pct']:.1f}%)")
            print()

        if improvements:
            print("✅ IMPROVEMENTS:")
            for item in improvements:
                sign = "+" if item["change_pct"] > 0 else ""
                print(f"  ✅ {item['metric']}: {item['baseline']:.4f} → {item['current']:.4f} "
                      f"({sign}{item['change_pct']:.1f}%)")
            print()

        if not errors and not warnings:
            print("✅ No regressions detected - all metrics within acceptable ranges")
            print()

        # Summary
        print("SUMMARY:")
        print(f"  Improvements: {len(improvements)}")
        print(f"  Warnings:     {len(warnings)}")
        print(f"  Errors:       {len(errors)}")
        print()

        return len(errors) == 0 and len(warnings) == 0


def get_git_info() -> Tuple[Optional[str], Optional[str]]:
    """Get current git commit and version."""
    import subprocess

    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except:
        commit = None

    try:
        # Try to get version from git tag
        version = subprocess.check_output(['git', 'describe', '--tags', '--always'],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except:
        version = None

    return version, commit


def main():
    parser = argparse.ArgumentParser(description='Baseline tracking and regression detection')

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--save', metavar='FILE', help='Save metrics as baseline')
    mode_group.add_argument('--compare', action='store_true', help='Compare against baseline')
    mode_group.add_argument('--check-regression', action='store_true',
                           help='Check for regressions (exit 1 if found)')
    mode_group.add_argument('--update-baseline', action='store_true',
                           help='Update baseline with current metrics (for CI)')

    # Common arguments
    parser.add_argument('--metrics', help='Path to current metrics JSON file')
    parser.add_argument('--baseline', help='Path to baseline JSON file')
    parser.add_argument('--baselines-dir', default='validation/baselines',
                       help='Directory for baseline files')

    args = parser.parse_args()

    tracker = BaselineTracker(baselines_dir=Path(args.baselines_dir))

    # Save mode
    if args.save:
        if not args.metrics:
            print("Error: --metrics required for --save mode", file=sys.stderr)
            sys.exit(1)

        with open(args.metrics, 'r') as f:
            metrics = json.load(f)

        version, commit = get_git_info()
        tracker.save_baseline(metrics, Path(args.save), version, commit)

    # Compare mode
    elif args.compare or args.check_regression:
        if not args.baseline or not args.metrics:
            print("Error: --baseline and --metrics required", file=sys.stderr)
            sys.exit(1)

        baseline = tracker.load_baseline(Path(args.baseline))
        with open(args.metrics, 'r') as f:
            current = json.load(f)

        passed = tracker.print_comparison_report(baseline, current)

        if args.check_regression and not passed:
            sys.exit(1)

    # Update baseline mode (for CI)
    elif args.update_baseline:
        if not args.metrics:
            print("Error: --metrics required for --update-baseline mode", file=sys.stderr)
            sys.exit(1)

        with open(args.metrics, 'r') as f:
            metrics = json.load(f)

        version, commit = get_git_info()

        # Auto-generate baseline filename from version
        if version:
            baseline_file = tracker.baselines_dir / f"baseline_{version}.json"
        else:
            baseline_file = tracker.baselines_dir / "baseline_latest.json"

        tracker.save_baseline(metrics, baseline_file, version, commit)


if __name__ == '__main__':
    main()
