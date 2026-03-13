#!/usr/bin/env python3
"""
Composite Quality Scorer for TTS Autotuning

Reduces multiple quality metrics (WER, MCD, SNR, THD, correlation)
into a single scalar score in [0.0, 1.0] for the autoresearch loop.

Usage:
    python scorer.py --audio output.wav --text "Hello world" --reference ref.wav
    python scorer.py --metrics-json metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


# Weight configuration — controls relative importance of each metric
WEIGHTS = {
    "intelligibility": 0.40,  # WER-based (most important)
    "acoustic_similarity": 0.25,  # MCD-based
    "signal_quality": 0.15,  # SNR-based
    "correlation": 0.10,  # Waveform correlation to reference
    "distortion": 0.10,  # THD-based
}


def normalize_mcd(mcd: float) -> float:
    """
    Normalize MCD to [0, 1] where 1 = excellent.

    Uses MFCC Euclidean distance (not traditional mel-cepstral MCD).
    MCD < 30 → 1.0 (very similar spectra)
    MCD > 150 → 0.0 (very different spectra)
    Calibrated from cross-implementation TTS comparison (Rust vs Python).
    """
    return float(np.clip(1.0 - (mcd - 30.0) / 120.0, 0.0, 1.0))


def normalize_snr(snr_db: float) -> float:
    """
    Normalize SNR to [0, 1] where 1 = excellent.

    SNR > 35 dB → 1.0 (very clean)
    SNR < 15 dB → 0.0 (very noisy)
    Tightened from [5, 40] based on observed TTS range (23-28 dB).
    """
    return float(np.clip((snr_db - 15.0) / 20.0, 0.0, 1.0))


def normalize_thd(thd_percent: float) -> float:
    """
    Normalize THD to [0, 1] where 1 = low distortion (good).

    THD < 5% → 1.0 (negligible distortion)
    THD > 40% → 0.0 (severe distortion)
    Tightened from [1, 50] based on observed TTS range (25-55%).
    """
    return float(np.clip(1.0 - (thd_percent - 5.0) / 35.0, 0.0, 1.0))


def compute_composite_score(
    wer: Optional[float] = None,
    mcd: Optional[float] = None,
    snr_db: Optional[float] = None,
    thd_percent: Optional[float] = None,
    correlation: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute a single composite quality score from individual metrics.

    All inputs are optional; missing metrics are excluded and weights
    are renormalized over available metrics.

    Args:
        wer: Word Error Rate (0.0 = perfect, 1.0 = all wrong)
        mcd: Mel-Cepstral Distortion in dB (lower = better)
        snr_db: Signal-to-Noise Ratio in dB (higher = better)
        thd_percent: Total Harmonic Distortion % (lower = better)
        correlation: Pearson correlation to reference (1.0 = identical)

    Returns:
        Dictionary with composite score and per-component breakdown
    """
    components = {}
    active_weights = {}

    if wer is not None:
        components["intelligibility"] = 1.0 - min(wer, 1.0)
        active_weights["intelligibility"] = WEIGHTS["intelligibility"]

    if mcd is not None:
        components["acoustic_similarity"] = normalize_mcd(mcd)
        active_weights["acoustic_similarity"] = WEIGHTS["acoustic_similarity"]

    if snr_db is not None:
        components["signal_quality"] = normalize_snr(snr_db)
        active_weights["signal_quality"] = WEIGHTS["signal_quality"]

    if correlation is not None:
        components["correlation"] = max(0.0, correlation)
        active_weights["correlation"] = WEIGHTS["correlation"]

    if thd_percent is not None:
        components["distortion"] = normalize_thd(thd_percent)
        active_weights["distortion"] = WEIGHTS["distortion"]

    if not components:
        return {"composite_score": 0.0, "components": {}, "error": "no metrics available"}

    # Renormalize weights to sum to 1.0 over available metrics
    total_weight = sum(active_weights.values())
    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

    # Weighted sum
    composite = sum(
        normalized_weights[key] * components[key]
        for key in components
    )

    return {
        "composite_score": float(composite),
        "components": {k: float(v) for k, v in components.items()},
        "weights_used": {k: float(v) for k, v in normalized_weights.items()},
        "status": _classify_score(composite),
    }


def _classify_score(score: float) -> str:
    """Classify composite score into quality tier."""
    if score >= 0.90:
        return "excellent"
    elif score >= 0.75:
        return "good"
    elif score >= 0.60:
        return "acceptable"
    else:
        return "poor"


def score_from_metrics_dict(metrics: Dict) -> Dict[str, float]:
    """
    Extract individual metrics from a quality_metrics.py output dict
    and compute composite score.
    """
    wer = None
    mcd = None
    snr_db = None
    thd_percent = None
    correlation = None

    if "wer" in metrics and isinstance(metrics["wer"], dict) and "wer" in metrics["wer"]:
        wer = metrics["wer"]["wer"]

    if "mcd" in metrics and isinstance(metrics["mcd"], dict) and "mcd" in metrics["mcd"]:
        mcd = metrics["mcd"]["mcd"]

    if "snr" in metrics and isinstance(metrics["snr"], dict) and "snr_db" in metrics["snr"]:
        snr_db = metrics["snr"]["snr_db"]

    if "thd" in metrics and isinstance(metrics["thd"], dict) and "thd_percent" in metrics["thd"]:
        thd_percent = metrics["thd"]["thd_percent"]

    if "correlation" in metrics:
        correlation = metrics["correlation"]

    return compute_composite_score(
        wer=wer,
        mcd=mcd,
        snr_db=snr_db,
        thd_percent=thd_percent,
        correlation=correlation,
    )


def main():
    parser = argparse.ArgumentParser(description="Composite quality scorer for TTS autotuning")
    parser.add_argument("--metrics-json", help="Path to metrics JSON from quality_metrics.py")
    parser.add_argument("--wer", type=float, help="Word Error Rate (0-1)")
    parser.add_argument("--mcd", type=float, help="Mel-Cepstral Distortion (dB)")
    parser.add_argument("--snr", type=float, help="Signal-to-Noise Ratio (dB)")
    parser.add_argument("--thd", type=float, help="Total Harmonic Distortion (percent)")
    parser.add_argument("--correlation", type=float, help="Pearson correlation (0-1)")
    parser.add_argument("--output-json", help="Save score to JSON file")

    args = parser.parse_args()

    if args.metrics_json:
        with open(args.metrics_json) as f:
            metrics = json.load(f)
        result = score_from_metrics_dict(metrics)
    else:
        result = compute_composite_score(
            wer=args.wer,
            mcd=args.mcd,
            snr_db=args.snr,
            thd_percent=args.thd,
            correlation=args.correlation,
        )

    # Print report
    print(f"Composite Score: {result['composite_score']:.4f} ({result['status'].upper()})")
    print()
    print("Components:")
    for name, value in result.get("components", {}).items():
        weight = result.get("weights_used", {}).get(name, 0)
        contribution = weight * value
        print(f"  {name:25s}: {value:.4f} (weight: {weight:.2f}, contribution: {contribution:.4f})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {args.output_json}")


if __name__ == "__main__":
    main()
