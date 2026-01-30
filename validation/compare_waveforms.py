#!/usr/bin/env python3
"""
Compare Rust output waveforms with Python reference waveforms.

This script provides consistent, reproducible waveform comparison for verification reports.
"""

import argparse
import numpy as np
from scipy.io import wavfile
import sys


def load_wav(filepath):
    """Load WAV file and normalize to float32."""
    rate, audio = wavfile.read(filepath)
    # Normalize based on data type
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.float32:
        pass  # Already float32
    else:
        raise ValueError(f"Unsupported audio dtype: {audio.dtype}")
    return rate, audio


def compute_stats(audio):
    """Compute basic audio statistics."""
    return {
        'samples': len(audio),
        'max_amplitude': float(np.max(np.abs(audio))),
        'rms': float(np.sqrt(np.mean(audio**2))),
        'dc_offset': float(np.mean(audio)),
    }


def compute_correlation(ref, rust, max_offset=5000, offset_step=100):
    """
    Compute correlation between two waveforms.
    
    Also attempts to find best alignment by trying different sample offsets.
    """
    min_len = min(len(ref), len(rust))
    
    # Direct correlation (no offset)
    direct_corr = np.corrcoef(ref[:min_len], rust[:min_len])[0, 1]
    
    # Try to find best alignment
    best_corr = direct_corr
    best_offset = 0
    
    for offset in range(-max_offset, max_offset, offset_step):
        if offset < 0:
            # Rust is ahead, shift reference forward
            ref_slice = ref[-offset:min_len]
            rust_slice = rust[:min_len+offset]
        elif offset > 0:
            # Reference is ahead, shift rust forward
            ref_slice = ref[:min_len-offset]
            rust_slice = rust[offset:min_len]
        else:
            continue
        
        if len(ref_slice) > 0 and len(rust_slice) > 0:
            corr = np.corrcoef(ref_slice, rust_slice)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_offset = offset
    
    return direct_corr, best_corr, best_offset


def main():
    parser = argparse.ArgumentParser(description='Compare Rust and Python waveforms')
    parser.add_argument('--reference', required=True, help='Path to reference WAV file')
    parser.add_argument('--rust', required=True, help='Path to Rust output WAV file')
    parser.add_argument('--max-offset', type=int, default=5000,
                       help='Maximum sample offset for alignment search (default: 5000)')
    parser.add_argument('--offset-step', type=int, default=100,
                       help='Step size for offset search (default: 100)')
    parser.add_argument('--quality-metrics', action='store_true',
                       help='Run comprehensive quality metrics (WER, MCD, SNR, THD)')
    parser.add_argument('--text', help='Reference text for WER calculation (requires --quality-metrics)')
    parser.add_argument('--output-json', help='Save all results to JSON file')
    
    args = parser.parse_args()
    
    # Load both files
    try:
        ref_rate, ref = load_wav(args.reference)
        rust_rate, rust = load_wav(args.rust)
    except Exception as e:
        print(f"Error loading audio files: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check sample rates match
    if ref_rate != rust_rate:
        print(f"WARNING: Sample rates differ (ref: {ref_rate} Hz, rust: {rust_rate} Hz)", 
              file=sys.stderr)
    
    # Compute statistics
    ref_stats = compute_stats(ref)
    rust_stats = compute_stats(rust)
    
    # Compute correlation
    direct_corr, best_corr, best_offset = compute_correlation(
        ref, rust, args.max_offset, args.offset_step
    )
    
    # Print results
    print("=" * 70)
    print("WAVEFORM COMPARISON RESULTS")
    print("=" * 70)
    print()
    
    print("REFERENCE STATS:")
    print(f"  Samples:       {ref_stats['samples']}")
    print(f"  Max amplitude: {ref_stats['max_amplitude']:.6f}")
    print(f"  RMS:           {ref_stats['rms']:.6f}")
    print(f"  DC offset:     {ref_stats['dc_offset']:.6f}")
    print()
    
    print("RUST STATS:")
    print(f"  Samples:       {rust_stats['samples']}")
    print(f"  Max amplitude: {rust_stats['max_amplitude']:.6f}")
    print(f"  RMS:           {rust_stats['rms']:.6f}")
    print(f"  DC offset:     {rust_stats['dc_offset']:.6f}")
    print()
    
    print("RATIOS (Rust/Reference):")
    if ref_stats['max_amplitude'] > 0:
        amp_ratio = (rust_stats['max_amplitude'] / ref_stats['max_amplitude']) * 100
        print(f"  Amplitude ratio: {amp_ratio:.1f}%")
    else:
        print(f"  Amplitude ratio: N/A (reference max amplitude is zero)")
    
    if ref_stats['rms'] > 0:
        rms_ratio = (rust_stats['rms'] / ref_stats['rms']) * 100
        print(f"  RMS ratio:       {rms_ratio:.1f}%")
    else:
        print(f"  RMS ratio:       N/A (reference RMS is zero)")
    print()
    
    print("CORRELATION:")
    print(f"  Direct (no offset):  {direct_corr:.6f}")
    print(f"  Best aligned:        {best_corr:.6f} (offset: {best_offset} samples)")
    print()
    
    # Interpret correlation
    if best_corr > 0.95:
        status = "EXCELLENT - Very high similarity"
    elif best_corr > 0.80:
        status = "GOOD - High similarity"
    elif best_corr > 0.50:
        status = "MODERATE - Some similarity"
    else:
        status = "LOW - Significant differences"
    
    print(f"STATUS: {status}")
    print()

    # Note about random noise
    if ref_stats['max_amplitude'] > 0.001:  # Reference has actual audio
        print("NOTE: If random noise is enabled in both implementations,")
        print("      correlation will naturally be low due to different RNG.")
        print("      Focus on amplitude/RMS ratios for quality validation.")
    else:
        print("WARNING: Reference file appears to have near-zero amplitude.")
        print("         This may indicate a test phrase mismatch or corrupted file.")
    print()

    # Prepare results dictionary
    results = {
        "reference_file": args.reference,
        "rust_file": args.rust,
        "reference_stats": ref_stats,
        "rust_stats": rust_stats,
        "correlation": {
            "direct": direct_corr,
            "best_aligned": best_corr,
            "best_offset": best_offset
        },
        "ratios": {
            "amplitude": amp_ratio if ref_stats['max_amplitude'] > 0 else None,
            "rms": rms_ratio if ref_stats['rms'] > 0 else None
        }
    }

    # Run quality metrics if requested
    if args.quality_metrics:
        try:
            from quality_metrics import QualityMetrics

            print("=" * 70)
            print("RUNNING COMPREHENSIVE QUALITY METRICS")
            print("=" * 70)
            print()

            metrics = QualityMetrics()
            quality_results = metrics.analyze_audio(
                args.rust,
                reference_text=args.text,
                reference_audio=args.reference
            )

            results["quality_metrics"] = quality_results

            # Print quality report
            from quality_metrics import print_report
            print_report(quality_results)

        except ImportError:
            print("WARNING: quality_metrics module not available. Install dependencies:", file=sys.stderr)
            print("  pip install openai-whisper librosa jiwer", file=sys.stderr)
        except Exception as e:
            print(f"ERROR running quality metrics: {e}", file=sys.stderr)

    # Save JSON if requested
    if args.output_json:
        import json
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_json}")


if __name__ == '__main__':
    main()
