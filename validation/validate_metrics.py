#!/usr/bin/env python3
"""
Meta-Validation: Test the Quality Metrics Themselves

Before trusting quality metrics for regression detection, we need to validate
that the metrics are implemented correctly. This script tests the quality
metrics implementation against known cases.

Usage:
    python validate_metrics.py
"""

import argparse
import sys
import numpy as np
from scipy.io import wavfile
from pathlib import Path

# Try to import quality metrics
try:
    from quality_metrics import QualityMetrics
except ImportError:
    print("Error: quality_metrics module not found", file=sys.stderr)
    sys.exit(1)


class MetricsValidator:
    """Validates that quality metrics behave correctly."""

    def __init__(self):
        self.metrics = QualityMetrics(whisper_model="tiny")  # Use tiny for speed
        self.tests_passed = 0
        self.tests_failed = 0
        self.sample_rate = 24000

    def print_test(self, name: str, passed: bool, details: str = ""):
        """Print test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if details:
            print(f"        {details}")

        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1

    def generate_test_audio(self, duration_sec: float = 1.0,
                          frequency: float = 440.0,
                          noise_level: float = 0.0) -> np.ndarray:
        """Generate synthetic test audio."""
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec))

        # Pure tone
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)

        # Add noise if requested
        if noise_level > 0:
            audio += noise_level * np.random.randn(len(audio))

        return audio.astype(np.float32)

    def save_test_audio(self, audio: np.ndarray, filename: str):
        """Save test audio to file."""
        output_dir = Path("validation/test_audio")
        output_dir.mkdir(exist_ok=True)

        # Convert to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(output_dir / filename, self.sample_rate, audio_int16)

        return str(output_dir / filename)

    def test_mcd_identical_audio(self):
        """Test: MCD should be ~0 for identical audio."""
        print("\n[TEST] MCD on identical audio")

        audio = self.generate_test_audio(duration_sec=2.0)
        result = self.metrics.compute_mcd(audio, audio.copy())

        mcd = result["mcd"]
        passed = mcd < 0.1  # Should be very close to 0

        self.print_test(
            "MCD(audio, audio) ≈ 0",
            passed,
            f"MCD = {mcd:.4f} dB (expected <0.1 dB)"
        )

    def test_mcd_very_different_audio(self):
        """Test: MCD should be high for very different audio."""
        print("\n[TEST] MCD on very different audio")

        audio1 = self.generate_test_audio(duration_sec=2.0, frequency=200)
        audio2 = self.generate_test_audio(duration_sec=2.0, frequency=2000)

        result = self.metrics.compute_mcd(audio1, audio2)
        mcd = result["mcd"]

        passed = mcd > 10.0  # Should be significantly different

        self.print_test(
            "MCD(200Hz, 2000Hz) >> 0",
            passed,
            f"MCD = {mcd:.2f} dB (expected >10 dB)"
        )

    def test_snr_clean_signal(self):
        """Test: SNR should be very high for clean signal."""
        print("\n[TEST] SNR on clean signal")

        audio = self.generate_test_audio(duration_sec=2.0, noise_level=0.0)
        result = self.metrics.compute_snr(audio)

        snr = result["snr_db"]
        passed = snr > 50.0  # Clean signal should have very high SNR

        self.print_test(
            "SNR(clean signal) > 50 dB",
            passed,
            f"SNR = {snr:.1f} dB (expected >50 dB)"
        )

    def test_snr_noisy_signal(self):
        """Test: SNR should be low for noisy signal."""
        print("\n[TEST] SNR on noisy signal")

        audio = self.generate_test_audio(duration_sec=2.0, noise_level=0.3)
        result = self.metrics.compute_snr(audio)

        snr = result["snr_db"]
        passed = snr < 15.0  # Noisy signal should have low SNR

        self.print_test(
            "SNR(noisy signal) < 15 dB",
            passed,
            f"SNR = {snr:.1f} dB (expected <15 dB)"
        )

    def test_thd_pure_tone(self):
        """Test: THD should be low for pure tone."""
        print("\n[TEST] THD on pure tone")

        audio = self.generate_test_audio(duration_sec=2.0, frequency=440)
        result = self.metrics.compute_thd(audio, fundamental_freq=440.0)

        thd = result["thd_percent"]
        passed = thd < 5.0  # Pure tone should have very low THD

        self.print_test(
            "THD(pure tone) < 5%",
            passed,
            f"THD = {thd:.2f}% (expected <5%)"
        )

    def test_wer_perfect_transcription(self):
        """Test: WER should be 0 for identical text."""
        print("\n[TEST] WER on identical text")

        # This tests the WER calculation, not Whisper
        # We'll use the jiwer library directly
        from jiwer import wer

        reference = "Hello this is a test"
        hypothesis = "Hello this is a test"

        error_rate = wer(reference, hypothesis)
        passed = error_rate == 0.0

        self.print_test(
            "WER(identical) = 0%",
            passed,
            f"WER = {error_rate*100:.1f}% (expected 0%)"
        )

    def test_wer_one_error(self):
        """Test: WER calculation with known error."""
        print("\n[TEST] WER with one substitution")

        from jiwer import wer

        reference = "Hello this is a test"
        hypothesis = "Hello this is the test"  # "a" → "the"

        error_rate = wer(reference, hypothesis)
        expected_wer = 1.0 / 5.0  # 1 error in 5 words = 20%

        passed = abs(error_rate - expected_wer) < 0.01

        self.print_test(
            "WER(1 error / 5 words) = 20%",
            passed,
            f"WER = {error_rate*100:.1f}% (expected 20%)"
        )

    def test_spectral_features_reasonable(self):
        """Test: Spectral features in reasonable ranges."""
        print("\n[TEST] Spectral features")

        # Speech-like signal: mix of frequencies
        audio = self.generate_test_audio(duration_sec=2.0, frequency=200)
        audio += 0.3 * self.generate_test_audio(duration_sec=2.0, frequency=800)
        audio += 0.2 * self.generate_test_audio(duration_sec=2.0, frequency=2000)

        result = self.metrics.compute_spectral_features(audio)

        centroid = result["spectral_centroid_hz"]
        rolloff = result["spectral_rolloff_hz"]
        flatness = result["spectral_flatness"]

        # Reasonable ranges for speech-like audio
        # Note: For low-frequency dominated signals, rolloff can be below 1000Hz
        centroid_ok = 100 < centroid < 5000
        rolloff_ok = 500 < rolloff < 10000  # Adjusted for low-freq signals
        flatness_ok = 0.0 < flatness < 1.0

        passed = centroid_ok and rolloff_ok and flatness_ok

        details = f"Centroid={centroid:.1f}Hz, Rolloff={rolloff:.1f}Hz, Flatness={flatness:.6f}"
        self.print_test(
            "Spectral features in reasonable ranges",
            passed,
            details
        )

    def test_metric_stability(self):
        """Test: Metrics should be stable across runs."""
        print("\n[TEST] Metric stability (3 runs)")

        audio = self.generate_test_audio(duration_sec=2.0)

        # Run MCD 3 times
        mcds = []
        for _ in range(3):
            result = self.metrics.compute_mcd(audio, audio.copy())
            mcds.append(result["mcd"])

        # Check variance is very low
        variance = np.var(mcds)
        passed = variance < 0.001

        self.print_test(
            "MCD variance < 0.001 across runs",
            passed,
            f"MCDs = {mcds}, variance = {variance:.6f}"
        )

    def test_amplitude_detection(self):
        """Test: Amplitude and RMS calculations."""
        print("\n[TEST] Amplitude and RMS calculations")

        # Known amplitude: 0.5
        audio = self.generate_test_audio(duration_sec=2.0)

        max_amp = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))

        # Pure sine at amplitude 0.5 should have:
        # - max = 0.5
        # - RMS = 0.5 / sqrt(2) ≈ 0.354

        amp_ok = abs(max_amp - 0.5) < 0.01
        rms_ok = abs(rms - (0.5 / np.sqrt(2))) < 0.01

        passed = amp_ok and rms_ok

        details = f"Max={max_amp:.4f} (expected 0.5), RMS={rms:.4f} (expected 0.354)"
        self.print_test(
            "Amplitude and RMS calculations correct",
            passed,
            details
        )

    def run_all_tests(self):
        """Run all validation tests."""
        print("=" * 70)
        print("META-VALIDATION: Testing Quality Metrics Implementation")
        print("=" * 70)

        # Test each metric
        self.test_mcd_identical_audio()
        self.test_mcd_very_different_audio()
        self.test_snr_clean_signal()
        self.test_snr_noisy_signal()
        self.test_thd_pure_tone()
        self.test_wer_perfect_transcription()
        self.test_wer_one_error()
        self.test_spectral_features_reasonable()
        self.test_metric_stability()
        self.test_amplitude_detection()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_failed}")
        print()

        if self.tests_failed == 0:
            print("✅ All tests passed - Quality metrics implementation validated!")
            print()
            print("Next steps:")
            print("  1. Run quality metrics on real TTS audio (Run 1)")
            print("  2. Cross-validate against Python reference (Run 2)")
            print("  3. Test stability across multiple runs (Run 3)")
            print("  4. Only then establish baseline")
            return 0
        else:
            print("❌ Some tests failed - Quality metrics need fixing!")
            print()
            print("DO NOT establish baseline until all tests pass.")
            return 1


def main():
    parser = argparse.ArgumentParser(description='Validate quality metrics implementation')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    validator = MetricsValidator()
    exit_code = validator.run_all_tests()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
