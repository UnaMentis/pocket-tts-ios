#!/usr/bin/env python3
"""
Audio Quality Metrics for TTS Validation

Provides objective, RNG-independent quality measurements:
- WER (Word Error Rate) - Intelligibility via Whisper ASR
- MCD (Mel-Cepstral Distortion) - Acoustic similarity
- SNR (Signal-to-Noise Ratio) - Signal quality
- THD (Total Harmonic Distortion) - Distortion measurement
- Spectral Features - Timbre and frequency characteristics

Usage:
    python quality_metrics.py --audio output.wav --text "Hello world" --reference ref.wav
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import whisper
from jiwer import wer
from scipy import signal
from scipy.io import wavfile


# ─── Text Normalization for WER ──────────────────────────────────────────────
# Whisper often transcribes numbers as digits ("1, 2, 3") while reference text
# uses words ("one two three"). Normalize both sides before comparison.

_DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}


def normalize_text_for_wer(text: str) -> str:
    """Normalize text for fair WER comparison.

    - Lowercases
    - Replaces digit strings with word equivalents
    - Strips punctuation
    - Collapses whitespace
    """
    text = text.lower()
    # Replace standalone digit sequences with word equivalents
    text = re.sub(
        r'\b(\d+)\b',
        lambda m: _DIGIT_TO_WORD.get(m.group(1), m.group(1)),
        text,
    )
    # Strip punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class QualityMetrics:
    """Comprehensive audio quality measurement suite."""

    def __init__(self, sample_rate: int = 24000, whisper_model: str = "base"):
        """
        Initialize quality metrics calculator.

        Args:
            sample_rate: Target sample rate for analysis
            whisper_model: Whisper model size (tiny, base, small, medium, large-v3)
        """
        self.sample_rate = sample_rate
        self.whisper_model_name = whisper_model
        self._whisper_model = None

    @property
    def whisper_model(self):
        """Lazy-load Whisper model."""
        if self._whisper_model is None:
            print(f"Loading Whisper {self.whisper_model_name} model...", file=sys.stderr)
            self._whisper_model = whisper.load_model(self.whisper_model_name)
        return self._whisper_model

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and normalize to float32.

        Returns:
            Tuple of (audio array, sample rate)
        """
        rate, audio = wavfile.read(audio_path)

        # Normalize to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed
        if rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=self.sample_rate)
            rate = self.sample_rate

        return audio, rate

    def compute_wer(self, audio_path: str, reference_text: str) -> Dict[str, any]:
        """
        Compute Word Error Rate using Whisper ASR.

        Args:
            audio_path: Path to audio file
            reference_text: Ground truth text

        Returns:
            Dictionary with WER and transcription
        """
        # Transcribe with Whisper
        result = self.whisper_model.transcribe(audio_path, language="en")
        hypothesis = result["text"].strip()

        # Normalize both sides for fair comparison (digit/word equivalence, punctuation, case)
        ref_normalized = normalize_text_for_wer(reference_text)
        hyp_normalized = normalize_text_for_wer(hypothesis)

        # Compute WER on normalized text
        error_rate = wer(ref_normalized, hyp_normalized)

        return {
            "wer": error_rate,
            "reference": reference_text,
            "hypothesis": hypothesis,
            "reference_normalized": ref_normalized,
            "hypothesis_normalized": hyp_normalized,
            "status": "excellent" if error_rate < 0.05 else "acceptable" if error_rate < 0.10 else "investigate"
        }

    def compute_mcd(self, audio1: np.ndarray, audio2: np.ndarray,
                    n_mfcc: int = 13) -> Dict[str, float]:
        """
        Compute Mel-Cepstral Distortion between two audio signals.

        MCD measures spectral similarity via MFCC distance.
        Skips C0 (energy coefficient) per standard practice.

        Args:
            audio1: First audio signal
            audio2: Second audio signal
            n_mfcc: Number of MFCC coefficients (typically 13)

        Returns:
            Dictionary with MCD value and status
        """
        # Extract MFCCs
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=self.sample_rate, n_mfcc=n_mfcc)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=self.sample_rate, n_mfcc=n_mfcc)

        # Skip C0 (energy/DC coefficient) — standard MCD practice
        mfcc1 = mfcc1[1:, :]
        mfcc2 = mfcc2[1:, :]

        # Align lengths (take minimum)
        min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = mfcc1[:, :min_frames]
        mfcc2 = mfcc2[:, :min_frames]

        # Compute MCD per frame, then average
        # Using Euclidean distance of MFCCs (without 10/ln10 scaling which is
        # designed for mel-cepstral coefficients from speech analysis tools,
        # not librosa MFCCs which have much larger magnitude)
        diff = mfcc1 - mfcc2
        per_frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))
        mcd = float(np.mean(per_frame_dist))

        return {
            "mcd": float(mcd),
            "n_frames": min_frames,
            "status": "excellent" if mcd < 50.0 else "good" if mcd < 100.0 else "investigate"
        }

    def compute_correlation(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """
        Compute Pearson correlation between two audio signals.

        Aligns by truncating to shorter length. Returns correlation
        coefficient in [-1, 1] where 1 = identical waveforms.
        """
        min_len = min(len(audio1), len(audio2))
        a1 = audio1[:min_len]
        a2 = audio2[:min_len]

        if np.std(a1) < 1e-10 or np.std(a2) < 1e-10:
            return 0.0

        return float(np.corrcoef(a1, a2)[0, 1])

    def compute_snr(self, audio: np.ndarray, noise_percentile: float = 5.0) -> Dict[str, float]:
        """
        Compute Signal-to-Noise Ratio.

        Estimates noise floor using variance of high-frequency components.

        Args:
            audio: Audio signal
            noise_percentile: Percentile for minimum noise floor (for very clean signals)

        Returns:
            Dictionary with SNR in dB
        """
        # Compute signal RMS
        signal_rms = np.sqrt(np.mean(audio ** 2))
        signal_power = signal_rms ** 2

        # Estimate noise using high-pass filtered signal variance
        # This captures high-frequency noise while preserving speech
        from scipy import signal as sp_signal

        # Design high-pass filter (> 8kHz for speech)
        nyquist = self.sample_rate / 2
        cutoff = min(8000, nyquist * 0.7)  # 8kHz or 70% of Nyquist
        b, a = sp_signal.butter(4, cutoff / nyquist, btype='high')

        # Filter to get high-frequency component (noise estimate)
        try:
            high_freq = sp_signal.filtfilt(b, a, audio)
            noise_rms = np.sqrt(np.mean(high_freq ** 2))
        except:
            # Fallback: use minimum amplitude regions
            threshold = np.percentile(np.abs(audio), noise_percentile)
            noise_samples = audio[np.abs(audio) < threshold]
            if len(noise_samples) > 0:
                noise_rms = np.sqrt(np.mean(noise_samples ** 2))
            else:
                # Very clean signal - use machine epsilon
                noise_rms = np.finfo(audio.dtype).eps * signal_rms

        # Ensure minimum noise floor (for very clean signals)
        min_noise = max(np.finfo(audio.dtype).eps, signal_rms * 1e-6)
        noise_rms = max(noise_rms, min_noise)
        noise_power = noise_rms ** 2

        # Compute SNR
        snr_db = 10 * np.log10(signal_power / noise_power)

        return {
            "snr_db": float(snr_db),
            "noise_floor": float(noise_rms),
            "signal_rms": float(signal_rms),
            "status": "excellent" if snr_db > 25 else "good" if snr_db > 15 else "investigate"
        }

    def compute_thd(self, audio: np.ndarray, fundamental_freq: Optional[float] = None) -> Dict[str, float]:
        """
        Compute Total Harmonic Distortion.

        Args:
            audio: Audio signal
            fundamental_freq: Expected fundamental frequency (Hz). If None, auto-detect.

        Returns:
            Dictionary with THD percentage
        """
        # Compute FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        magnitude = np.abs(fft)

        # Find fundamental frequency if not provided
        if fundamental_freq is None:
            # Use peak in reasonable speech range (80-300 Hz)
            speech_mask = (freqs >= 80) & (freqs <= 300)
            if np.any(speech_mask):
                fundamental_idx = np.argmax(magnitude[speech_mask])
                fundamental_freq = freqs[speech_mask][fundamental_idx]
            else:
                fundamental_freq = 150.0  # Default

        # Find fundamental peak
        fundamental_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fundamental_power = magnitude[fundamental_idx] ** 2

        # Find harmonics (2f, 3f, 4f, 5f)
        harmonic_power = 0.0
        for n in range(2, 6):
            harmonic_freq = n * fundamental_freq
            if harmonic_freq < freqs[-1]:
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_power += magnitude[harmonic_idx] ** 2

        # Compute THD
        if fundamental_power == 0:
            thd_percent = 0.0
        else:
            thd_percent = 100.0 * np.sqrt(harmonic_power / fundamental_power)

        return {
            "thd_percent": float(thd_percent),
            "fundamental_hz": float(fundamental_freq),
            "status": "excellent" if thd_percent < 10.0 else "acceptable" if thd_percent < 40.0 else "investigate"
        }

    def compute_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral characteristics.

        Returns:
            Dictionary with spectral features
        """
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)

        # Spectral rolloff (bandwidth)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)

        # Spectral flatness (noisiness)
        flatness = librosa.feature.spectral_flatness(y=audio)

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)

        return {
            "spectral_centroid_hz": float(np.mean(centroid)),
            "spectral_rolloff_hz": float(np.mean(rolloff)),
            "spectral_flatness": float(np.mean(flatness)),
            "zero_crossing_rate": float(np.mean(zcr)),
        }

    def analyze_audio(self, audio_path: str, reference_text: Optional[str] = None,
                     reference_audio: Optional[str] = None) -> Dict[str, any]:
        """
        Run comprehensive quality analysis on audio file.

        Args:
            audio_path: Path to audio to analyze
            reference_text: Ground truth text (for WER)
            reference_audio: Path to reference audio (for MCD)

        Returns:
            Dictionary with all quality metrics
        """
        results = {
            "audio_file": audio_path,
            "sample_rate": self.sample_rate,
        }

        # Load audio
        audio, rate = self.load_audio(audio_path)
        results["duration_sec"] = len(audio) / rate

        # Basic stats
        results["amplitude_max"] = float(np.max(np.abs(audio)))
        results["rms"] = float(np.sqrt(np.mean(audio ** 2)))

        # WER (if reference text provided)
        if reference_text:
            try:
                results["wer"] = self.compute_wer(audio_path, reference_text)
            except Exception as e:
                results["wer"] = {"error": str(e)}

        # MCD and correlation (if reference audio provided)
        if reference_audio:
            try:
                ref_audio, _ = self.load_audio(reference_audio)
                results["mcd"] = self.compute_mcd(audio, ref_audio)
                results["correlation"] = self.compute_correlation(audio, ref_audio)
            except Exception as e:
                results["mcd"] = {"error": str(e)}

        # Signal quality metrics
        try:
            results["snr"] = self.compute_snr(audio)
        except Exception as e:
            results["snr"] = {"error": str(e)}

        try:
            results["thd"] = self.compute_thd(audio)
        except Exception as e:
            results["thd"] = {"error": str(e)}

        try:
            results["spectral"] = self.compute_spectral_features(audio)
        except Exception as e:
            results["spectral"] = {"error": str(e)}

        return results


def print_report(results: Dict[str, any]):
    """Print formatted quality report."""
    print("=" * 70)
    print("AUDIO QUALITY REPORT")
    print("=" * 70)
    print()

    print(f"File: {results['audio_file']}")
    print(f"Duration: {results['duration_sec']:.2f}s")
    print(f"Sample Rate: {results['sample_rate']} Hz")
    print()

    print("BASIC METRICS:")
    print(f"  Max Amplitude: {results['amplitude_max']:.4f}")
    print(f"  RMS Level:     {results['rms']:.4f}")
    print()

    if "wer" in results and "error" not in results["wer"]:
        wer_data = results["wer"]
        print("INTELLIGIBILITY (WER):")
        print(f"  Word Error Rate: {wer_data['wer']*100:.1f}% ({wer_data['status'].upper()})")
        print(f"  Reference: {wer_data['reference']}")
        print(f"  Hypothesis: {wer_data['hypothesis']}")
        print()

    if "mcd" in results and "error" not in results["mcd"]:
        mcd_data = results["mcd"]
        print("ACOUSTIC SIMILARITY (MCD):")
        print(f"  MCD: {mcd_data['mcd']:.2f} dB ({mcd_data['status'].upper()})")
        print(f"  Frames: {mcd_data['n_frames']}")
        print()

    if "correlation" in results:
        corr = results["correlation"]
        corr_status = "excellent" if corr > 0.8 else "good" if corr > 0.5 else "investigate"
        print("WAVEFORM CORRELATION:")
        print(f"  Pearson r: {corr:.4f} ({corr_status.upper()})")
        print()

    if "snr" in results and "error" not in results["snr"]:
        snr_data = results["snr"]
        print("SIGNAL QUALITY (SNR):")
        print(f"  SNR: {snr_data['snr_db']:.1f} dB ({snr_data['status'].upper()})")
        print(f"  Signal RMS: {snr_data['signal_rms']:.4f}")
        print(f"  Noise Floor: {snr_data['noise_floor']:.6f}")
        print()

    if "thd" in results and "error" not in results["thd"]:
        thd_data = results["thd"]
        print("DISTORTION (THD):")
        print(f"  THD: {thd_data['thd_percent']:.2f}% ({thd_data['status'].upper()})")
        print(f"  Fundamental: {thd_data['fundamental_hz']:.1f} Hz")
        print()

    if "spectral" in results and "error" not in results["spectral"]:
        spec_data = results["spectral"]
        print("SPECTRAL FEATURES:")
        print(f"  Centroid:  {spec_data['spectral_centroid_hz']:.1f} Hz")
        print(f"  Rolloff:   {spec_data['spectral_rolloff_hz']:.1f} Hz")
        print(f"  Flatness:  {spec_data['spectral_flatness']:.4f}")
        print(f"  ZCR:       {spec_data['zero_crossing_rate']:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Audio quality metrics for TTS validation')
    parser.add_argument('--audio', required=True, help='Path to audio file to analyze')
    parser.add_argument('--text', help='Reference text for WER calculation')
    parser.add_argument('--reference', help='Path to reference audio for MCD calculation')
    parser.add_argument('--whisper-model', default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--output-json', help='Save results to JSON file')

    args = parser.parse_args()

    # Run analysis
    metrics = QualityMetrics(whisper_model=args.whisper_model)
    results = metrics.analyze_audio(args.audio, args.text, args.reference)

    # Print report
    print_report(results)

    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_json}")


if __name__ == '__main__':
    main()
