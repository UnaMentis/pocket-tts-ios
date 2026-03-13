#!/usr/bin/env python3
"""
Autoresearch-Style TTS Autotuning Loop

Iteratively explores TTS configuration parameters, measuring quality
after each change, keeping improvements and discarding regressions.

Inspired by Karpathy's autoresearch pattern:
  modify → evaluate → keep/discard → loop

Usage:
    # Establish baseline
    python autotune.py --phase baseline --model-dir ./kyutai-pocket-ios

    # Sweep a single parameter
    python autotune.py --phase sweep --param temperature --model-dir ./kyutai-pocket-ios

    # Full joint optimization
    python autotune.py --phase optimize --iterations 100 --model-dir ./kyutai-pocket-ios

    # Resume from previous state
    python autotune.py --phase optimize --resume --model-dir ./kyutai-pocket-ios
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add autotuning dir and validation dir to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "validation"))

from scorer import compute_composite_score, score_from_metrics_dict
from memory import ExperimentMemory

# ─── Test Corpus ───────────────────────────────────────────────────────────

TEST_PHRASES = [
    {
        "id": "short_greeting",
        "text": "Hello, how are you today?",
        "category": "short",
    },
    {
        "id": "medium_numbers",
        "text": "The temperature is 72 degrees Fahrenheit.",
        "category": "medium",
    },
    {
        "id": "long_narrative",
        "text": "In the beginning, there was nothing but darkness and silence, until a single spark of light appeared.",
        "category": "long",
    },
    {
        "id": "question_prosody",
        "text": "Did you really think that was going to work?",
        "category": "prosody",
    },
    {
        "id": "technical",
        "text": "The API returns a JSON response with the authentication token.",
        "category": "technical",
    },
]

# ─── Parameter Search Space ────────────────────────────────────────────────

PARAM_SPACE = {
    "temperature": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "default": 0.7,
    },
    "top_p": {
        "type": "float",
        "min": 0.1,
        "max": 1.0,
        "step": 0.05,
        "default": 0.9,
    },
    "consistency_steps": {
        "type": "int",
        "min": 1,
        "max": 4,
        "step": 1,
        "default": 2,
    },
    "speed": {
        "type": "float",
        "min": 0.8,
        "max": 1.2,
        "step": 0.05,
        "default": 1.0,
    },
}


class TTSRunner:
    """Wraps the Rust TTS binary for synthesis."""

    def __init__(self, project_dir: Path, model_dir: Path):
        self.project_dir = project_dir
        self.model_dir = model_dir
        self.binary = project_dir / "target" / "release" / "test-tts"

    def ensure_built(self):
        """Build the Rust binary if needed."""
        if not self.binary.exists():
            print("Building Rust binary (release mode)...")
            result = subprocess.run(
                ["cargo", "build", "--release", "--bin", "test-tts"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"Build failed: {result.stderr}", file=sys.stderr)
                raise RuntimeError("Failed to build TTS binary")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        config: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Synthesize text to audio file with given config.

        Returns timing information.
        """
        cmd = [
            str(self.binary),
            "--model-dir", str(self.model_dir),
            "--text", text,
            "--output", str(output_path),
            "--temperature", str(config.get("temperature", 0.7)),
            "--top-p", str(config.get("top_p", 0.9)),
            "--consistency-steps", str(config.get("consistency_steps", 2)),
            "--speed", str(config.get("speed", 1.0)),
            "--seed", str(config.get("seed", 42)),
        ]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start

        if result.returncode != 0:
            raise RuntimeError(f"Synthesis failed: {result.stderr}")

        return {"synthesis_time_sec": elapsed}


class QualityEvaluator:
    """Evaluates audio quality using the validation metrics suite."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.quality_script = project_dir / "validation" / "quality_metrics.py"

    def evaluate(
        self,
        audio_path: Path,
        reference_text: str,
        reference_audio: Optional[Path] = None,
    ) -> Dict:
        """Run quality metrics on an audio file."""
        cmd = [
            sys.executable,
            str(self.quality_script),
            "--audio", str(audio_path),
            "--text", reference_text,
            "--output-json", str(audio_path.with_suffix(".metrics.json")),
        ]

        if reference_audio and reference_audio.exists():
            cmd.extend(["--reference", str(reference_audio)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"Quality evaluation warning: {result.stderr}", file=sys.stderr)

        # Load results
        metrics_path = audio_path.with_suffix(".metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)

        return {"error": "metrics file not generated"}


class ExperimentLog:
    """TSV log of all experiments (autoresearch-style)."""

    COLUMNS = [
        "experiment_id", "timestamp", "temperature", "top_p",
        "consistency_steps", "speed", "seed", "composite_score",
        "wer", "mcd", "snr_db", "thd_percent", "correlation",
        "synthesis_time_sec", "status", "description",
    ]

    def __init__(self, log_path: Path):
        self.log_path = log_path
        if not log_path.exists():
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(self.COLUMNS)

    def append(self, entry: Dict):
        """Append an experiment result."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            row = [entry.get(col, "") for col in self.COLUMNS]
            writer.writerow(row)

    def get_best_score(self) -> float:
        """Get the best composite score so far."""
        best = 0.0
        if not self.log_path.exists():
            return best

        with open(self.log_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    score = float(row.get("composite_score", 0))
                    if score > best:
                        best = score
                except (ValueError, TypeError):
                    pass
        return best

    def get_experiment_count(self) -> int:
        """Count total experiments."""
        if not self.log_path.exists():
            return 0
        with open(self.log_path) as f:
            return sum(1 for _ in f) - 1  # subtract header

    def get_best_config(self) -> Optional[Dict]:
        """Get the config that produced the best score."""
        best_score = 0.0
        best_row = None

        if not self.log_path.exists():
            return None

        with open(self.log_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    score = float(row.get("composite_score", 0))
                    if score > best_score:
                        best_score = score
                        best_row = row
                except (ValueError, TypeError):
                    pass

        if best_row:
            return {
                "temperature": float(best_row.get("temperature", 0.7)),
                "top_p": float(best_row.get("top_p", 0.9)),
                "consistency_steps": int(best_row.get("consistency_steps", 2)),
                "speed": float(best_row.get("speed", 1.0)),
                "seed": int(best_row.get("seed", 42)),
                "composite_score": best_score,
            }
        return None


class AutoTuner:
    """Main autotuning orchestrator."""

    def __init__(
        self,
        project_dir: Path,
        model_dir: Path,
        output_dir: Path,
        reference_dir: Optional[Path] = None,
    ):
        self.project_dir = project_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.reference_dir = reference_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "baselines").mkdir(exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)

        self.runner = TTSRunner(project_dir, model_dir)
        self.evaluator = QualityEvaluator(project_dir)
        self.log = ExperimentLog(output_dir / "results.tsv")
        self.memory = ExperimentMemory(output_dir / "memory.json")

    def _get_reference_audio(self, phrase_id: str) -> Optional[Path]:
        """Get reference audio for a phrase if available."""
        if self.reference_dir:
            ref_path = self.reference_dir / f"{phrase_id}.wav"
            if ref_path.exists():
                return ref_path
        return None

    def evaluate_config(
        self,
        config: Dict[str, Any],
        experiment_id: str,
        description: str = "",
        hypothesis: str = "",
        reasoning: str = "",
    ) -> Dict:
        """
        Run full evaluation of a config across all test phrases.

        Returns aggregate metrics and composite score.
        """
        phrase_scores = []
        phrase_metrics = []
        total_synthesis_time = 0.0

        for phrase in TEST_PHRASES:
            audio_path = self.output_dir / "audio" / f"{experiment_id}_{phrase['id']}.wav"

            try:
                # Synthesize
                timing = self.runner.synthesize(phrase["text"], audio_path, config)
                total_synthesis_time += timing["synthesis_time_sec"]

                # Evaluate
                ref_audio = self._get_reference_audio(phrase["id"])
                metrics = self.evaluator.evaluate(
                    audio_path, phrase["text"], ref_audio
                )
                phrase_metrics.append(metrics)

                # Score
                score_result = score_from_metrics_dict(metrics)
                phrase_scores.append(score_result["composite_score"])

            except Exception as e:
                print(f"  Error on '{phrase['id']}': {e}", file=sys.stderr)
                phrase_scores.append(0.0)
                phrase_metrics.append({"error": str(e)})

        # Aggregate
        avg_score = float(np.mean(phrase_scores)) if phrase_scores else 0.0
        min_score = float(np.min(phrase_scores)) if phrase_scores else 0.0

        # Extract average individual metrics
        avg_wer = self._avg_metric(phrase_metrics, "wer", "wer")
        avg_mcd = self._avg_metric(phrase_metrics, "mcd", "mcd")
        avg_snr = self._avg_metric(phrase_metrics, "snr", "snr_db")
        avg_thd = self._avg_metric(phrase_metrics, "thd", "thd_percent")

        result = {
            "experiment_id": experiment_id,
            "config": config,
            "composite_score": avg_score,
            "min_score": min_score,
            "phrase_scores": phrase_scores,
            "avg_wer": avg_wer,
            "avg_mcd": avg_mcd,
            "avg_snr": avg_snr,
            "avg_thd": avg_thd,
            "total_synthesis_time": total_synthesis_time,
        }

        # Log to TSV
        self.log.append({
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "temperature": config.get("temperature", ""),
            "top_p": config.get("top_p", ""),
            "consistency_steps": config.get("consistency_steps", ""),
            "speed": config.get("speed", ""),
            "seed": config.get("seed", ""),
            "composite_score": f"{avg_score:.6f}",
            "wer": f"{avg_wer:.4f}" if avg_wer is not None else "",
            "mcd": f"{avg_mcd:.2f}" if avg_mcd is not None else "",
            "snr_db": f"{avg_snr:.1f}" if avg_snr is not None else "",
            "thd_percent": f"{avg_thd:.2f}" if avg_thd is not None else "",
            "correlation": "",
            "synthesis_time_sec": f"{total_synthesis_time:.2f}",
            "status": description.split(":")[0] if ":" in description else "evaluated",
            "description": description,
        })

        # Compute per-metric deltas from current best for memory system
        best = self.memory.data.get("best", {})
        best_per_metric = best.get("per_metric", {}) if best else {}
        per_metric = {}
        metric_deltas = {}

        if avg_wer is not None:
            per_metric["wer"] = avg_wer
            if "wer" in best_per_metric:
                metric_deltas["wer"] = avg_wer - best_per_metric["wer"]
        if avg_mcd is not None:
            per_metric["mcd"] = avg_mcd
            if "mcd" in best_per_metric:
                metric_deltas["mcd"] = avg_mcd - best_per_metric["mcd"]
        if avg_snr is not None:
            per_metric["snr_db"] = avg_snr
            if "snr_db" in best_per_metric:
                metric_deltas["snr_db"] = avg_snr - best_per_metric["snr_db"]
        if avg_thd is not None:
            per_metric["thd_percent"] = avg_thd
            if "thd_percent" in best_per_metric:
                metric_deltas["thd_percent"] = avg_thd - best_per_metric["thd_percent"]

        result["per_metric"] = per_metric
        result["metric_deltas"] = metric_deltas

        return result

    def _avg_metric(
        self, metrics_list: List[Dict], category: str, key: str
    ) -> Optional[float]:
        """Extract and average a specific metric across phrases."""
        values = []
        for m in metrics_list:
            if category in m and isinstance(m[category], dict) and key in m[category]:
                values.append(m[category][key])
        return float(np.mean(values)) if values else None

    # ─── Phase: Baseline ───────────────────────────────────────────────

    def run_baseline(self) -> Dict:
        """Establish baseline with default configuration."""
        print("=" * 60)
        print("PHASE 1: ESTABLISHING BASELINE")
        print("=" * 60)

        config = {
            "temperature": PARAM_SPACE["temperature"]["default"],
            "top_p": PARAM_SPACE["top_p"]["default"],
            "consistency_steps": PARAM_SPACE["consistency_steps"]["default"],
            "speed": PARAM_SPACE["speed"]["default"],
            "seed": 42,
        }

        self.runner.ensure_built()
        result = self.evaluate_config(
            config, "baseline", "baseline: default configuration",
            hypothesis="Establish baseline with default parameters",
            reasoning="Starting point for all future comparisons",
        )

        # Record in memory
        self.memory.set_baseline(
            result["composite_score"], config, result.get("per_metric", {})
        )
        self.memory.record(
            experiment_id="baseline",
            config=config,
            composite_score=result["composite_score"],
            per_metric=result.get("per_metric", {}),
            decision="kept",
            hypothesis="Establish baseline with default parameters",
            reasoning="Starting point for all future comparisons",
            changes_made="Default config: temp=0.7, top_p=0.9, consistency=2, speed=1.0",
        )

        # Save baseline
        baseline_path = self.output_dir / "baselines" / "initial.json"
        with open(baseline_path, "w") as f:
            json.dump(result, f, indent=2)

        # Save as current best
        best_path = self.output_dir / "configs" / "best.json"
        with open(best_path, "w") as f:
            json.dump({"config": config, "score": result["composite_score"]}, f, indent=2)

        print(f"\nBaseline score: {result['composite_score']:.4f}")
        print(f"Saved to: {baseline_path}")
        return result

    # ─── Phase: Parameter Sweep ────────────────────────────────────────

    def run_sweep(self, param_name: str) -> Dict:
        """Sweep a single parameter across its range."""
        if param_name not in PARAM_SPACE:
            raise ValueError(f"Unknown parameter: {param_name}. Choose from: {list(PARAM_SPACE.keys())}")

        space = PARAM_SPACE[param_name]
        print("=" * 60)
        print(f"PHASE 2: SWEEPING '{param_name}'")
        print(f"  Range: {space['min']} to {space['max']} (step {space['step']})")
        print("=" * 60)

        # Load current best config
        best_config = self._load_best_config()
        best_score = self.log.get_best_score()

        # Generate sweep values
        if space["type"] == "int":
            values = list(range(space["min"], space["max"] + 1, space["step"]))
        else:
            values = list(np.arange(space["min"], space["max"] + space["step"] / 2, space["step"]))
            values = [round(v, 4) for v in values]

        print(f"  Testing {len(values)} values: {values[:5]}...{values[-3:]}")
        print(f"  Current best score: {best_score:.4f}")
        print()

        self.runner.ensure_built()
        sweep_results = []

        for i, value in enumerate(values):
            config = deepcopy(best_config)
            config[param_name] = value

            exp_id = f"sweep_{param_name}_{i:03d}"
            desc = f"sweep: {param_name}={value}"

            print(f"  [{i+1}/{len(values)}] {param_name}={value} ...", end=" ", flush=True)

            hyp = f"Testing if {param_name}={value} improves quality"
            result = self.evaluate_config(config, exp_id, desc, hypothesis=hyp, reasoning=f"Systematic sweep of {param_name}")
            sweep_results.append(result)

            score = result["composite_score"]
            delta = score - best_score
            decision = "kept" if score > best_score else "discarded"

            self.memory.record(
                experiment_id=exp_id,
                config=config,
                composite_score=score,
                per_metric=result.get("per_metric", {}),
                decision=decision,
                hypothesis=hyp,
                reasoning=f"Systematic sweep of {param_name}",
                changes_made=f"{param_name}={value}",
                metric_deltas=result.get("metric_deltas", {}),
            )

            if score > best_score:
                print(f"score={score:.4f} (+{delta:.4f}) *** NEW BEST ***")
                best_score = score
                best_config = deepcopy(config)
                self._save_best_config(best_config, best_score)
            else:
                print(f"score={score:.4f} ({delta:+.4f})")

        print(f"\nSweep complete. Best {param_name} = {best_config[param_name]}")
        print(f"Best score: {best_score:.4f}")

        return {
            "param": param_name,
            "best_value": best_config[param_name],
            "best_score": best_score,
            "num_experiments": len(values),
        }

    # ─── Phase: Joint Optimization ─────────────────────────────────────

    def run_optimize(self, iterations: int = 100) -> Dict:
        """
        Joint optimization via random search with shrinking neighborhood.

        Uses a simple strategy:
        1. Start from best known config
        2. Perturb 1-2 parameters randomly
        3. Evaluate
        4. Keep if improved, discard if not
        5. Shrink perturbation radius over time
        """
        print("=" * 60)
        print(f"PHASE 3: JOINT OPTIMIZATION ({iterations} iterations)")
        print("=" * 60)

        best_config = self._load_best_config()
        best_score = self.log.get_best_score()
        start_exp = self.log.get_experiment_count()

        print(f"  Starting from score: {best_score:.4f}")
        print(f"  Config: {best_config}")
        print()

        self.runner.ensure_built()
        improvements = 0

        for i in range(iterations):
            # Decay factor: perturbations get smaller over time
            decay = 1.0 - (i / iterations) * 0.7  # 1.0 → 0.3

            # Perturb 1-2 random parameters
            config = deepcopy(best_config)
            params_to_perturb = np.random.choice(
                list(PARAM_SPACE.keys()),
                size=np.random.randint(1, 3),
                replace=False,
            )

            perturbation_desc = []
            for param in params_to_perturb:
                space = PARAM_SPACE[param]
                current = config.get(param, space["default"])

                if space["type"] == "int":
                    delta = np.random.choice([-1, 0, 1])
                    new_val = int(np.clip(current + delta, space["min"], space["max"]))
                else:
                    range_size = space["max"] - space["min"]
                    delta = np.random.normal(0, range_size * 0.1 * decay)
                    new_val = round(float(np.clip(current + delta, space["min"], space["max"])), 4)
                    # Snap to step grid
                    new_val = round(new_val / space["step"]) * space["step"]
                    new_val = round(float(np.clip(new_val, space["min"], space["max"])), 4)

                config[param] = new_val
                perturbation_desc.append(f"{param}={new_val}")

            exp_id = f"opt_{start_exp + i:04d}"
            desc = f"optimize: {', '.join(perturbation_desc)}"

            changes_str = ", ".join(perturbation_desc)
            hyp = f"Random perturbation of {changes_str} might find better optimum"
            print(f"  [{i+1}/{iterations}] {changes_str} ...", end=" ", flush=True)

            result = self.evaluate_config(config, exp_id, desc, hypothesis=hyp, reasoning="Joint optimization random search")
            score = result["composite_score"]
            delta = score - best_score
            decision = "kept" if score > best_score else "discarded"

            self.memory.record(
                experiment_id=exp_id,
                config=config,
                composite_score=score,
                per_metric=result.get("per_metric", {}),
                decision=decision,
                hypothesis=hyp,
                reasoning="Joint optimization random search",
                changes_made=changes_str,
                metric_deltas=result.get("metric_deltas", {}),
            )

            if score > best_score:
                improvements += 1
                print(f"score={score:.4f} (+{delta:.4f}) *** IMPROVED (#{improvements}) ***")
                best_score = score
                best_config = deepcopy(config)
                self._save_best_config(best_config, best_score)
            else:
                print(f"score={score:.4f} ({delta:+.4f}) discarded")

        print(f"\nOptimization complete.")
        print(f"  Iterations: {iterations}")
        print(f"  Improvements: {improvements} ({improvements/iterations*100:.1f}%)")
        print(f"  Final best score: {best_score:.4f}")
        print(f"  Best config: {best_config}")

        return {
            "iterations": iterations,
            "improvements": improvements,
            "best_score": best_score,
            "best_config": best_config,
        }

    # ─── Helpers ───────────────────────────────────────────────────────

    def _load_best_config(self) -> Dict:
        """Load the current best configuration."""
        best_path = self.output_dir / "configs" / "best.json"
        if best_path.exists():
            with open(best_path) as f:
                data = json.load(f)
                return data.get("config", self._default_config())
        return self._default_config()

    def _save_best_config(self, config: Dict, score: float):
        """Save new best configuration."""
        best_path = self.output_dir / "configs" / "best.json"
        with open(best_path, "w") as f:
            json.dump({
                "config": config,
                "score": score,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            param: space["default"]
            for param, space in PARAM_SPACE.items()
        } | {"seed": 42}

    def print_summary(self):
        """Print current autotuning summary."""
        total = self.log.get_experiment_count()
        best = self.log.get_best_config()

        print("=" * 60)
        print("AUTOTUNING SUMMARY")
        print("=" * 60)
        print(f"  Total experiments: {total}")
        if best:
            print(f"  Best score: {best['composite_score']:.4f}")
            print(f"  Best config:")
            for k, v in best.items():
                if k != "composite_score":
                    print(f"    {k}: {v}")
        print()

        # Print memory summary if available
        if self.memory.data.get("experiments"):
            dead = len(self.memory.data.get("dead_ends", []))
            leads = len(self.memory.data.get("promising_leads", []))
            rules = len(self.memory.data.get("rules_learned", []))
            print(f"  Memory: {dead} dead ends, {leads} promising leads, {rules} rules learned")
            if leads > 0:
                print(f"  Run `python autotuning/memory.py` for full memory summary")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch-style TTS autotuning loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Establish baseline
  python autotune.py --phase baseline --model-dir ./kyutai-pocket-ios

  # Sweep temperature
  python autotune.py --phase sweep --param temperature --model-dir ./kyutai-pocket-ios

  # Joint optimization (100 iterations)
  python autotune.py --phase optimize --iterations 100 --model-dir ./kyutai-pocket-ios

  # Print summary
  python autotune.py --phase summary --model-dir ./kyutai-pocket-ios
        """,
    )

    parser.add_argument(
        "--phase",
        required=True,
        choices=["baseline", "sweep", "optimize", "summary"],
        help="Which phase to run",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("kyutai-pocket-ios"),
        help="Path to model directory with safetensors + tokenizer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("autotuning"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory with reference audio files (from Python TTS)",
    )
    parser.add_argument(
        "--param",
        help="Parameter to sweep (for --phase sweep)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations (for --phase optimize)",
    )

    args = parser.parse_args()
    project_dir = Path(__file__).parent.parent

    tuner = AutoTuner(
        project_dir=project_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        reference_dir=args.reference_dir,
    )

    if args.phase == "baseline":
        tuner.run_baseline()

    elif args.phase == "sweep":
        if not args.param:
            print("Error: --param required for sweep phase", file=sys.stderr)
            print(f"Available: {list(PARAM_SPACE.keys())}", file=sys.stderr)
            sys.exit(1)
        tuner.run_sweep(args.param)

    elif args.phase == "optimize":
        tuner.run_optimize(args.iterations)

    elif args.phase == "summary":
        tuner.print_summary()

    # Always print summary at end
    if args.phase != "summary":
        tuner.print_summary()


if __name__ == "__main__":
    main()
