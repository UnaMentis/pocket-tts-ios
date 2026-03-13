#!/usr/bin/env python3
"""
Experiment Memory System for TTS Autotuning

Goes beyond the flat results.tsv to provide structured memory that:
1. Tracks failures so they aren't repeated
2. Highlights mixed results (improved some metrics, regressed others)
3. Records reasoning and hypotheses
4. Provides fast lookup for the agent to avoid repeating work

The memory is stored as JSON at autotuning/memory.json and is meant to be
read by the autotuning agent at the start of each loop iteration.

Usage:
    from memory import ExperimentMemory
    mem = ExperimentMemory(Path("autotuning/memory.json"))
    mem.record(experiment_id, config, metrics, decision, reasoning)
    mem.get_summary()  # compact summary for agent context
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentMemory:
    """Structured experiment memory with failure tracking and mixed-result highlighting."""

    DEFAULT_SCHEMA = {
        "version": 2,
        "created": None,
        "bootstrapped": False,
        "baseline": None,
        "best": None,
        "experiments": [],
        "dead_ends": [],
        "promising_leads": [],
        "rules_learned": [],
        "safe_ranges": {},
        "interaction_rules": [],
        "methodology_guidance": {},
        "sensitivity_rankings": {},
    }

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            with open(path) as f:
                self.data = json.load(f)
            # Migrate v1 → v2: add new fields if missing
            for key, default in self.DEFAULT_SCHEMA.items():
                if key not in self.data:
                    self.data[key] = default
        else:
            self.data = {**self.DEFAULT_SCHEMA, "created": datetime.now().isoformat()}

    def save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def set_baseline(self, score: float, config: Dict, per_metric: Dict):
        """Record the initial baseline."""
        self.data["baseline"] = {
            "score": score,
            "config": config,
            "per_metric": per_metric,
            "timestamp": datetime.now().isoformat(),
        }
        if self.data["best"] is None:
            self.data["best"] = {
                "score": score,
                "config": config,
                "per_metric": per_metric,
                "experiment_id": "baseline",
            }
        self.save()

    def record(
        self,
        experiment_id: str,
        config: Dict,
        composite_score: float,
        per_metric: Dict[str, float],
        decision: str,  # "kept", "discarded", "crashed"
        hypothesis: str,
        reasoning: str,
        changes_made: str,
        metric_deltas: Optional[Dict[str, float]] = None,
    ):
        """
        Record an experiment with full context.

        Args:
            experiment_id: Unique ID (e.g., "sweep_temp_003")
            config: Full config dict
            composite_score: The overall score
            per_metric: Individual metric scores (e.g., {"wer": 0.05, "mcd": 4.2, ...})
            decision: "kept", "discarded", or "crashed"
            hypothesis: What the agent expected to happen
            reasoning: Why this was tried
            changes_made: What was actually changed (param value, code change, etc.)
            metric_deltas: Per-metric change from previous best (e.g., {"wer": -0.02, "mcd": +0.3})
        """
        best = self.data.get("best", {})
        best_score = best.get("score", 0.0) if best else 0.0

        entry = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "composite_score": composite_score,
            "per_metric": per_metric,
            "delta_from_best": composite_score - best_score,
            "metric_deltas": metric_deltas or {},
            "decision": decision,
            "hypothesis": hypothesis,
            "reasoning": reasoning,
            "changes_made": changes_made,
        }

        # Classify the result
        if metric_deltas:
            positive = {k: v for k, v in metric_deltas.items() if v > 0.001}
            negative = {k: v for k, v in metric_deltas.items() if v < -0.001}

            if positive and negative:
                entry["classification"] = "mixed"
                entry["improved_metrics"] = list(positive.keys())
                entry["regressed_metrics"] = list(negative.keys())
            elif positive and not negative:
                entry["classification"] = "pure_improvement"
            elif negative and not positive:
                entry["classification"] = "pure_regression"
            else:
                entry["classification"] = "neutral"
        else:
            entry["classification"] = decision

        self.data["experiments"].append(entry)

        # Update best if kept
        if decision == "kept" and composite_score > best_score:
            self.data["best"] = {
                "score": composite_score,
                "config": config,
                "per_metric": per_metric,
                "experiment_id": experiment_id,
            }

        # Auto-classify into dead_ends or promising_leads
        if decision == "discarded":
            if entry.get("classification") == "mixed":
                self._add_promising_lead(entry)
            elif entry.get("classification") == "pure_regression":
                self._add_dead_end(entry)

        self.save()

    def _add_dead_end(self, entry: Dict):
        """Record a dead end to avoid repeating."""
        self.data["dead_ends"].append({
            "experiment_id": entry["experiment_id"],
            "changes_made": entry["changes_made"],
            "hypothesis": entry["hypothesis"],
            "result": f"score {entry['composite_score']:.4f} (delta {entry['delta_from_best']:+.4f})",
            "timestamp": entry["timestamp"],
        })

    def _add_promising_lead(self, entry: Dict):
        """Record a mixed result that might be worth revisiting."""
        self.data["promising_leads"].append({
            "experiment_id": entry["experiment_id"],
            "changes_made": entry["changes_made"],
            "hypothesis": entry["hypothesis"],
            "improved": entry.get("improved_metrics", []),
            "regressed": entry.get("regressed_metrics", []),
            "metric_deltas": entry.get("metric_deltas", {}),
            "composite_delta": entry["delta_from_best"],
            "note": (
                f"Improved {', '.join(entry.get('improved_metrics', []))} "
                f"but regressed {', '.join(entry.get('regressed_metrics', []))}. "
                f"Net composite delta: {entry['delta_from_best']:+.4f}"
            ),
            "timestamp": entry["timestamp"],
        })

    def add_rule(self, rule: str, evidence: str):
        """Record a learned rule/principle from experimentation."""
        self.data["rules_learned"].append({
            "rule": rule,
            "evidence": evidence,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def set_safe_range(self, param: str, safe_min: float, optimal: float, safe_max: float, notes: str = ""):
        """Record a safe parameter range based on evidence."""
        self.data["safe_ranges"][param] = {
            "min": safe_min,
            "optimal": optimal,
            "max": safe_max,
            "notes": notes,
        }
        self.save()

    def get_safe_range(self, param: str) -> Optional[Dict]:
        """Get the safe range for a parameter, if known."""
        return self.data.get("safe_ranges", {}).get(param)

    def add_interaction_rule(self, rule: str, params: List[str], evidence: str):
        """Record an interaction between parameters."""
        self.data["interaction_rules"].append({
            "rule": rule,
            "params": params,
            "evidence": evidence,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def set_methodology_guidance(self, bottleneck: str, guidance: Dict[str, str]):
        """Record what to do when a specific metric is the bottleneck."""
        self.data["methodology_guidance"][bottleneck] = {
            **guidance,
            "timestamp": datetime.now().isoformat(),
        }
        self.save()

    def set_sensitivity(self, param: str, gradient: float, evidence: str = ""):
        """Record sensitivity of composite score to a parameter."""
        self.data["sensitivity_rankings"][param] = {
            "gradient": gradient,
            "evidence": evidence,
            "timestamp": datetime.now().isoformat(),
        }
        self.save()

    def is_bootstrapped(self) -> bool:
        """Check if memory has been bootstrapped with project knowledge."""
        return self.data.get("bootstrapped", False)

    def mark_bootstrapped(self):
        """Mark memory as bootstrapped."""
        self.data["bootstrapped"] = True
        self.save()

    def was_tried(self, changes_description: str) -> Optional[Dict]:
        """
        Check if something similar was already tried.

        Returns the experiment entry if found, None otherwise.
        Simple substring matching — the agent should use this as a hint,
        not an absolute gate.
        """
        desc_lower = changes_description.lower()
        for exp in self.data["experiments"]:
            if desc_lower in exp.get("changes_made", "").lower():
                return exp
        return None

    def get_summary(self, max_dead_ends: int = 10, max_leads: int = 5) -> str:
        """
        Generate a compact summary for the agent's context window.

        This is the primary interface — the agent reads this at the start
        of each iteration to know what's been tried and what to avoid.
        """
        lines = []
        lines.append("# Experiment Memory Summary")
        lines.append("")

        # Current state
        best = self.data.get("best", {})
        baseline = self.data.get("baseline", {})
        total = len(self.data.get("experiments", []))

        lines.append(f"## Status")
        lines.append(f"- Total experiments: {total}")
        lines.append(f"- Baseline score: {baseline.get('score', 'N/A')}")
        lines.append(f"- Best score: {best.get('score', 'N/A')} (exp: {best.get('experiment_id', 'N/A')})")
        if best and baseline:
            delta = best.get("score", 0) - baseline.get("score", 0)
            lines.append(f"- Improvement from baseline: {delta:+.4f}")
        lines.append("")

        # Best config
        if best.get("config"):
            lines.append("## Current Best Config")
            for k, v in best["config"].items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        # Safe ranges
        safe_ranges = self.data.get("safe_ranges", {})
        if safe_ranges:
            lines.append("## Safe Parameter Ranges")
            for param, r in safe_ranges.items():
                lines.append(f"- **{param}**: [{r['min']}, {r['max']}] optimal={r['optimal']}" +
                             (f" — {r['notes']}" if r.get('notes') else ""))
            lines.append("")

        # Rules learned
        rules = self.data.get("rules_learned", [])
        if rules:
            lines.append("## Rules Learned")
            for r in rules:
                lines.append(f"- {r['rule']} (evidence: {r['evidence']})")
            lines.append("")

        # Interaction rules
        interactions = self.data.get("interaction_rules", [])
        if interactions:
            lines.append("## Interaction Rules")
            for ir in interactions:
                lines.append(f"- [{', '.join(ir['params'])}] {ir['rule']}")
            lines.append("")

        # Methodology guidance
        guidance = self.data.get("methodology_guidance", {})
        if guidance:
            lines.append("## Methodology Guidance (by bottleneck)")
            for bottleneck, g in guidance.items():
                lines.append(f"- **{bottleneck}**: {g.get('summary', '')}")
                if g.get("tier1"):
                    lines.append(f"  - Tier 1: {g['tier1']}")
                if g.get("tier2"):
                    lines.append(f"  - Tier 2: {g['tier2']}")
            lines.append("")

        # Sensitivity rankings
        sensitivity = self.data.get("sensitivity_rankings", {})
        if sensitivity:
            ranked = sorted(sensitivity.items(), key=lambda x: x[1]["gradient"], reverse=True)
            lines.append("## Sensitivity Rankings (highest impact first)")
            for param, s in ranked:
                lines.append(f"- {param}: gradient={s['gradient']:.4f}")
            lines.append("")

        # Promising leads (mixed results worth revisiting)
        leads = self.data.get("promising_leads", [])
        if leads:
            lines.append(f"## Promising Leads ({len(leads)} total, showing latest {max_leads})")
            for lead in leads[-max_leads:]:
                lines.append(f"- **{lead['changes_made']}**: {lead['note']}")
            lines.append("")

        # Dead ends (things not to retry)
        dead = self.data.get("dead_ends", [])
        if dead:
            lines.append(f"## Dead Ends ({len(dead)} total, showing latest {max_dead_ends})")
            for d in dead[-max_dead_ends:]:
                lines.append(f"- {d['changes_made']}: {d['result']}")
            lines.append("")

        # Recent experiments (last 5)
        exps = self.data.get("experiments", [])
        if exps:
            lines.append("## Recent Experiments (last 5)")
            for exp in exps[-5:]:
                status = exp.get("classification", exp.get("decision", "?"))
                lines.append(
                    f"- [{exp['experiment_id']}] {exp['changes_made']} → "
                    f"score {exp['composite_score']:.4f} ({exp['delta_from_best']:+.4f}) "
                    f"[{status}]"
                )
            lines.append("")

        return "\n".join(lines)

    def get_metric_trends(self, metric_name: str) -> List[Dict]:
        """Get the trajectory of a specific metric across experiments."""
        trend = []
        for exp in self.data.get("experiments", []):
            per_metric = exp.get("per_metric", {})
            if metric_name in per_metric:
                trend.append({
                    "experiment_id": exp["experiment_id"],
                    "value": per_metric[metric_name],
                    "composite": exp["composite_score"],
                    "decision": exp["decision"],
                })
        return trend


def main():
    """Print memory summary."""
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("autotuning/memory.json")
    if not path.exists():
        print(f"No memory file at {path}")
        return
    mem = ExperimentMemory(path)
    print(mem.get_summary())


if __name__ == "__main__":
    main()
