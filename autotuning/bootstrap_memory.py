#!/usr/bin/env python3
"""
Bootstrap Memory from Project History

Seeds the autotuning memory system with hard-won knowledge from the Pocket TTS
porting effort. This knowledge comes from:
  - docs/KNOWLEDGE_INDEX.md (8 critical lessons, failed approaches)
  - 24 completed experiments in memory.json
  - iOS reference testing infrastructure

Run once to bootstrap, then the memory system takes over:
    .venv/bin/python autotuning/bootstrap_memory.py
"""

import sys
from pathlib import Path

# Add autotuning dir to path
sys.path.insert(0, str(Path(__file__).parent))

from memory import ExperimentMemory


def bootstrap(memory_path: Path):
    mem = ExperimentMemory(memory_path)

    if mem.is_bootstrapped():
        print("Memory already bootstrapped. Skipping.")
        print(f"  Total experiments: {len(mem.data.get('experiments', []))}")
        print(f"  Rules learned: {len(mem.data.get('rules_learned', []))}")
        print(f"  Safe ranges: {len(mem.data.get('safe_ranges', {}))}")
        return

    print("Bootstrapping memory from project history...")

    # ── Dead End Rules ─────────────────────────────────────────────────────
    # These are recorded as rules_learned with "DEAD END" prefix for visibility

    dead_end_rules = [
        {
            "rule": "DEAD END: top_p > 0.9 causes catastrophic word dropping (40% WER at 0.95). Keep top_p at 0.9.",
            "evidence": "manual_tp95_001: WER jumped to 40%, dropped 'Hello' entirely",
        },
        {
            "rule": "DEAD END: Deterministic zero latents (Tensor::zeros in FlowNet) mask real quality issues. Always use production random noise.",
            "evidence": "KNOWLEDGE_INDEX lesson #1: Rust used zeros while Python uses normal_(mean=0, std=sqrt(temp))",
        },
        {
            "rule": "DEAD END: Amplitude scaling as a fix for low output. Root cause is always a deeper bug (e.g., batch-mode SEANet).",
            "evidence": "KNOWLEDGE_INDEX 'Things Tried and Failed': 5x scaling removed after fixing SEANet streaming",
        },
        {
            "rule": "DEAD END: Crossfade at chunk boundaries. Streaming ConvTranspose1d state handles continuity — crossfade blends different audio content.",
            "evidence": "KNOWLEDGE_INDEX 'Things Tried and Failed'",
        },
        {
            "rule": "DEAD END: min_gen_steps > 0 (forced minimum generation). Prevents natural EOS on short phrases. Use min_gen_steps=0.",
            "evidence": "KNOWLEDGE_INDEX 'Things Tried and Failed'",
        },
        {
            "rule": "DEAD END: temperature < 0.05 produces perfect WER but unnatural speech (composite 0.8059 < best 0.8251). Too deterministic.",
            "evidence": "sweep_temperature_000: temp=0.0, WER=0.0 but overall score lower than temp=0.7",
        },
    ]

    for rule in dead_end_rules:
        mem.add_rule(rule["rule"], rule["evidence"])

    # ── Learned Principles ─────────────────────────────────────────────────

    principles = [
        {
            "rule": "Temperature is the dominant tuning lever. Parabolic quality curve, peak at 0.7. Controls FlowNet noise via sqrt(temp).",
            "evidence": "20-point temperature sweep (0.0-1.0): best=0.8251 at temp=0.7",
        },
        {
            "rule": "WER and THD trade off inversely. Cleaner latents (lower temp) improve intelligibility but increase harmonic distortion. Must balance.",
            "evidence": "Temperature sweep: low temp = low WER but high THD; high temp = higher WER but lower THD",
        },
        {
            "rule": "SNR is weakly correlated with temperature (only 3.4dB spread across full range). Low-priority optimization target (15% weight).",
            "evidence": "Temperature sweep: SNR ranged 23.7-27.1dB across 0.0-1.0",
        },
        {
            "rule": "consistency_steps=3 is a promising lead: SNR +3.4dB and WER→0.0, but THD +8.4%. Worth revisiting paired with THD reduction.",
            "evidence": "manual_cs3_001: net composite -0.0106 but individual metrics show clear trade-off",
        },
        {
            "rule": "Latent cosine similarity of 1.0 does NOT guarantee good audio. Mimi decoding introduces divergence via streaming state accumulation.",
            "evidence": "KNOWLEDGE_INDEX lesson #1 + research-advisor reports: streaming vs batch correlation ~-0.04",
        },
        {
            "rule": "ALL SEANet layers must use streaming mode. Partial streaming causes correlation to drop from 0.69 to 0.13.",
            "evidence": "KNOWLEDGE_INDEX lesson #3: forward method used streaming for ConvTranspose1d but batch for Conv1d",
        },
        {
            "rule": "EOS detection diverges on long sequences (>17 tokens): Rust detects 2-4 frames earlier. Root cause is cumulative precision error. Threshold -4.0 is correct.",
            "evidence": "KNOWLEDGE_INDEX lesson #7: e.g., step 145 vs 166",
        },
        {
            "rule": "Voice conditioning order must be [voice_embedding, text_embedding] concatenated, not added or reversed. Attention is order-sensitive.",
            "evidence": "KNOWLEDGE_INDEX: architecture decision, verified against Python implementation",
        },
        {
            "rule": "Config-level changes likely plateau around score 0.82-0.85. Code-level changes (Tier 2: FlowNet, Mimi decoder) needed to break through.",
            "evidence": "Temperature sweep exhausted, consistency_steps and top_p explored with minimal gains",
        },
    ]

    for p in principles:
        mem.add_rule(p["rule"], p["evidence"])

    # ── Safe Parameter Ranges ──────────────────────────────────────────────

    mem.set_safe_range(
        "temperature", 0.25, 0.7, 0.75,
        notes="<0.25 too deterministic; >0.75 WER degrades. Peak at 0.7 from 20-point sweep."
    )
    mem.set_safe_range(
        "top_p", 0.85, 0.9, 0.9,
        notes=">0.9 catastrophic (40% WER at 0.95). Stay at 0.9."
    )
    mem.set_safe_range(
        "consistency_steps", 1, 2, 3,
        notes="2 is optimal. 3 is mixed (SNR+, THD-). 1 untested."
    )
    mem.set_safe_range(
        "speed", 0.9, 1.0, 1.1,
        notes="Untested, conservative range. Default 1.0."
    )
    mem.set_safe_range(
        "eos_threshold", -5.0, -4.0, -3.5,
        notes="-4.0 documented correct. Not a tuning parameter for autotuning (code-level)."
    )

    # ── Interaction Rules ──────────────────────────────────────────────────

    mem.add_interaction_rule(
        "Temperature and consistency_steps interact: more denoising steps amplify temperature-induced differences. Re-sweep temperature when changing consistency_steps.",
        params=["temperature", "consistency_steps"],
        evidence="manual_cs3_001 only tested at temp=0.7; interaction untested",
    )
    mem.add_interaction_rule(
        "WER and THD are inversely coupled in composite score. Changes that improve WER (~62% effective weight) often regress THD. Accept mixed results where composite delta > -0.01.",
        params=["wer", "thd_percent"],
        evidence="Consistent pattern across temperature sweep experiments",
    )

    # ── Methodology Guidance ───────────────────────────────────────────────

    mem.set_methodology_guidance("high_wer", {
        "summary": "WER > 0.05 — intelligibility is the bottleneck",
        "tier1": "Lower temperature in small steps (max 0.05), verify top_p=0.9, check EOS threshold",
        "tier2": "Check tokenizer edge cases, verify voice conditioning order",
    })
    mem.set_methodology_guidance("high_thd", {
        "summary": "THD > 30% — harmonic distortion is the bottleneck",
        "tier1": "Try consistency_steps+1 (but watch WER). Speed adjustments may help.",
        "tier2": "Investigate Mimi decoder precision (f32→f64 in upsampling), post-synthesis low-pass filter",
    })
    mem.set_methodology_guidance("low_snr", {
        "summary": "SNR < 24dB — signal quality is weak (but only 15% weight)",
        "tier1": "Deprioritize unless all other metrics are optimized. Only 3.4dB variance across full temp range.",
        "tier2": "Check for numerical instability in attention softmax, FlowNet noise schedule",
    })
    mem.set_methodology_guidance("high_mcd", {
        "summary": "MCD > 6dB — spectral mismatch with reference",
        "tier1": "Fine-tune temperature around optimal (±0.05 steps)",
        "tier2": "FlowNet time embedding, Mimi decoder architecture",
    })
    mem.set_methodology_guidance("plateau", {
        "summary": "Composite score plateaued — config changes exhausted",
        "tier1": "Verify all metrics are being measured (MCD, correlation). Try unexplored parameter combos.",
        "tier2": "Move to code-level changes: FlowNet noise schedule (linear→cosine), attention softmax temperature, Mimi decoder precision",
    })

    # ── Sensitivity Rankings ───────────────────────────────────────────────
    # Computed from temperature sweep data (the only complete sweep)

    mem.set_sensitivity(
        "temperature", 0.15,
        evidence="20-point sweep: score ranged 0.71-0.83 across 0.0-1.0"
    )
    mem.set_sensitivity(
        "top_p", 0.01,
        evidence="Only two data points (0.9 and 0.95). 0.95 was catastrophic, not a gradient signal."
    )
    mem.set_sensitivity(
        "consistency_steps", 0.04,
        evidence="Two data points (2 and 3). Small composite delta (-0.0106) for +1 step."
    )
    mem.set_sensitivity(
        "speed", 0.0,
        evidence="Not yet swept. No data."
    )

    # ── Mark as bootstrapped ───────────────────────────────────────────────

    mem.mark_bootstrapped()

    # Print summary
    print(f"\nBootstrap complete:")
    print(f"  Rules learned: {len(mem.data.get('rules_learned', []))}")
    print(f"  Safe ranges: {len(mem.data.get('safe_ranges', {}))}")
    print(f"  Interaction rules: {len(mem.data.get('interaction_rules', []))}")
    print(f"  Methodology guidance: {len(mem.data.get('methodology_guidance', {}))}")
    print(f"  Sensitivity rankings: {len(mem.data.get('sensitivity_rankings', {}))}")
    print(f"  Existing experiments preserved: {len(mem.data.get('experiments', []))}")


if __name__ == "__main__":
    memory_path = Path("autotuning/memory.json")
    bootstrap(memory_path)
