# v0.5.0 Release Prep — Overnight Run Report (2026-06-10 → 2026-06-11)

Authorized scope: execute the approved release-readiness plan end-to-end (history rewrite included),
then port the remaining languages on a separate branch. No-compromise directive: full validation
rigor everywhere; take the slow path where quality demands it.

## State this morning

| Branch | Contents | Status |
|---|---|---|
| `main` (origin) | rewritten history, v2 feature commit `ce0a41d` | pushed, CI green |
| `release/v0.5.0-prep` | 9 commits: cleanup + validation + docs + v0.5.0 bump | **local only — review & push** |
| `feature/multilingual-v2` | +2 commits: multilingual validation + harness support | **local only** |

**Your morning checklist:** review `release/v0.5.0-prep` → push → CI green → merge to main →
`git tag v0.5.0 && git push origin v0.5.0` → release workflow runs → verify the zip (framework +
bindings + v1 weights only) → announce. The multilingual branch follows separately once Italian is
root-caused.

## 1. History rewrite — done, verified

- Widened purge per the red-team verdict: `validation/venv/` + `validation/mimi_debug/` + the
  committed stale `PocketTTS.xcframework/`. Fresh clone: **190MB → 15.5MB pack, 23MB on disk**
  (the old checkout alone was 463MB). 57 commits preserved; both tags force-updated **in place**
  (release assets intact — v0.4.1 zip still shows its 82 downloads); stale merged branch deleted;
  Release workflow disabled during the operation and re-enabled after; local repo reclaimed in
  place (606→16MB). Backup mirror: `~/backups/pocket-tts-mirror-2026-06-10.git` (keep until
  comfortable). Courtesy notice for fork owners: issue #2.

## 2. Code cleanup (commits `ea86d68`, `c1f1166`, `443e064`, `6129af5`, `879d40e`)

- **Dead code deleted:** `seanet.rs` (was never even compiled — no mod declaration), `ModelManifest`
  (unused, wrong vocab_size).
- **Library silent by default:** ~85 porting-era `eprintln!` removed. These weren't cosmetic — they
  flattened tensors to CPU vectors **on every frame and every flow step** (FlowNet's per-step
  `to_vec1` syncs, full-latent stats in `synthesize`, full-audio stats in both Mimi paths).
  One-shot lifecycle messages moved to `log::debug!`/`log::warn!`.
- **`diagnostics` cargo feature** gates all .npy dump tooling (the instrumentation that won the
  March parity campaign — preserved, not shipped). `cargo check --features diagnostics` kept green.
- **Real bug found & fixed:** the streaming generation path never loaded v2 KV-state voices — it ran
  the placeholder embedding through the transformer (wrong voice conditioning for v2 in streaming;
  masked because the parity gate only exercised the sync path). Voice priming is now ONE shared
  function (`FlowLM::prime_voice_conditioning`) used by both paths — they can no longer diverge.
- **Fail-fast validation:** `verify_model_shapes()` checks 8 load-bearing tensor shapes against the
  safetensors header (header-only read) before any module is built, with errors naming the supported
  models. Voice files are format/shape/dim-checked; KV-state layer count must match the model.
- **Deterministic noise matching:** noise-matched generation stops exactly where the captured parity
  region ends and never silently falls back to RNG; noise files validated at load (latent_dim
  elements). **12 consecutive runs produce byte-identical output** — the intermittent loop failure
  documented in V2_MIGRATION §5d is resolved.
- **Mutex hygiene:** poisoned-lock recovery or typed errors replace `unwrap()` in engine.rs/mimi.rs.

## 3. The phrase_02 mystery — solved

The March audit flagged an unexplained catastrophic phrase_02 correlation (0.0112) that was never
re-measured. Root cause found tonight: `run_baseline.sh` had the WRONG SENTENCE for phrase_02
("I can speak with different voices…" — not a reference phrase). The harness was correlating two
different utterances. With the fix, **v1 phrase_02 = 1.000000** like everything else. The script now
reads phrases from the reference manifest (single source of truth) and honors MODEL_DIR/NOISE_DIR/
REFERENCE_DIR overrides.

## 4. Verification gate results (all evidence committed)

| Check | Result | Evidence |
|---|---|---|
| Noise-matched correlation, v1 × 4 phrases | **1.000000 each**, exact reference lengths | `docs/audit/correlation-v0.5.0-2026-06-11.txt` |
| Noise-matched correlation, v2 × 4 phrases | **1.000000 each** | same artifact |
| Determinism (12× loop, v2 phrase_00) | byte-identical (1 unique MD5) | report §2 |
| Latency v2 (host, streaming) | **TTFA 137ms avg** (147/118/146 by length), RTF 2.94x — PASS | `benchmark-results/latency_streaming_20260610_232845.json` |
| Latency v1 (host, streaming) | TTFA 252ms avg, RTF 2.64x — PASS | `..._233050.json` |
| Tests / lints | 95/95, clippy `-D warnings` clean, both feature configs build | pre-commit hooks on every commit |
| iOS (CLAUDE.md hard req) | clean `cargo clean` + XCFramework rebuild → demo build → sim install → **Compare tab 1.0000** on two phrases → screenshot `/tmp/ios-verify-v0.5.0-compare-phrase00.png` → app left running | on simulator now |

January's TTFA numbers (~1,040ms average, growing with phrase length) are obsolete — that pathology
is gone; TTFA is now flat across phrase lengths.

## 5. Weights distribution (Phase 3)

`scripts/download-model.py --model v2` downloads english_2026-04 from the gated `kyutai/pocket-tts`
repo (tested end-to-end; all 8 v2 voices fetched — they all exist upstream, so the local v2 model
dir now has the full voice bank). Friendly gate-acceptance instructions print on 401/403. **Policy
recorded in RELEASE_PROCESS/README: v2 weights are never redistributed in release artifacts; the
release zip bundles v1 (public) weights only, as before.** The committed-xcframework path is now
gitignored so it can't come back.

## 6. Documentation (Phase 4) — all in sync

CHANGELOG (0.5.0 entry + history-rewrite notice + date-typo fixes), README (measured numbers with
dates/platforms, model-versions section, obtaining-weights section), PORTING_STATUS (1.000 status +
v2 section + corrected metric-semantics note), V2_MIGRATION (header un-staled, per-model status
table, resolved-issue updates), INTEGRATION (v1-vs-v2 + HF auth), LATENCY_TESTING (fresh numbers,
obsolescence note), ios-harness README (Compare tab docs), project-story (new chapter: perfect
correlation & the v2 era; stale "What's Next" replaced with multilingual → quantization/fusion
roadmap), CONTRIBUTING (validation requirements), RELEASE_PROCESS (correlation-gate step + weights
policy), plus a stale-string sweep across validation/ and prompt docs.

**Left for you (permission-blocked for me):** `.claude/skills/progress/SKILL.md` and
`.claude/skills/research/SKILL.md` still present 0.839 as the current metric to agent sessions —
two small manual edits.

## 7. Multilingual (branch `feature/multilingual-v2`)

All four 6L language packs downloaded (model + tokenizer + 8 voices each, gitignored dirs);
Python references generated per language with **native-language phrases**
(`validation/phrases/<lang>.json`, harness gained `--phrases-file`); references + captured noise
committed (~9MB, same pattern as English ground truth); validated noise-matched against the
**unchanged** v0.5.0 engine — zero Rust code changes needed, exactly as V2_MIGRATION predicted:

| Language | Result |
|---|---|
| German | **1.000000 × 4 phrases** |
| Portuguese | **1.000000 × 4 phrases** |
| Spanish | 1.000000 × 2, 0.999992 / 0.999999 (float drift — effectively at target) |
| Italian | 3/4 at 1.000000; **phrase_01 = 0.996785 — held back** |

**The Italian finding (no-compromise handling):** deterministic, localized divergence at frames
27-28 of phrase_01 — one voiced frame reaches max_abs_diff 0.216, then frames return to 1.000.
The recovery rules out autoregressive latent divergence (a diverged latent would poison everything
after it), pointing at the Mimi decode of 1-2 specific frames. Regenerating the Python reference
reproduces 0.996785 *identically*, so it is a real Python↔Rust numeric difference, not capture
contamination. It is above the 0.95 floor but below the 1.000 target, so per the project's own gate
("no variant ships below the bar") **Italian is not declared validated**. Investigation plan:
per-layer dumps (`--features diagnostics`) on that phrase, March-style. Raw artifact:
`docs/audit/correlation-multilingual-2026-06-11.txt`.

## 8. Known minor items (documented, deliberately not changed tonight)

- `engine.rs` `model_version()` returns the hardcoded "1.0.2" regardless of loaded model — an
  API-visible wart; changing UDL-adjacent behavior hours before a release wasn't worth the risk.
- The two `.claude/skills/` files above.
- gitleaks isn't installed locally, so the pre-commit secrets scan self-skips (`brew install gitleaks`).
- Post-release performance track (from upstream v2, each gated by parity re-validation): int8
  dynamic quantization, transformer fusion, comma-splitting for long sentences.

## Commit inventory

`release/v0.5.0-prep` (on rewritten main `ce0a41d`):
`ea86d68` dead code → `c1f1166` logging/diagnostics + streaming-voice fix → `443e064` fail-fast
validation → `6129af5` noise semantics + harness fix + correlation artifact → `879d40e` v2 download
+ gitignore → `43d6294` docs batch → `ff64e6e` version 0.5.0 → `cf4bc5a` bench evidence → this report.

`feature/multilingual-v2`: `f8143e5` multilingual validation + harness `--phrases-file`.
