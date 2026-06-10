# Pocket TTS iOS — Full Top-Down Project Review

**Date:** 2026-06-10
**Scope:** Complete fresh review — project goal, health, methodology, results — plus an explicit review of the v2 migration work done the week of 2026-06-06. Nothing was assumed; every claim was checked against artifacts in the repository.
**Method:** Three independent deep-dive reviews (port health, validation methodology/results, recent v2 work) plus direct verification of build health, git contents, and key documents.

---

## Executive Summary

**Verdict: This project is real, healthy, and successful — with documentation drift and repo-hygiene debt as its main weaknesses.**

The core bet paid off. The Rust/Candle port of Pocket TTS works on iOS: it compiles cleanly, passes 96/96 tests with zero clippy warnings, is published as v0.4.1 on GitHub with working CI, and produces intelligible, natural speech on-device at real-time-factor 2.7–3.2x. The hardest claim — bit-level numerical parity with the Python reference (1.000 waveform correlation, noise-matched) — is supported by a coherent, well-documented diagnostic chain and was independently re-confirmed on 2026-06-06 for both the v1 and v2 models.

The week's v2 migration work (english_2026-04) is **high quality and essentially complete for English**: minimal, surgical code changes; 1.000 correlation on all 4 canonical phrases both on host and on-device; v1 path preserved byte-identical; on-device TTFA of 159ms meeting the ≤200ms target. It is, however, **entirely uncommitted**, and the new 209MB model directory has no .gitignore protection.

The most significant issues found:

1. **121MB Python virtualenv committed to git** (2,835 files under `validation/venv/`) — the single biggest repo-hygiene problem; git pack is 606MB.
2. **Documentation drift**: PORTING_STATUS.md still says correlation is 0.839; V2_MIGRATION.md's status header says "Phase 1 blocked on HF auth" while its own body documents full completion.
3. **Raw measurement artifacts weren't saved in March** when 1.000 was first claimed — the claim rested on log entries until the June 6 re-confirmation.
4. **Narrow validation corpus**: 4 short English phrases, 1 voice. No perceptual (MOS) testing, no long-text or edge-case coverage.

---

## Part A — Project Goal & Health

### A.1 Identity and goal (verified)

The project is what it claims to be: a native iOS port of Kyutai Pocket TTS (~100M params: FlowLM ~70M → FlowNet ~10M → Mimi decoder ~20M) using Rust/Candle for inference and UniFFI for Swift bindings, born from the discovery that upstream Pocket TTS could not run in the Unamentis iOS app directly.

- **Published**: `github.com/UnaMentis/pocket-tts-ios`, tags v0.4.0 (beta) and v0.4.1, release artifacts (XCFramework + Swift bindings), CI badges.
- **Version consistency**: Cargo.toml, CHANGELOG.md, README, and git tags all agree on 0.4.1.
- **Attribution chain is clean**: Kyutai (CC-BY-4.0 weights / MIT code) → babybirdprd Rust port (MIT) → this iOS integration (MIT). Licenses compatible.

### A.2 Build & test health (verified directly, on the current working tree including v2 changes)

| Check | Result |
|---|---|
| `cargo check` | ✅ Clean |
| `cargo test` | ✅ 96/96 pass |
| `cargo clippy -- -D warnings` | ✅ Zero warnings |
| TODO/FIXME/HACK markers in src/ | ✅ Zero |

### A.3 Code health

Architecture is coherent and matches the documented pipeline (Tokenizer → FlowLM → FlowNet → Mimi/SEANet). CI is complete (rust.yml, ios.yml, release.yml, security.yml, validation.yml) with lint gates and a 70% coverage threshold. Documentation is exceptional — PORTING_STATUS.md alone is a 1,183-line technical journal of the port.

Known debt (all minor, none functional):

- **`src/models/seanet.rs` is dead code** (122 lines, never instantiated). Confirmed independently in V2_MIGRATION.md §5b: the operative SEANet lives inside `src/models/mimi.rs:851`. Flagged as a cleanup candidate; still present.
- **~17 debug `eprintln!` statements** in `src/models/pocket_tts.rs` (plus one in mimi.rs) — porting-era instrumentation that should be removed or gated before a user-facing release.
- **~35 `unwrap()` calls**, mostly in tests/bins (fine); a handful of Mutex-lock unwraps in `engine.rs` (acceptable, could be hardened).
- **`config.rs::ModelManifest.vocab_size: 32000` is wrong/unused** (real vocab = 4001); documented in V2_MIGRATION.md as a do-not-propagate trap.

### A.4 Repo hygiene — the big one

- **`validation/venv/` is committed: 2,835 files, 121MB** (a full Python 3.9 virtualenv including numpy test data). The `.gitignore` entry was added after the files were committed, so they remain tracked. Git pack size is 606MB, which makes cloning slow and will only get worse. Removing it requires a history rewrite (or at minimum `git rm -r --cached` going forward).
- The v1 model dir (`/kyutai-pocket-ios/`) is correctly gitignored; no model weights are tracked. Reference WAV/NPY ground-truth files in `validation/reference_outputs/` are intentionally tracked (reasonable, ~2MB).
- **New risk (this week)**: `kyutai-pocket-ios-en2026-04/` (209MB model) and `validation/reference_outputs_en2026-04/` are untracked but **not gitignored** — one careless `git add -A` away from a 200MB+ commit.

### A.5 Documentation drift

- **PORTING_STATUS.md** still reports "correlation = 0.839, target >0.95" — stale since 2026-03-22 when 1.000 was achieved.
- **V2_MIGRATION.md line 7** says "Phase 1 in progress. Hard blocker: HuggingFace auth" — contradicted by its own §5d–5f, which document full validation and on-device verification. The header was never updated.
- README describes the FlowNet stage as "MLP sampler" — simplified terminology, structurally accurate.

---

## Part B — Methodology & Results

### B.1 The validation methodology: genuinely rigorous in design

The approach evolved from naive to sophisticated, and the artifacts show real scientific discipline:

1. **Noise matching**: `validation/reference_harness.py` intercepts PyTorch's `normal_` to capture every noise tensor as .npy; Rust loads the same tensors. This is the *correct* way to compare stochastic generative models — it eliminates RNG as a variable and makes waveform correlation meaningful. (The project learned this the hard way: a January report shows correlation of −0.013 before noise matching, correctly diagnosed as meaningless.)
2. **Layered metrics**: latent cosine similarity, waveform correlation, Whisper ASR round-trip (WER), MCD/SNR/THD, signal-health checks.
3. **Binary-search debugging**: the March campaign (`docs/audit/approaches-tried.md`) is exemplary — per-layer transformer dumps proved FlowLM/FlowNet matched to 1e-6, isolating the gap to the Mimi decoder; per-block Mimi dumps isolated it to the decoder transformer (cos=0.178); sub-layer dumps found the root cause (Rust was missing Python's causal + 250-context attention mask, 62.2% of attention entries unmasked). Fix applied → 0.839 → 1.000.
4. **Honest negative results**: the log records approaches that had *zero* effect (softmax_last_dim, rope_i) and keeps them anyway as reference alignment — a sign the process wasn't chasing metrics.

The correlation journey, reconstructed from artifacts: ~0 (Jan, no noise matching) → 0.839 (Mar 18, noise off-by-one fix: Rust was consuming `noise_step_000` which Python discards) → **1.000** (Mar 21–22, Mimi decoder causal mask).

### B.2 Where the methodology's execution fell short

- **The 1.000 claim was not backed by a saved raw measurement in March.** The only raw measurement file in `docs/audit/` is `baseline-2026-03-19.txt` — *pre-fix* — showing mean 0.66 across 4 phrases with phrase_02 at a catastrophic **0.0112**. The 1.000 result lives only as a log entry in approaches-tried.md (apparently single-phrase: "45 frames" matches phrase_00). **This gap was closed on 2026-06-06**, when the v2 work independently re-confirmed v1 phrase_00 = 1.000000 (45/45 frames >0.9) on a clean rebuild, and v2 = 1.000000 on *all four* phrases. The claim is now credible — but for ~11 weeks it rested on an unverifiable log line. Verification reports should capture raw output files, always.
- **phrase_02's 0.011 pre-fix failure was never explicitly root-caused for v1.** It's plausibly explained by the missing causal mask (and v2 phrase_02 now measures 1.000 with exact length match), but no v1 post-fix per-phrase artifact exists.
- **Corpus is tiny**: 4 short English phrases (1–4 seconds), one voice (alba), seed 42. No long-form text, no edge cases (numbers were tested only as one phrase), no other voices, no perceptual/MOS evaluation. Numerical parity at 1.000 makes perceptual testing less critical (output *is* the Python output), but free-running generation (random noise, real usage) is validated only by ear and signal-health checks.
- **Advisory, not blocking, gates**: verification reports don't gate commits. The March 22 commit message claimed 1.0 without a committed measurement artifact — exactly the failure mode blocking gates prevent.
- **Composite scoring is shaky**: the autotuning scorer's weights are arbitrary, MCD baselines were admittedly calibrated to whatever Rust-vs-Python produced, and a known WER false positive (digits vs words on phrase_02) pollutes averages. This subsystem was scaffolding for the optimization loop, not the parity proof, so the impact is limited.

### B.3 Latency results — messy history, currently passing

Targets: TTFA ~200ms (≤300 acceptable), RTF 3–4x (≥2.5 acceptable).

| Measurement | Date | TTFA | RTF |
|---|---|---|---|
| Host streaming bench | 2026-01-26 | avg **1,040ms** (312ms short → 2,970ms long) | 3.55x ✅ |
| Host sync bench | 2026-03-13 | avg 1,362ms (sync = full synthesis first; not a fair TTFA) | 3.31x ✅ |
| **On-device (iPhone 17 Pro sim), v2** | **2026-06-06** | **159ms ✅** | **2.70x stream / 3.20x sync ✅** |

RTF has always met target. TTFA badly missed it in the January host benchmarks but the most recent on-device measurement meets it. These were measured under different conditions (host CLI vs in-app streaming, different phrase lengths), so the honest statement is: **the latest real measurement passes; a systematic TTFA re-benchmark across phrase lengths on the current code would settle it**. The January numbers suggest TTFA grows with phrase length, which true streaming shouldn't — worth one focused look.

### B.4 The multi-agent optimization process

The orchestration pattern (/optimize, /verify, /research, /cleanup with fresh-context iterations and report rotation in docs/audit/) demonstrably contributed: the research-advisor reports fed hypotheses, the per-layer dump strategy emerged from that loop, and both breakthrough fixes (noise off-by-one, causal mask) are traceable through its artifacts. It is not process theater. Its real weaknesses are the ones above: reports are advisory, raw outputs weren't archived, and stale documents (PORTING_STATUS.md) don't get reconciled.

---

## Part C — The Last Week's Work (v2 Migration, 2026-06-06)

### C.1 What was done

Goal: port upstream Pocket TTS v2.0.0/v2.1.0 (`english_2026-04`, then multilingual 6L) without losing the 1.000 correlation. All work is **uncommitted** on main.

**Analysis (excellent):** V2_MIGRATION.md documents a verified weight diff — only 3 of 214 tensors changed (`bos_before_voice` new, `speaker_proj_weight` reshaped, Mimi *encoder* downsample reshaped). The Mimi *decode* path is unchanged, so all 8 hard-won numerical fixes transfer. The central discovery: **v2 voice files are a different representation** — not an embedding sequence but a precomputed transformer KV-cache state (126 positions = 1 bos_before_voice + 125 voice frames) with `speaker_proj` and `bos_before_voice` baked in offline. Consequence: the runtime never needs the 3 changed tensors at all.

**Code changes (minimal and surgical, ~240 lines across 5 files):**
- `attention.rs`: `KVCache::set(k, v)` to preload saved state — opt-in, existing path untouched.
- `embeddings.rs`: `VoiceKvState` + format detection in `VoiceEmbedding::from_bytes` (v1 `audio_prompt` ⇒ embedding path; v2 `self_attn/cache` ⇒ KV-state path), with correct layout transform (`squeeze(0).transpose(1,2).contiguous()`).
- `flowlm.rs` `generate_latents`: if KV state present, preload caches and skip the voice-prompt transformer run; else the v1 path, byte-identical.
- `engine.rs` + `pocket_tts.udl`: new `synthesize_noise_matched(text, voice_index, noise_dir, phrase_id, seed)` with config save/restore — making the *iOS app itself* a numerical parity gate.
- iOS harness: Compare tab rebuilt for 3-way comparison (Python reference / saved release baseline / current build) with on-device Pearson correlation; manifest gains `noise_id`/`seed`; 122 captured noise tensors bundled.

**Validation (per V2_MIGRATION.md §5d–5f):**
- Host, noise-matched: **all 4 phrases = 1.000000** for english_2026-04 against its own fresh Python reference (pocket-tts 2.1.0 env).
- v1 regression check: clean rebuild reproduces english_2026-01 phrase_00 = 1.000000.
- On-device E2E: model loads in 0.29s; streaming TTFA **159ms**; RTF 2.70x/3.20x; full-length healthy speech verified by signal analysis; Compare tab reads **1.0000 on all 4 phrases on-device**.
- The post-RoPE-K assumption for saved caches was identified as the key risk in advance and confirmed empirically — good engineering.
- Also fixed a pre-existing build bug (`PocketTTSSwift.swift:196` called `startStreaming` instead of `startTrueStreaming`).

### C.2 Quality assessment

This is the strongest week of work in the repo's history. The scoping decision — load the precomputed KV state rather than implement `bos_before_voice`/`speaker_proj` at runtime — eliminated most of the planned port surface while *increasing* fidelity (the baked-in state can't drift). The "in-app correlation is ~0 by design with random noise" insight (§5e) shows the team understands its own metric deeply, and instead of rationalizing it, they built the noise-matched in-app gate to fix it. The code itself is clean: correct tensor layouts, bounds checks, config restore on error paths, v1 byte-identical.

### C.3 Gaps and risks in the week's work

1. **All of it is uncommitted.** ~612 insertions across 13 tracked files plus critical untracked assets, sitting on main with no branch, no commit, for 4 days. A `git checkout` mistake loses it.
2. **No .gitignore protection** for `kyutai-pocket-ios-en2026-04/` (209MB) or `validation/reference_outputs_en2026-04/`. Conversely, the harness's bundled `ReferenceAudio/noise/` (122 .npy, referenced by the committed manifest) is untracked — a fresh clone would have a Compare tab that can't run.
3. **The config-driven YAML loader — the plan's stated "Approach" (§1) — was not built.** The doc itself re-scopes it honestly (§5d: all 6L variants are dim-identical, so it's now "maintainability, not correctness"), which is defensible, but the plan's own "hunt the coincidence bug" principle (§1) argued for the dim-assertion loader. Currently nothing fails loudly if a future model's dims differ.
4. **Multilingual (it/de/es/pt) not started** — though the week's findings reduce it to per-language model-dir assembly + validation, likely zero code changes.
5. **Known robustness issue**: the validation loop intermittently fails noise loading when `test-tts` is spawned 12× rapidly (empty output or wrong-RNG fallback). Documented, plausibly a CLI race, doesn't affect the in-process iOS engine — but "deterministic-but-wrong runs" in a validation harness deserves a fix before the harness is trusted in a loop.
6. **Stale status header** in V2_MIGRATION.md (says blocked; work is done).
7. New diagnostic `eprintln!`s were added in the KV-preload path — joining the existing cleanup backlog.

### C.4 Remaining work, concretely

| Item | Effort | Urgency |
|---|---|---|
| Commit the v2 work (after .gitignore additions) | Small | **High — data-loss risk** |
| Add .gitignore entries for `kyutai-pocket-ios-*/` and `reference_outputs_en2026-04/`; decide whether to track `ReferenceAudio/noise/` | Small | **High** |
| Update V2_MIGRATION.md header + PORTING_STATUS.md correlation status | Small | Medium |
| Save raw validation outputs (per-phrase correlation logs) into docs/audit/ as the post-fix artifact | Small | Medium |
| Dim-assertion loader (fail loudly on config mismatch) | Medium | Medium |
| Multilingual: assemble + validate it/de/es/pt | Medium | When wanted |
| Harden noise loader race; remove debug eprintlns; delete seanet.rs | Small | Low |
| Purge `validation/venv/` from git (history rewrite decision) | Medium | Low but compounding |

---

## Consolidated Verdict

| Dimension | Grade | One-line justification |
|---|---|---|
| Goal & viability | **A** | On-device TTS on iOS, working, published, used in anger — the "crazy person" port succeeded. |
| Core port health | **A−** | Clean build/tests/lints/CI; minor debt (dead seanet.rs, debug prints). |
| Methodology design | **A−** | Noise matching + per-layer binary search is the right way to validate an ML port. |
| Methodology execution | **B−** | Raw artifacts not archived at the critical moment; tiny corpus; advisory gates; stale docs. |
| Results credibility | **A−** | 1.000 correlation now multiply re-confirmed (June 6, both models, host + device); latency targets met on latest measurement. |
| Last week's work | **A** | Surgical, fully validated v2 port — but commit it and protect the assets. |
| Repo hygiene | **C** | 121MB committed venv; 606MB pack; unprotected 209MB model dir. |

**Overall: a successful, well-engineered project whose engineering is ahead of its bookkeeping.** The recurring failure mode is not technical — it's that documents and archives lag reality (stale status lines, unsaved measurement artifacts, uncommitted work). Every hard claim checked in this review ultimately held up, but several were temporarily unverifiable because the proof wasn't filed. Tightening that loop — commit the work, archive raw outputs, update status headers — would bring the project's paper trail up to the standard of its code.
