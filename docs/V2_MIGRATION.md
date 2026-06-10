# Pocket TTS v2 Migration — Strategy & Spec

**Goal:** Port the new Pocket TTS (v2.0.0/v2.1.0) to iOS — primary `english_2026-04`, then
multilingual 6L (`italian`, `german`, `spanish`, `portuguese`) — **without losing the 1.000
waveform correlation** we achieved on `english_2026-01`.

**Status:** Phase 1 in progress. Hard blocker: HuggingFace auth for gated weight downloads.

Last updated: 2026-06-06.

---

## 1. Decision (approved)

- **Scope:** English (`english_2026-04`) + multilingual 6L. The `*_24l` undistilled models are
  **out** for iOS (Kyutai flags them as possibly too slow for real-time CPU — our core value prop).
- **Approach:** **Config-driven adaptation** — model reads the upstream YAML config (dims, layers,
  `insert_bos_before_voice`, `inner/outer_dim`) instead of hardcoding constants; reuse the
  correlation-verified Mimi/SEANet/FlowNet modules; add only the genuine v2 deltas.
- **Why this is the *uncompromising* route, not just the easy one:** the quality guarantee comes
  from the **validation gate**, not the coding style. Our modules are numerically proven to 1.000
  correlation; a from-scratch rewrite would force re-deriving all 8 hard-won fixes (each a new bug
  risk). Config-driven keeps verified numerics and changes only what upstream changed.

### Non-negotiable quality gates
1. **Per-layer parity** (not just end-to-end): every shipped variant gets per-layer cosine-similarity
   dumps confirmed against *its own* Python reference, so a config flag can't paper over a real
   numerical difference.
2. **1.000 target / 0.95 floor, noise-matched, per language.** No variant ships below the bar.
3. **Hunt the coincidence bug:** hardcoded constants that *happen* to be equal today (e.g. two
   different concepts both = 512) and would silently break when generalized. Per-layer dumps catch this.

---

## 2. What actually changed upstream (v1.1.x → v2.x)

Upstream published an explicit migration guide for alternative implementations:
commit `c90fc8ced65ce71b99b610969704b01e322df05f` with inline comments. Verified locally against
the bundled v2.1.0 package source in `validation/.venv-v2/.../pocket_tts/`.

### 2.1 The 6L models are all architecturally identical
`english_2026-04`, `english` (alias), `italian`, `german`, `spanish`, `portuguese` share:
- `flow_lm.insert_bos_before_voice: true`
- flow depth 6, dim 512; transformer d_model 1024, num_heads 16, num_layers 6; n_bins 4000
- mimi inner_dim **32**, outer_dim 512; mimi transformer d_model 512, num_heads 8, num_layers 2
- seanet ratios [6,5,4], etc.

They differ **only** in `weights_path` and `tokenizer_path` (per-language). ⇒ Once
`english_2026-04` is validated, each language is a pure **weight + tokenizer swap**.

### 2.2 Deltas vs our current `english_2026-01`
| Field | english_2026-01 (current) | english_2026-04 (new) | Impact on text→audio decode |
|---|---|---|---|
| `flow_lm.insert_bos_before_voice` | `false` | **`true`** | **NEW code** — see §2.3 |
| `mimi.inner_dim` | 512 | **32** | **None on decode.** Only reshapes Mimi *encoder* downsample (voice-clone encoding). Decode uses `outer_dim`=512 (unchanged). |
| `pad_with_spaces_for_short_inputs` | `true` | (absent) | Text-preprocessing flag; verify behavior |
| weights | `languages/english_2026-01/` | `languages/english_2026-04/` (219MB) | new retrained weights |
| tokenizer | shared root | per-language | new tokenizer.model per language |

**Headline:** the **Mimi decode pipeline did not change** (causal+context mask, SEANet streaming,
overlap-add all transfer unchanged). The only new decode-affecting logic is `bos_before_voice`.

### 2.3 The `bos_before_voice` delta (the one genuinely new piece)
- `flow_lm.py:81` `self.bos_emb = Parameter(randn(ldim))` — existing latent-space BOS (ldim=32). We
  already load this as `bos_emb`.
- `flow_lm.py:85` `self.bos_before_voice = Parameter(randn((1,1,dim)))` — **NEW**, in transformer/
  hidden space (dim=1024).
- `tts_model.py:890-894`: voice prompt is `_encode_audio(...)` → then
  `prompt = cat([bos_before_voice, prompt], dim=1)` (prepend along sequence) → then
  `text_embeddings = cat([text_embeddings, audio_conditioning], dim=1)` (tts_model.py:356).
- **Voice files:** predefined/`.safetensors` voices load as full model *states* via
  `_import_model_state` (tts_model.py:845-869) and **already bake in** `bos_before_voice`; the
  prepend at 893-894 only runs for fresh audio encoding. ⇒ We must use the **v2** voice files in
  `languages/<lang>/embeddings/`; our v1 `alba.safetensors` is **not** compatible.

### 2.4 Other v2 changes (not required for parity, noted)
- `ConvDownsample1d` gains `out_dimension`, `ConvTrUpsample1d` gains `in_dimension`
  (resample.py) — only the downsample side changes for 6L (encoder); decode upsample is 512→512.
- int8 dynamic quantization, transformer fusion, comma-splitting of long sentences — optional perf/UX.

---

## 3. New repos & artifacts
- `kyutai/pocket-tts` — **gated (`gated: auto`)**, with voice cloning. Layout:
  `languages/<lang>/model.safetensors@<rev>`, `.../tokenizer.model`, `.../embeddings/<voice>.safetensors`.
- `kyutai/pocket-tts-without-voice-cloning` — tokenizers etc.
- Pinned revisions live in each YAML (`@<sha>`).

---

## 4. Environment
- Proven v1 reference env preserved untouched: `validation/.venv` (uv-managed, pocket-tts **1.0.3**,
  torch 2.10). Reproduces existing `reference_outputs/`. Freeze: `/tmp/validation_venv_v1.0.3_freeze.txt`.
- New v2 env: `validation/.venv-v2` (pocket-tts **2.1.0**, torch 2.12). Has all 12 model configs.
- **Blocker:** `english_2026-04` (and every v2 model, incl. `english_2026-01` in the new layout)
  requires HF auth to download. No token configured locally. v1 cache only has the old flat
  `tts_b6369a24.safetensors`, which v2.1.0's loader does not use.

---

## 5. Phased plan
1. **Evaluate** — [BLOCKED on auth] download `english_2026-04`; A/B vs `_01` on canonical phrases;
   capture reference audio + noise tensors + per-layer dumps.
2. **Adapt** — config-driven loader (parse YAML); implement `bos_before_voice`; load v2 weights/voices.
3. **Validate to 1.000** — re-run noise-matched + per-layer methodology until parity bar met.
4. **iOS** — rebuild XCFramework, wire demo, screenshot-verify on simulator (CLAUDE.md hard req).
5. **Multilingual** — weight/tokenizer swaps for it/de/es/pt; validate each to the bar.

## 5b. Critical code-truth findings (verified against source, not summaries)
- **Confirmed "before" baseline (2026-06-06):** clean release rebuild reproduces
  `english_2026-01` phrase_00 **CORRELATION = 1.000000** (frame median 1.0, 45/45 frames > 0.9).
  This is the protected anchor for the refactor.
- **`src/models/seanet.rs` is DEAD CODE.** Its `SEANetConfig` (strides `[8,5,4,2,2]`, =640×) is never
  instantiated. The operative decode SEANet is a *separate* `SEANetDecoder` defined inside
  `src/models/mimi.rs:851`, built from `decoder.model.*` (`SEANetDecoder::new(vb.pp("decoder"))` at
  mimi.rs:978), with ratios `[6,5,4]` (=120×) × the 16× Mimi upsampler = 1920 = 24000/12.5. This
  matches the upstream `mimi.seanet` YAML block. ⇒ the config-driven loader must NOT try to drive
  `seanet.rs`; the real decode dims live in mimi.rs and are **identical across all 6L v2 models**, so
  the Mimi decode path needs **no change** for v2. (seanet.rs is a separate cleanup candidate.)
- **Loader injection point:** `src/models/pocket_tts.rs:60-70` (`FlowLMConfig::default()` /
  `MimiConfig { latent_dim, ..default() }`). This is the single place to make config-driven.
- **`config.rs::ModelManifest` has `vocab_size: 32000`** (wrong/unused; real vocab = n_bins+1 = 4001).
  Don't propagate it; treat n_bins from the YAML as source of truth.
- **Design refinement (uncompromising):** since all in-scope 6L variants share identical dims, the
  loader will *both* drive the genuinely-varying fields (`insert_bos_before_voice`, weights/tokenizer
  paths, num_layers for any future 24L) **and assert** that config dims equal what the verified code
  expects — so a coincidental-constant mismatch fails loudly instead of silently. Fall back to
  english_2026-01 defaults when no config file is present (preserves current behavior exactly).

## 5c. THE central porting finding — voice representation changed (verified 2026-06-06)
Weight diff v1 → english_2026-04: only 3 tensors differ; the other 211 are shape-identical.
- NEW `flow_lm.bos_before_voice (1,1,1024)`
- `flow_lm.speaker_proj_weight (1024,512)→(1024,32)`  (voice-encode projection — offline only)
- `mimi.downsample.conv.conv.weight (512,512,32)→(32,512,32)`  (Mimi **encoder** — not decode)

**Voice files are a different representation:**
- v1 `alba.safetensors`: `audio_prompt (1,125,1024)` — a voice *embedding sequence*; our Rust runs it
  through the transformer to fill the KV cache (flowlm.rs `generate_latents` Phase 1, ~L354-382).
- v2 `alba.safetensors`: a precomputed **KV-cache state** — `transformer.layers.{0..5}.self_attn/cache
  (2,1,126,16,64)` + `/offset (1,)`. 126 = 1 `bos_before_voice` + 125 voice positions; `speaker_proj`
  and `bos_before_voice` are **baked in offline**.

**Consequence (scopes the whole port):** for synthesis with predefined voices, we do NOT need
`speaker_proj_weight`, `bos_before_voice`, or the encoder `downsample` at runtime. We need to **load the
precomputed KV state directly into our `KVCache`** and start text prompting at offset 126. Decode path,
FlowNet, Mimi, SEANet, transformer weights are all unchanged. v1 path stays intact (branch on voice
file format: `audio_prompt` ⇒ v1 embedding path; `*self_attn/cache` ⇒ v2 state-load path).

**Layout mapping:** v2 cache `(2,1,S,H,D)` = [K|V, batch, seq, heads, head_dim]; our `k_cache`/`v_cache`
are `[batch, heads, seq, head_dim]` ⇒ `cache[0/1].squeeze(0).transpose(1,2)`. **Risk to validate:**
whether the saved K is post-RoPE (Moshi caches post-RoPE — expected to match; per-layer dump confirms).

**Reference captured:** `validation/reference_outputs_en2026-04/` (4 phrases + 122 noise tensors, seed 42,
voice alba). Rust model dir assembled at `kyutai-pocket-ios-en2026-04/` (model + tokenizer + v2 alba).
Note english_2026-04 audio is tighter (phrase_00 2.56s/34 steps vs v1 3.6s/45) — "better short sentences".

## 5d. ✅ english_2026-04 VALIDATED to 1.000 correlation (2026-06-06)
Implemented v2 voice KV-state loading and validated `english_2026-04` against its own Python reference
(noise-matched, per-phrase seed 42+i, consistency-steps 1). **All 4 canonical phrases = CORR 1.000000**
when run individually (phrase_00, _01 ×3, _02 exact-length 63360==63360, _03). The v1 `english_2026-01`
path is untouched (still 1.000). The post-RoPE-K assumption for the saved cache held (no fix needed).

**Code changes (minimal, branch-on-format — v1 path byte-identical):**
- `src/modules/attention.rs`: `KVCache::set(k,v)` to preload a saved state.
- `src/modules/embeddings.rs`: `VoiceKvState` + `VoiceEmbedding::from_bytes` detects v2 KV-state files
  (`transformer.layers.{i}.self_attn/cache (2,1,S,H,D)` → `[batch,heads,seq,head_dim]` via
  `cache[0/1].squeeze(0).transpose(1,2)`), exposes `kv_state()`.
- `src/models/flowlm.rs` `generate_latents`: if `voice.kv_state()` is set, preload each layer's KV cache
  and skip the embedding-run (text prompting then continues at offset 126); else the v1 embedding path.
- The existing hardcoded FlowLMConfig/MimiConfig already match (dims identical), and our loader never
  reads the 3 changed tensors (`speaker_proj_weight`, `bos_before_voice`, `mimi.downsample`), so the new
  weights load as-is. Model dir: `kyutai-pocket-ios-en2026-04/`.

**Known robustness follow-up (not a production blocker):** running `test-tts` 12× in a tight loop
intermittently fails to load noise tensors (empty output or seeded-RNG fallback → deterministic-but-wrong
runs). Individually every phrase is 1.000. Likely a rapid CLI-spawn / noise-load race; production iOS uses
the in-process engine with sampled noise, so unaffected. Worth hardening the noise loader / understanding
the empty-output failure before shipping the validation harness as a loop.

**Implication for multilingual:** all 6L configs are dim-identical, and per-language differs only in
weights + tokenizer (+ v2 KV-state voices). So it/de/es/pt should work by assembling a model dir per
language — likely zero further code changes. Config-driven refactor is now a maintainability/explicitness
improvement (dim-assertions, model identity), not required for correctness.

## 5e. ✅ On-device iOS E2E verification (2026-06-06, iPhone 17 Pro sim)
Clean XCFramework rebuild (`cargo clean && build-ios.sh`) → copied framework+bindings into demo →
bundled `Models/` = english_2026-04 (209M weights + v2 alba KV-state) and `ReferenceAudio/` = v2 Int16
refs+manifest → cleared derived data → built+installed+launched. Drove the app via simulator UI:
- **Model loads on-device in 0.29s.**
- **Stream synth:** TTFA **159ms** (✓ "Meets baseline ≤200ms"), RTF **2.70x**, 3 chunks.
- **Sync synth:** 4.24s audio in 1.33s = RTF **3.20x** (✓ 3–4x target); saved output analyzed =
  **full-length healthy speech** (peak 0.607, 46% voiced, 1288× envelope dynamic range — real prosody).
- **Compare tab** loads the 4 v2 reference phrases and generates on-device per phrase.

**Pre-existing build bug fixed (not v2-related):** `PocketTTSSwift.swift:196` called `startStreaming`;
the binding method is `startTrueStreaming`. The bad call made the `AsyncThrowingStream` closure fail to
typecheck → misleading "expects 0 arguments" error. One-line fix → BUILD SUCCEEDED.

**IMPORTANT — in-app "vs Reference (Python)" correlation is ~0 BY DESIGN, not a defect.** The iOS engine
synthesizes with *randomly sampled* noise; the Python reference used one captured noise sequence. Two
valid utterances of the same text with different noise are ~uncorrelated in raw waveform (we saw -0.05).
The numerical 1.000 parity is the *host noise-matched* measurement (§5d); on-device the valid gates are
functional + latency + the user's ear-check (all pass). To make the app itself a numerical gate, expose
a noise-injection synth path over UniFFI and bundle the captured noise (future enhancement).

## 5f. ✅ Test client brought into compliance — on-device noise-matched gate = 1.0000 (2026-06-06)
The demo's Compare tab was out of sync with our standard (it synthesized with *random* noise → ~0
correlation vs the fixed-noise Python reference, reading "-0.05 Very poor"). Fixed by adding a
noise-matched path end-to-end:
- **Rust:** `PocketTTSEngine::synthesize_noise_matched(text, voice_index, noise_dir, phrase_id, seed)`
  (engine.rs) — loads captured noise tensors, forces consistency-steps=1 + fixed seed, synthesizes,
  restores config. Exposed in `pocket_tts.udl` → binding `synthesizeNoiseMatched`.
- **Bundle:** captured noise tensors copied to `ReferenceAudio/noise/` (122 .npy); `manifest.json`
  gains `noise_id` (phrase_00..03) + `seed` (42..45) per phrase.
- **Swift:** `ReferenceTestView.generateWithRustTTS` now calls `synthesizeNoiseMatched` (was
  `startTrueStreaming`); `ReferencePhrase` gains `noise_id`/`seed`.

**Result (drove the app via simulator):** Compare tab "vs Reference (Python)" now reads **1.0000
("Excellent – Nearly Identical")** for ALL 4 phrases on-device — Short 1.0000 (1114ms), Medium 1.0000
(1013ms), Numbers 1.0000 (859ms), Question 1.0000 (417ms). The on-device numerical gate matches the
host. (Free-running Synthesize tab still uses random noise by design — that's for latency/quality, not
parity.)

## 6. Canonical test phrases (must reuse — from validation/reference_harness.py)
1. "Hello, this is a test of the Pocket TTS system."
2. "The quick brown fox jumps over the lazy dog."
3. "One two three four five six seven eight nine ten."
4. "How are you doing today?"
