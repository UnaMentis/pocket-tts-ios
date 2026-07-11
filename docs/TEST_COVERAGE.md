# Capability × Test-Coverage Map

Every capability the library advertises, crossed against where it is actually
exercised. Produced by the v0.5.0 coverage audit (2026-07-07); update this
when the UDL surface or the test suites change.

**Coverage classes:** `AUTOMATED` (Rust test / CI) · `HOST-GATE` (correlation
harness `run_baseline.sh`) · `DEVICE-UI` (a human can trigger it from the demo
app screen) · `DEV-TOOL` (only `src/bin/*`) · `NONE`.

**Structural facts to know:**
- The 95 lib unit tests cover pure helpers (config validation, error display,
  audio/WAV utils, tensor shapes). `tests/engine_integration.rs` adds
  engine-level tests (load, unload→error, dangling voice→error, cancel
  mid-stream, synthesize-after-cancel) that run wherever model weights exist
  locally and **skip vacuously in CI** (no weights there).
- CI never synthesizes audio. The strongest quality gate — noise-matched
  waveform correlation vs the Python reference — is the host harness and the
  demo's Compare tab, both run manually per `docs/RELEASE_CHECKLIST.md`.

## Matrix

| Capability | Coverage | Where | Notes / gap |
|---|---|---|---|
| `version()` | AUTOMATED | `src/lib_tests.rs` | |
| `build_info()` | AUTOMATED, DEVICE-UI | `tests/engine_integration.rs`; demo footer | |
| `available_voices()` | AUTOMATED | `src/lib_tests.rs` | Canonical full set only — UIs must use `loaded_voices()` |
| constructor (model load) | AUTOMATED*, DEVICE-UI, DEV-TOOL | integration tests; demo load | *model-gated |
| `is_ready()` | AUTOMATED* | integration tests | |
| `model_version()` | NONE (trivial) | returns crate version | Does **not** identify v1-vs-v2 model; accept |
| `parameter_count()` | NONE (hardcoded) | `src/engine.rs` | Informational only; accept |
| `loaded_voices()` | AUTOMATED*, DEVICE-UI | integration tests; demo picker | Source of truth for UIs |
| `configure()` | AUTOMATED (validation), DEVICE-UI | `src/config_tests.rs`; demo | |
| `get_config()` | NONE | — | Trivial getter; accept |
| `synthesize()` (sync WAV) | AUTOMATED*, DEVICE-UI, DEV-TOOL | integration tests; demo Sync mode | |
| `synthesize_with_voice()` | AUTOMATED* (error paths) | integration tests | Happy path via demo voice picker + synthesize |
| `synthesize_noise_matched()` | HOST-GATE, DEVICE-UI | `run_baseline.sh`; demo Compare tab | **Primary release quality gate** |
| `start_true_streaming()` | AUTOMATED*, DEVICE-UI, DEV-TOOL | cancel test; demo Stream mode; latency bench | |
| `cancel()` | AUTOMATED*, DEVICE-UI | integration test; demo Stop button | Was a real shipped bug; keep both |
| `set_reference_audio()` | by design returns not-implemented | `src/engine.rs` | Documented in CHANGELOG |
| `clear_reference_audio()` | NONE | — | No-op without cloning; accept |
| `decode_latents()` | NONE | — | Reference tooling; accept |
| `unload()` | AUTOMATED* | integration test (unload→`ModelNotLoaded`) | Memory-reclaim not asserted |
| `TtsConfig.voice_index` | DEVICE-UI (audible), AUTOMATED* (errors) | demo picker; integration tests | Per-voice numerical parity never verified — audible-only |
| `TtsConfig.temperature` | range-validated only | `src/config_tests.rs` | Effect on output not asserted |
| `TtsConfig.top_p` | **INERT** | — | Silent no-op; documented (UDL, INTEGRATION, CHANGELOG) |
| `TtsConfig.speed` | **INERT** | — | Silent no-op; documented (UDL, INTEGRATION, CHANGELOG) |
| `TtsConfig.consistency_steps` | HOST-GATE (=1), DEVICE (=2) | harness / demo defaults | Step-count effect not directly asserted |
| `use_fixed_seed`/`seed` | HOST-GATE (foundational) | correlation ≈1.0 requires it | |
| `on_audio_chunk` | AUTOMATED*, DEVICE-UI | cancel test; demo streaming | |
| `on_progress` | fires; never asserted | demo handler ignores it | Accept (coarse 0.5/1.0 values only) |
| `on_complete` | AUTOMATED*, DEVICE-UI | cancel test asserts absence; demo streaming | |
| `on_error` | wired; hard to trigger | — | Only fires on mid-stream engine error; accept |
| Error variant reachability | AUTOMATED* (`ModelNotLoaded`, `InvalidVoice`, `InvalidConfig`) | integration tests | Others (`InferenceFailed`, `AudioEncodingFailed`, …) unprovoked; accept |
| WAV validity / 24 kHz mono | AUTOMATED | `src/audio_tests.rs` | |
| Streaming = raw Float32-LE PCM | DEVICE-UI | demo parses chunks | Documented in INTEGRATION.md |
| v1 vs v2 model dirs | HOST-GATE | `run_baseline.sh` env overrides | |
| Repeated synthesis stability | AUTOMATED* (2 runs), DEVICE (manual step 8) | cancel test tail; script below | |
| Concurrency (mutex serialization) | NONE | — | UI serializes calls; accept, engine is mutex-guarded |
| Empty / unicode / very long input | DEVICE (manual step 9) | script below | Engine-level behavior on `""` untested |

`AUTOMATED*` = model-gated: runs locally where weights exist, skips in CI.

## On-device manual test script (physical iPhone)

Run top to bottom in the demo app; each step names its pass condition. This is
the script behind gate 2 of `docs/RELEASE_CHECKLIST.md`.

1. **Launch.** Model auto-loads; status shows "Model loaded in N.NNs"; the
   build-provenance line is visible at the bottom (correct version + git SHA,
   no `-dirty` for a release build).
2. **Voice list.** Picker lists exactly the voices shipped in the bundled
   model directory — no more, no fewer.
3. **Sync synthesis.** Mode=Sync, Synthesize: waveform renders, "Sync
   complete", RTF reported.
4. **Playback.** Play: intelligible speech, correct pitch/rate.
5. **Streaming + TTFA.** Mode=Stream, Synthesize: "Streaming complete!" with
   TTFA ≤ 300 ms and chunk count > 1.
6. **Cancel mid-stream.** Long paragraph, Stream, Synthesize, then tap **Stop
   Synthesis** while running: status "Synthesis cancelled" promptly; the app
   stays responsive; a following synthesis completes normally.
7. **Per-voice sweep.** Every voice in the picker: Synthesize + Play —
   audibly distinct, correct-gender speech; no errors.
8. **Repeat stability.** Same text 5× alternating Sync/Stream: clean audio
   every time; Memory in the resources card does not climb monotonically.
9. **Input robustness.** One word; a long paragraph; text with emoji and
   accented characters: audio or a graceful error each time — never a hang or
   crash.
10. **Reload cycling.** Toolbar Reload Model 3×, synthesizing after each:
    loads succeed; memory returns near baseline.
11. **Compare tab parity.** Each phrase: "Generate with Rust TTS" →
    correlation vs Python reference ≥ 0.95 (v0.5.0 reads 1.0000).
12. **A/B playback.** Reference vs Current for one phrase: perceptually the
    same utterance.

Record device model, iOS version, and results in the release notes/audit log.
On ANY failure capture the exact `PocketTtsError` message before anything else.

## Accepted-untested (deliberate, v0.5.0)

Low-consequence surface with no consumers: `get_config`, `parameter_count`,
`model_version` (crate version only), `clear_reference_audio`,
`decode_latents`, `on_progress` values, exotic error variants, engine-level
empty-string input, concurrent-call stress. Revisit if a consumer starts
depending on any of them.
