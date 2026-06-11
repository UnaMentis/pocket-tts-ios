# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-06-11

### NOTICE: Git History Rewrite (2026-06-10)

The repository history was rewritten on 2026-06-10 to purge accidentally
committed artifacts (a Python venv, debug tensor dumps, and a stale committed
XCFramework). The repository is now ~15MB to clone (was ~190MB).

- **Anyone with an old clone must RE-CLONE the repository (not pull).**
- Release zips for v0.4.x are unaffected.
- See [issue #2](https://github.com/UnaMentis/pocket-tts-ios/issues/2) for details.

### Added
- Pocket TTS v2 model support (`english_2026-04`). v1 (`english_2026-01`)
  remains fully supported; the engine auto-detects which voice-file format it
  is given:
  - v1 voices: embedding sequences (`audio_prompt`, `[1, seq, 1024]`)
  - v2 voices: precomputed transformer KV-states
    (`transformer.layers.{i}.self_attn/cache`, with `bos_before_voice` and
    speaker projection baked in)
- `synthesize_noise_matched` exposed over UniFFI for on-device parity testing
- iOS demo **Compare** tab: 3-way comparison (Python reference / saved release
  baseline / current build) with on-device noise-matched synthesis via bundled
  noise tensors — the on-device parity gate (1.0000 correlation on all 4
  canonical phrases)
- `diagnostics` cargo feature gating the `.npy` tensor dump tooling
- Load-time validation: weight shape assertions fail fast with clear errors on
  incompatible models; voice files are format- and dimension-checked
- `scripts/download-model.py --model v2` downloads v2 weights from the gated
  `kyutai/pocket-tts` repo (requires accepting the Hugging Face gate)

### Changed
- Library is now silent by default; one-shot informational messages moved
  behind `log::debug`
- Latency improved substantially after removing debug instrumentation from the
  hot path (measured 2026-06-11, host Apple-silicon Mac, release build,
  streaming):
  - v2: TTFA 137ms average (short 147 / medium 118 / long 146ms), RTF 2.94x
  - v1: TTFA 252ms average, RTF 2.64x (v1 voices run a 125-position prompt
    through the transformer; v2 voices are precomputed KV states)
- Release zip still bundles v1 weights only; v2 weights are user-downloaded
  due to the Hugging Face gate and are never redistributed in release artifacts

### Fixed
- Streaming path now loads v2 KV-state voices (voice conditioning is shared
  between sync and streaming; previously v2 voices were silently not loaded
  when streaming)
- Noise-matched generation is fully deterministic (12× repeat runs are
  byte-identical) and never falls back to RNG
- Baseline harness phrase_02 mismatch: `run_baseline.sh` synthesized a
  non-reference sentence for phrase_02, producing a spurious 0.011 correlation
  in the 2026-03-19 baseline; fixed 2026-06-11, phrase_02 = 1.000000

### Numerical Parity
- Noise-matched waveform correlation = 1.000000 on all 4 canonical phrases for
  both v1 and v2 (per-frame minima >0.998, output length exactly matches the
  reference). Raw artifact: `docs/audit/correlation-v0.5.0-2026-06-11.txt`.

## [0.4.1] - 2026-01-27

### Removed
- **BREAKING**: Removed legacy token-chunked streaming API (`start_streaming`, `synthesize_streaming`)
  - This method chunked text into token batches with higher latency than true streaming
  - Use `start_true_streaming()` instead for optimal TTFA (~200ms)

### Fixed
- True streaming audio quality now matches sync mode (removed broken crossfade logic)
- Fixed callback return behavior to properly generate `frames_after_eos` padding
- Natural EOS detection now works correctly (changed `min_gen_steps` from 3 to 0)

### Changed
- `synthesize()` (sync mode) is now documented as "for debugging/batch processing"
- `synthesize_true_streaming()` is now the sole recommended streaming method

### Documentation
- Updated API documentation to clarify the two synthesis modes:
  - `synthesize()` - Sync mode for debugging and batch processing
  - `synthesize_true_streaming()` - Preferred method for on-device TTS (~200ms TTFA)

## [0.4.0] - 2026-01-24 (Beta)

### Added
- Initial beta release of Pocket TTS iOS XCFramework
- FlowLM transformer model (~70M params) for text-to-speech generation
- MLP consistency sampler (~10M params) for audio token sampling
- Mimi VAE decoder (~20M params) with streaming support
- SEANet upsampling convolutions for high-quality audio synthesis
- 8 built-in voices: Alba, Marius, Javert, Jean, Fantine, Cosette, Eponine, Azelma
- Swift bindings via UniFFI for seamless iOS integration
- High-level async/await Swift wrapper (`PocketTTSSwift`)
- iOS device (arm64) and simulator (arm64-sim) support
- Streaming synthesis with overlap-add for low-latency playback
- Configurable temperature, speed, and voice parameters

### Technical
- CPU-only inference optimized for iOS (Candle ML framework)
- Memory-mapped safetensors for efficient model loading
- KV caching for efficient autoregressive generation
- Release build with LTO and symbol stripping for minimal binary size
