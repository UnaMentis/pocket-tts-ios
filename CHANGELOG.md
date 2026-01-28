# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2025-01-27

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

## [0.4.0] - 2025-01-24 (Beta)

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
