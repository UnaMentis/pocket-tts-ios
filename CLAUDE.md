# CLAUDE.md - Pocket TTS iOS (Rust/Candle)

This directory contains the Rust/Candle implementation of Kyutai Pocket TTS for native iOS inference.

## Quick Commands

```bash
# Check compilation
cargo check

# Build debug
cargo build

# Build release
cargo build --release

# Run tests
cargo test

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt

# Build for iOS (creates XCFramework)
./scripts/build-ios.sh
```

## Architecture

The implementation follows the Pocket TTS architecture:

```
Text → Tokenizer → FlowLM (Transformer) → MLP Sampler → Mimi Decoder → Audio
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| FlowLM | `src/models/flowlm.rs` | 6-layer transformer (~70M params) |
| MLP Sampler | `src/modules/mlp.rs` | Consistency sampling (~10M params) |
| Mimi Decoder | `src/models/mimi.rs` | VAE decoder (~20M params) |
| SEANet | `src/models/seanet.rs` | Upsampling convolutions |
| Tokenizer | `src/tokenizer.rs` | SentencePiece wrapper |

### Modules

| Module | File | Description |
|--------|------|-------------|
| Attention | `src/modules/attention.rs` | Multi-head attention with KV cache |
| RoPE | `src/modules/rotary.rs` | Rotary position embeddings |
| MLP | `src/modules/mlp.rs` | Feed-forward and gated MLP |
| Conv | `src/modules/conv.rs` | Causal convolutions, SEANet blocks |
| Embeddings | `src/modules/embeddings.rs` | Text and voice embeddings |
| LayerNorm | `src/modules/layer_norm.rs` | RMS and standard layer norm |

## UniFFI Interface

The Swift interface is defined in `src/pocket_tts.udl`:

- `PocketTTSEngine` - Main engine class
- `TTSConfig` - Configuration (voice, temperature, speed, etc.)
- `SynthesisResult` - Audio output
- `AudioChunk` - Streaming chunk
- `TTSEventHandler` - Streaming callback

## iOS Build Process

1. Build for device: `cargo build --release --target aarch64-apple-ios`
2. Build for simulator: `cargo build --release --target aarch64-apple-ios-sim`
3. Generate Swift bindings: `cargo run --bin uniffi-bindgen generate ...`
4. Create XCFramework: `xcodebuild -create-xcframework ...`

The `scripts/build-ios.sh` script automates this process.

## Model Files

The Rust implementation loads model files from:

```
kyutai-pocket-ios/
├── model.safetensors     # Main model weights
├── tokenizer.model       # SentencePiece tokenizer
└── voices/               # Voice embeddings
    ├── alba.safetensors
    └── ...
```

## Performance Notes

- CPU-only on iOS (Candle doesn't support Metal on iOS)
- Pocket TTS is optimized for CPU, targeting ~3-4x realtime on iPhone
- Memory-mapped safetensors for efficient loading
- KV caching for efficient streaming

## Dependencies

Key crates:
- `candle-core`, `candle-nn` - ML framework
- `uniffi` - Swift FFI bindings
- `safetensors` - Model loading
- `tokenizers` - SentencePiece
- `rubato` - Audio resampling
- `hound` - WAV encoding

## Integration with iOS App

After building the XCFramework:

1. Add `PocketTTS.xcframework` to Xcode project
2. Add `pocket_tts_ios.swift` (generated bindings)
3. Optionally add `swift/PocketTTSSwift.swift` for async/await API
4. Bundle model files in app resources

See `rust/pocket-tts-ios/README.md` for detailed integration instructions.
