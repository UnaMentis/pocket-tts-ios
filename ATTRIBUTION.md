# Attribution

This project includes code derived from [babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts), licensed under MIT.

## Core Inference Engine

The following components are based on babybirdprd/pocket-tts:
- `src/models/flowlm.rs` - FlowLM transformer implementation
- `src/models/mimi.rs` - Mimi VAE decoder
- `src/models/seanet.rs` - SEANet upsampling decoder
- `src/models/pocket_tts.rs` - Model orchestration
- `src/modules/attention.rs` - Multi-head attention with KV cache
- `src/modules/conv.rs` - Convolution operations
- `src/modules/embeddings.rs` - Text and voice embeddings
- `src/modules/layer_norm.rs` - RMSNorm implementation
- `src/modules/mlp.rs` - Feed-forward networks
- `src/modules/rotary.rs` - Rotary position embeddings
- `src/tokenizer.rs` - Tokenization (adapted to JSON format)
- `src/audio.rs` - Audio processing utilities

## Original Contributions for iOS

The following are original contributions by UnaMentis:
- `src/pocket_tts.udl` - UniFFI interface definition
- `src/engine.rs` - UniFFI-exposed API with Swift async wrappers
- `src/config.rs` - iOS-specific configuration
- `scripts/build-ios.sh` - XCFramework build system
- `swift/PocketTTSSwift.swift` - Swift async/await integration
- All tests in `tests/` - iOS integration test suite
- iOS service layer (KyutaiPocketTTSService, KyutaiPocketModelManager)

## Attribution Chain

### Original Research & Model
- **Kyutai Labs** - Pocket TTS model architecture and trained weights
- License: CC-BY-4.0 (model weights), MIT (reference code)
- Release: January 13, 2026

### Rust/Candle Port
- **babybirdprd/pocket-tts** - Complete Rust/Candle implementation
- GitHub: https://github.com/babybirdprd/pocket-tts
- License: MIT
- Created: January 14, 2026

### iOS Integration
- **UnaMentis** - UniFFI bindings, XCFramework build system, iOS service layer
- License: MIT
- Created: January 21, 2026

## License Compliance

All components are MIT-licensed and compatible:
- Kyutai model code: MIT
- babybirdprd/pocket-tts: MIT
- Candle ML framework: Apache 2.0 / MIT dual license
- UnaMentis iOS integration: MIT

## Acknowledgments

Special thanks to:
- **Kyutai Labs** for creating and open-sourcing the Pocket TTS model
- **babybirdprd** for the excellent Rust/Candle port that made iOS integration possible
- **HuggingFace Candle team** for the ML framework
- **Mozilla UniFFI** for the FFI bindings system
