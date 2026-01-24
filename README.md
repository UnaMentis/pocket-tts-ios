# Pocket TTS iOS

Native iOS implementation of Kyutai Pocket TTS using Rust/Candle.

## Overview

This crate provides on-device text-to-speech for iOS using the Kyutai Pocket TTS model. It uses the Candle ML framework for inference and UniFFI for Swift bindings.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Swift/SwiftUI App                   │
├─────────────────────────────────────────────────┤
│         Generated Swift Bindings (UniFFI)        │
├─────────────────────────────────────────────────┤
│               PocketTTSEngine                    │
├─────────────────────────────────────────────────┤
│  FlowLM    │   MLPSampler   │   MimiDecoder    │
│ (70M)      │    (10M)       │     (20M)        │
└─────────────────────────────────────────────────┘
```

## Building

### Prerequisites

1. Rust toolchain with iOS targets:
   ```bash
   rustup target add aarch64-apple-ios
   rustup target add aarch64-apple-ios-sim
   ```

2. Xcode with iOS SDK

### Build XCFramework

```bash
./scripts/build-ios.sh
```

This creates:
- `target/xcframework/PocketTTS.xcframework` - Static library
- `target/xcframework/pocket_tts_ios.swift` - Swift bindings

### Integration with Xcode

1. Drag `PocketTTS.xcframework` into your Xcode project
2. Add `pocket_tts_ios.swift` to your Swift sources
3. Import and use:

```swift
import Foundation

// Initialize engine with model path
let modelPath = Bundle.main.path(forResource: "kyutai-pocket-ios", ofType: nil)!
let engine = try PocketTTSEngine(modelPath: modelPath)

// Configure
let config = TTSConfig(
    voiceIndex: 0,  // Alba
    temperature: 0.7,
    topP: 0.9,
    speed: 1.0,
    consistencySteps: 2,
    useFixedSeed: false,
    seed: 42
)
try engine.configure(config: config)

// Synthesize
let result = try engine.synthesize(text: "Hello, world!")
// result.audioData contains WAV bytes
```

## Model Files

The model files should be placed in:
```
kyutai-pocket-ios/
├── model.safetensors     # Main model weights (225MB)
├── tokenizer.model       # SentencePiece tokenizer (60KB)
└── voices/               # Voice embeddings (4.2MB)
    ├── alba.safetensors
    ├── marius.safetensors
    ├── javert.safetensors
    ├── jean.safetensors
    ├── fantine.safetensors
    ├── cosette.safetensors
    ├── eponine.safetensors
    └── azelma.safetensors
```

## Features

- **8 Built-in Voices**: Alba, Marius, Javert, Jean, Fantine, Cosette, Eponine, Azelma
- **Streaming Synthesis**: Low-latency audio generation with overlap-add
- **Configurable**: Temperature, top-p, speed, consistency steps
- **CPU Optimized**: Designed for efficient CPU inference

## Performance

- Time to first audio: ~200ms
- Real-time factor: ~3-4x on iPhone 15 Pro (estimated)
- Memory usage: ~150MB during inference

## Credits

This implementation builds upon excellent work from:

- **[Kyutai Labs](https://kyutai.org/)** - Original Pocket TTS model architecture and trained weights
- **[babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts)** - Complete Rust/Candle port that made iOS integration possible
- **[HuggingFace Candle](https://github.com/huggingface/candle)** - ML framework for efficient inference

See [ATTRIBUTION.md](ATTRIBUTION.md) for detailed attribution information.

## License

MIT (code), CC-BY-4.0 (model weights)
