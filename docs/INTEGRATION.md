# Pocket TTS iOS Integration Guide

This guide explains how to integrate Pocket TTS into your iOS application.

## Requirements

- iOS 17.0+
- Xcode 15+
- ~250MB for model files

## Installation

### From GitHub Releases

1. Download the latest `PocketTTS-vX.Y.Z.zip` from [Releases](https://github.com/UnaMentis/pocket-tts-ios/releases)
2. Extract the archive
3. Drag `PocketTTS.xcframework` into your Xcode project
4. When prompted, select "Copy items if needed" and add to your target
5. Add Swift files from `Sources/` directory to your project

### Framework Search Paths

The XCFramework should be automatically recognized. If you encounter issues:

1. Select your project in Xcode
2. Go to Build Settings → Framework Search Paths
3. Add the path to the folder containing `PocketTTS.xcframework`

## Model Files

Pocket TTS requires model files that are **not included** in the release due to size. You need to:

1. Download model files from the appropriate source
2. Add them to your app bundle

### Expected Structure

```
YourApp.app/
└── Models/
    ├── model.safetensors     # Main model (~225MB)
    ├── tokenizer.model       # Tokenizer (~60KB)
    └── voices/               # Voice embeddings (~4MB)
        ├── alba.safetensors
        ├── marius.safetensors
        ├── javert.safetensors
        ├── jean.safetensors
        ├── fantine.safetensors
        ├── cosette.safetensors
        ├── eponine.safetensors
        └── azelma.safetensors
```

### Adding to Xcode

1. Create a "Models" folder in your project
2. Drag the model files into Xcode
3. Ensure "Copy items if needed" is checked
4. Verify files appear in Build Phases → Copy Bundle Resources

## Quick Start

### Basic Usage

```swift
import Foundation

// Get path to model directory in bundle
guard let modelPath = Bundle.main.path(forResource: "Models", ofType: nil) else {
    fatalError("Models not found in bundle")
}

// Initialize engine
let engine = try PocketTTSEngine(modelPath: modelPath)

// Configure voice and settings
let config = TTSConfig(
    voiceIndex: 0,      // 0-7 for different voices
    temperature: 0.7,   // 0.0-1.0, higher = more variation
    speed: 1.0          // 0.5-2.0, speech rate
)
try engine.configure(config: config)

// Synthesize text
let result = try engine.synthesize(text: "Hello, world!")

// result.samples contains Float32 audio at 24kHz
// result.sampleRate is 24000
```

### Using the Swift Wrapper (Recommended)

The `PocketTTSSwift.swift` wrapper provides a modern async/await API:

```swift
import Foundation

// Create actor-based engine
let tts = PocketTTSSwift()

// Load model (async)
try await tts.load(modelPath: modelPath)

// Configure
try await tts.configure(.default)
// Or: .lowLatency, .highQuality, or custom TTSConfig

// Synthesize (async)
let result = try await tts.synthesize(text: "Hello, world!")

// Play audio...
```

### Available Voices

| Index | Name     | Description |
|-------|----------|-------------|
| 0     | Alba     | Female voice |
| 1     | Marius   | Male voice |
| 2     | Javert   | Male voice |
| 3     | Jean     | Male voice |
| 4     | Fantine  | Female voice |
| 5     | Cosette  | Female voice |
| 6     | Eponine  | Female voice |
| 7     | Azelma   | Female voice |

### Configuration Presets

```swift
// Default balanced settings
TTSConfig.default  // voice: 0, temp: 0.7, speed: 1.0

// Low latency for real-time
TTSConfig.lowLatency  // voice: 0, temp: 0.5, speed: 1.1

// High quality for offline
TTSConfig.highQuality  // voice: 0, temp: 0.8, speed: 0.95
```

## Playing Audio

### Using AVAudioEngine

```swift
import AVFoundation

let audioEngine = AVAudioEngine()
let playerNode = AVAudioPlayerNode()

// Setup audio session
try AVAudioSession.sharedInstance().setCategory(.playback)
try AVAudioSession.sharedInstance().setActive(true)

// Create audio format (24kHz mono Float32)
let format = AVAudioFormat(
    commonFormat: .pcmFormatFloat32,
    sampleRate: Double(result.sampleRate),
    channels: 1,
    interleaved: false
)!

// Create buffer
let buffer = AVAudioPCMBuffer(
    pcmFormat: format,
    frameCapacity: AVAudioFrameCount(result.samples.count)
)!
buffer.frameLength = buffer.frameCapacity

// Copy samples
result.samples.withUnsafeBufferPointer { ptr in
    buffer.floatChannelData![0].update(from: ptr.baseAddress!, count: ptr.count)
}

// Attach and connect
audioEngine.attach(playerNode)
audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)

// Play
try audioEngine.start()
playerNode.scheduleBuffer(buffer, completionHandler: nil)
playerNode.play()
```

## Error Handling

```swift
do {
    let result = try engine.synthesize(text: text)
} catch let error as TTSError {
    switch error {
    case .notLoaded:
        print("Engine not loaded")
    case .invalidConfig:
        print("Invalid configuration")
    case .synthesisError(let message):
        print("Synthesis failed: \(message)")
    default:
        print("TTS error: \(error)")
    }
} catch {
    print("Unexpected error: \(error)")
}
```

## Performance Tips

1. **Reuse the engine**: Creating `PocketTTSEngine` loads the model. Do this once at app startup.

2. **Pre-warm**: Call `synthesize` with a short phrase during loading to warm up the model.

3. **Background thread**: Synthesis is CPU-intensive. Use async/await or dispatch to background.

4. **Memory**: The model uses ~300MB RAM when loaded. Consider unloading if memory-constrained.

## Troubleshooting

### "Models not found"
- Verify model files are in Copy Bundle Resources
- Check the path matches your bundle structure

### Slow first synthesis
- First call loads caches. Subsequent calls are faster.
- Consider pre-warming during app launch

### Audio sounds robotic
- Try increasing temperature (0.8-0.9)
- Try different voices
- Ensure text has proper punctuation

### App crashes on launch
- Verify XCFramework is properly linked
- Check minimum iOS version is 17.0+
- Ensure all Swift files are added to target

## License

MIT License - See LICENSE file for details.
