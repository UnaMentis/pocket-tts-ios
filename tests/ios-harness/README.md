# PocketTTS iOS Demo

A comprehensive iOS test harness for validating the Pocket TTS XCFramework. This app provides a full-featured interface for testing TTS synthesis, measuring performance, and extracting audio for analysis.

![PocketTTS Demo Screenshot](screenshot.png)

## Demo App Overview

The screenshot above shows the PocketTTS Demo app after completing a synthesis:

- **System Resources** panel displays real-time memory usage, CPU load, and thermal state
- **Text Input** field where you enter text to synthesize
- **Voice Selector** with 8 built-in voices (segmented control)
- **Status Bar** shows model load time and synthesis results
- **Performance Metrics** display synthesis time, audio duration, and real-time factor (RTF)
- **Synthesize Button** triggers TTS generation
- **Play/Stop Button** controls audio playback (turns red when playing)
- **Waveform View** (appears after synthesis) shows visual preview of the generated audio

## Features

- **Text-to-Speech Synthesis**: Enter any text and synthesize speech
- **Voice Selection**: Choose from 8 built-in voices (Alba, Marius, Javert, Jean, Fantine, Cosette, Eponine, Azelma)
- **Real-time Resource Monitoring**:
  - Memory usage (system and app)
  - CPU utilization
  - Thermal state indicator
- **Performance Metrics**: Synthesis time, audio duration, and real-time factor (RTF)
- **Waveform Visualization**: Visual preview of synthesized audio
- **Audio Export**: Automatic saving of WAV files for external validation

## Quick Start

### Prerequisites

- macOS with Xcode 15+
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`)
- iOS Simulator or physical device (iOS 17.0+)

### Build and Run

```bash
# Navigate to the harness directory
cd tests/ios-harness/PocketTTSDemo

# Generate Xcode project
xcodegen generate

# Build and run (simulator)
xcodebuild -scheme PocketTTSDemo \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  build

# Install on booted simulator
xcrun simctl install booted \
  ~/Library/Developer/Xcode/DerivedData/PocketTTSDemo-*/Build/Products/Debug-iphonesimulator/PocketTTSDemo.app

# Launch
xcrun simctl launch booted com.unamentis.PocketTTSDemo
```

### Automated Setup (from release)

```bash
# From the ios-harness directory
chmod +x setup.sh
./setup.sh

# Or with a specific release zip
./setup.sh /path/to/PocketTTS-v0.4.0.zip
```

## Project Structure

```
ios-harness/
├── README.md                    # This file
├── screenshot.png               # App screenshot for documentation
├── setup.sh                     # Automated setup script
└── PocketTTSDemo/
    ├── project.yml              # XcodeGen project specification
    ├── Frameworks/
    │   └── PocketTTS.xcframework    # TTS framework (from build)
    └── PocketTTSDemo/
        ├── PocketTTSDemoApp.swift   # App entry point
        ├── ContentView.swift        # Main UI (SwiftUI)
        ├── TTSViewModel.swift       # TTS logic, state, audio handling
        ├── ResourceMonitor.swift    # System metrics collection
        ├── pocket_tts_ios.swift     # UniFFI-generated Swift bindings
        ├── PocketTTSSwift.swift     # High-level async Swift API
        ├── Info.plist               # App configuration
        └── Models/                  # Model files (folder reference)
            ├── model.safetensors        # Main FlowLM model (~225MB)
            ├── tokenizer.model          # SentencePiece tokenizer
            └── voices/
                └── alba.safetensors     # Voice embedding
```

## Architecture

### Component Overview

| Component | File | Description |
|-----------|------|-------------|
| **ContentView** | `ContentView.swift` | SwiftUI interface with text input, voice picker, controls, and metrics display |
| **TTSViewModel** | `TTSViewModel.swift` | `@MainActor` class managing TTS engine lifecycle, synthesis, and audio playback |
| **ResourceMonitor** | `ResourceMonitor.swift` | Tracks memory, CPU, and thermal state via system APIs |
| **UniFFI Bindings** | `pocket_tts_ios.swift` | Auto-generated Swift bindings from Rust via UniFFI |

### Data Flow

```
User Input (text)
    ↓
TTSViewModel.synthesize()
    ↓
PocketTtsEngine.synthesize(text:)  ← UniFFI call to Rust
    ↓
SynthesisResult { audioData: Data, sampleRate, durationSeconds }
    ↓
├── audioData → AVAudioPlayer (playback)
├── audioData → Documents/ (WAV file for extraction)
└── extractSamplesFromWav() → WaveformView (visualization)
```

### Key Types (from UniFFI)

```swift
// TTS Engine
class PocketTtsEngine {
    init(modelPath: String) throws
    func configure(config: TtsConfig) throws
    func synthesize(text: String) throws -> SynthesisResult
}

// Configuration
struct TtsConfig {
    var voiceIndex: UInt32       // 0-7 for built-in voices
    var temperature: Float       // Sampling temperature (default: 0.7)
    var topP: Float              // Nucleus sampling (default: 0.9)
    var speed: Float             // Playback speed (default: 1.0)
    var consistencySteps: UInt32 // MLP sampler steps (default: 2)
    var useFixedSeed: Bool       // Reproducible output
    var seed: UInt32             // Random seed when fixed
}

// Synthesis Output
struct SynthesisResult {
    var audioData: Data          // WAV file (32-bit float, 24kHz mono)
    var sampleRate: UInt32       // Always 24000
    var channels: UInt32         // Always 1
    var durationSeconds: Double  // Audio length
}
```

## Audio Validation

### Extracting Audio from Simulator

Synthesized audio is automatically saved to the app's Documents directory:

```bash
# Get the app container path
APP_DATA=$(xcrun simctl get_app_container booted com.unamentis.PocketTTSDemo data)

# List saved audio files
ls -la "$APP_DATA/Documents/"

# Copy to local directory for analysis
cp "$APP_DATA/Documents/tts_output_*.wav" ./validation/
```

### Audio File Format

The exported WAV files use:
- **Format**: WAVE_FORMAT_EXTENSIBLE (IEEE Float)
- **Sample Rate**: 24,000 Hz
- **Bit Depth**: 32-bit float
- **Channels**: Mono

### Validation with Whisper

```bash
# Transcribe iOS output
whisper ./validation/tts_output_ios.wav --model base --language en

# Compare with native Rust output
whisper ./validation/native_output.wav --model base --language en

# Calculate Word Error Rate (WER)
python3 -c "
from jiwer import wer
reference = 'Hello! This is a test of the Pocket TTS text to speech system.'
ios_hypothesis = '...'  # from whisper
native_hypothesis = '...'
print(f'iOS WER: {wer(reference, ios_hypothesis):.1%}')
print(f'Native WER: {wer(reference, native_hypothesis):.1%}')
"
```

### Sample Analysis

```python
import struct

with open('tts_output.wav', 'rb') as f:
    data = f.read()

# Find data chunk (after header)
offset = 60  # WAVE_FORMAT_EXTENSIBLE has 60-byte header
samples = struct.unpack('<' + 'f' * 100, data[offset:offset+400])
print(f"First 10 samples: {samples[:10]}")
print(f"Max amplitude: {max(abs(s) for s in samples)}")
```

## Build Configuration

### project.yml (XcodeGen)

```yaml
name: PocketTTSDemo
options:
  bundleIdPrefix: com.unamentis
  deploymentTarget:
    iOS: "17.0"
  xcodeVersion: "15.0"

targets:
  PocketTTSDemo:
    type: application
    platform: iOS
    sources:
      - PocketTTSDemo
      - path: PocketTTSDemo/Models
        type: folder              # Bundle as folder reference
    settings:
      PRODUCT_BUNDLE_IDENTIFIER: com.unamentis.PocketTTSDemo
      INFOPLIST_FILE: PocketTTSDemo/Info.plist
      SWIFT_VERSION: "5.9"
      ENABLE_USER_SCRIPT_SANDBOXING: NO
      OTHER_LDFLAGS: ["-lc++"]    # Required for C++ dependencies
    dependencies:
      - framework: Frameworks/PocketTTS.xcframework
        embed: true
```

### Critical Build Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `OTHER_LDFLAGS` | `["-lc++"]` | Links C++ standard library (required by sentencepiece, protobuf) |
| `ENABLE_USER_SCRIPT_SANDBOXING` | `NO` | Allows build scripts to access resources |
| Models `type` | `folder` | Preserves directory structure in bundle |

## Performance Benchmarks

Typical measurements on various devices:

| Device | Model Load | Synthesis RTF | Peak Memory |
|--------|------------|---------------|-------------|
| iPhone 17 Pro (Sim) | ~0.3s | 3.0-3.5x | ~850MB |
| iPhone 15 Pro | ~0.3s | 2.5-3.0x | ~300MB |
| iPhone 14 | ~0.5s | 2.0-2.5x | ~350MB |
| iPad Pro M2 | ~0.2s | 4.0-5.0x | ~280MB |

*RTF = Real-Time Factor (audio duration / synthesis time). Higher is better.*

## Troubleshooting

### "Model files not found"

```
Error: Model path does not exist
```

**Solution**: Ensure `Models/` folder is:
1. Added as a **folder reference** (blue folder icon in Xcode)
2. Listed in "Copy Bundle Resources" build phase
3. Contains `model.safetensors`, `tokenizer.model`, and `voices/`

### "Framework not found" / Linker errors

```
ld: framework not found PocketTTS
```

**Solution**:
1. Verify `PocketTTS.xcframework` is in `Frameworks/` directory
2. Check "Embed & Sign" is selected in target settings
3. Ensure `OTHER_LDFLAGS: ["-lc++"]` is set

### "Symbol not found: _swift_..." errors

**Solution**: The XCFramework must be built with the same Swift version as your Xcode. Rebuild with `./scripts/build-ios.sh`.

### Audio playback issues

**Symptoms**: Play button doesn't respond, or audio cuts off

**Solution**: The app properly retains the `AVAudioPlayerDelegate` to prevent premature deallocation. If issues persist, check:
1. Audio session is active: `.setCategory(.playback)`
2. Delegate is stored as a property (not local variable)

### Build hangs on "Compiling Swift source files"

**Solution**: First build may take 2-3 minutes due to Swift module compilation. Subsequent builds are faster.

## Voices

| Index | Name | Gender | Description |
|-------|------|--------|-------------|
| 0 | Alba | Female | Default voice, clear and natural |
| 1 | Marius | Male | Warm, conversational tone |
| 2 | Javert | Male | Deep, authoritative voice |
| 3 | Jean | Male | Gentle, measured delivery |
| 4 | Fantine | Female | Soft, expressive voice |
| 5 | Cosette | Female | Young, bright tone |
| 6 | Eponine | Female | Rich, dramatic voice |
| 7 | Azelma | Female | Light, cheerful delivery |

## Development

### Updating the XCFramework

After rebuilding the Rust library:

```bash
# From project root
./scripts/build-ios.sh

# Copy to harness
cp -r output/PocketTTS.xcframework tests/ios-harness/PocketTTSDemo/Frameworks/

# Regenerate and rebuild
cd tests/ios-harness/PocketTTSDemo
xcodegen generate
xcodebuild -scheme PocketTTSDemo build
```

### Adding New Test Cases

Edit `TTSViewModel.swift` to add test scenarios:

```swift
// Example: Test multiple voices
func testAllVoices() async {
    for voice in TTSVoice.allCases {
        selectedVoice = voice
        synthesize()
        // Wait for completion...
    }
}
```

## License

MIT License - See main project LICENSE file.
