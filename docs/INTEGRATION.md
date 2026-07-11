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

Pocket TTS requires a model directory (weights + tokenizer + voices) bundled
in your app. As of v0.5.0 you can bundle either model generation — v1 or v2.

### Choosing a Model Version

| | v1 (`english_2026-01`) | v2 (`english_2026-04`) |
|---|---|---|
| Availability | Public Hugging Face repo (`kyutai/pocket-tts-without-voice-cloning`); also bundled in this project's release zip | **Gated** Hugging Face repo (`kyutai/pocket-tts`); user-downloaded only |
| Voice file format | Embedding sequence (`audio_prompt`, `[1, seq, 1024]`) | Precomputed transformer KV-state (`bos_before_voice` + speaker projection baked in) |
| TTFA (host, 2026-06-11) | 252ms avg | 137ms avg — voices skip the 125-position voice prompt |
| Local directory convention | `kyutai-pocket-ios/` | `kyutai-pocket-ios-en2026-04/` |

**Both versions work with the same XCFramework API** — no code changes needed.
The engine auto-detects which voice-file format it is given and errors clearly
on anything else. The two voice formats are incompatible with each other, so
bundle voices from the same generation as the model weights.

### Getting the Weights

Use `scripts/download-model.py` from the repository for either version:

```bash
# v1 — public repo, no login needed
python scripts/download-model.py

# v2 — gated repo, requires Hugging Face access (see below)
python scripts/download-model.py --model v2
```

v1 weights are also included in the release zip (`Models/` folder).

**For v2**, the weights are gated on Hugging Face and are **never
redistributed in this project's release artifacts** (respecting Kyutai's
gate). Each user must:

1. Visit https://huggingface.co/kyutai/pocket-tts and accept the gate
   (instant approval)
2. Authenticate locally:
   ```bash
   huggingface-cli login
   # or, for a single run:
   HF_TOKEN=hf_... python scripts/download-model.py --model v2
   ```
3. Run `python scripts/download-model.py --model v2`

### Expected Structure

Both model generations use the same layout — bundle the contents of whichever
model directory you chose (v1 `kyutai-pocket-ios/` or v2
`kyutai-pocket-ios-en2026-04/`) as your app's `Models/` folder:

```
YourApp.app/
└── Models/
    ├── model.safetensors     # Main model (~225MB)
    ├── tokenizer.model       # Tokenizer (~60KB)
    └── voices/               # Voice files (v1 embeddings or v2 KV-states)
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
import AVFoundation

// Get path to model directory in bundle (folder must be named "Models")
guard let modelPath = Bundle.main.path(forResource: "Models", ofType: nil) else {
    fatalError("Models not found in bundle")
}

// Initialize engine
let engine = try PocketTtsEngine(modelPath: modelPath)

// Configure voice and settings.
// TtsConfig has NO default field values — every field is required.
// Build the voice list from engine.loadedVoices() — the voices actually
// present in your model directory — never from a hardcoded list.
//
// ⚠️ topP and speed are INERT in v0.5.0: they are accepted and
// range-validated but not yet applied by the synthesis pipeline.
// Pass the defaults (0.9 / 1.0) and do not rely on them.
let config = TtsConfig(
    voiceIndex: 0,          // index into engine.loadedVoices()
    temperature: 0.7,       // higher = more variation
    topP: 0.9,              // INERT in v0.5.0 — reserved
    speed: 1.0,             // INERT in v0.5.0 — reserved
    consistencySteps: 2,    // sampler steps (1 = lowest latency)
    useFixedSeed: false,
    seed: 42
)
try engine.configure(config: config)

// Synthesize text.
// SynthesisResult.audioData is a COMPLETE WAV file (Int16 PCM, 24 kHz, mono).
// result.sampleRate == 24000, result.channels == 1, result.durationSeconds is the length.
let result = try engine.synthesize(text: "Hello, world!")

// Because it is a self-contained WAV, AVAudioPlayer plays it directly.
let player = try AVAudioPlayer(data: result.audioData)
player.play()
```

### Using the Swift Wrapper (Recommended)

The `PocketTTSSwift.swift` wrapper provides a modern async/await API:

```swift
import Foundation
import AVFoundation

// Create actor-based engine with the model path, then load.
let tts = PocketTTSSwift(modelPath: modelPath)

// Load model (async)
try await tts.load()

// Configure with a preset (or a custom PocketTTSSwift.Config)
try await tts.configure(.default)
// Or: .lowLatency, .highQuality, or PocketTTSSwift.Config(...)

// Synthesize (async). result.audioData is a complete WAV, same as the raw engine.
let result = try await tts.synthesize(text: "Hello, world!")

let player = try AVAudioPlayer(data: result.audioData)
player.play()
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

The raw `TtsConfig` binding has no presets — you always supply all seven fields.
Presets live on the `PocketTTSSwift.Config` wrapper type:

```swift
// Default balanced settings
PocketTTSSwift.Config.default      // temp 0.7, topP 0.9, speed 1.0, consistencySteps 2

// Low latency for real-time
PocketTTSSwift.Config.lowLatency   // as default but consistencySteps 1

// High quality for offline
PocketTTSSwift.Config.highQuality  // temp 0.5, consistencySteps 4
```

## Playing Audio

### Sync path — `AVAudioPlayer` (the WAV case)

`SynthesisResult.audioData` from `synthesize` / `synthesizeWithVoice` /
`synthesizeNoiseMatched` / `decodeLatents` is a **complete WAV file** (RIFF
header + Int16 PCM, 24 kHz, mono). It carries its own header, so the simplest
correct playback is `AVAudioPlayer(data:)`:

```swift
import AVFoundation

try AVAudioSession.sharedInstance().setCategory(.playback)
try AVAudioSession.sharedInstance().setActive(true)

let result = try engine.synthesize(text: "Hello, world!")

// audioData is a self-contained WAV — no manual buffer construction needed.
let player = try AVAudioPlayer(data: result.audioData)
player.play()
```

### Streaming path — raw PCM chunks (NOT WAV)

`startTrueStreaming(text:handler:)` delivers `AudioChunk`s to a `TtsEventHandler`.
**Streaming chunks are raw PCM, not WAV:** each `chunk.audioData` is a run of
**Float32 samples, little-endian, mono, 24 kHz** (`chunk.sampleRate == 24000`),
4 bytes per sample, with **no WAV header**. Do not hand these to
`AVAudioPlayer(data:)` — schedule them into `AVAudioEngine` yourself:

```swift
import AVFoundation

let audioEngine = AVAudioEngine()
let playerNode = AVAudioPlayerNode()

let format = AVAudioFormat(
    commonFormat: .pcmFormatFloat32,
    sampleRate: 24000,   // matches chunk.sampleRate
    channels: 1,
    interleaved: false
)!

audioEngine.attach(playerNode)
audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
try audioEngine.start()
playerNode.play()

final class PlaybackHandler: TtsEventHandler {
    let node: AVAudioPlayerNode
    let format: AVAudioFormat
    init(node: AVAudioPlayerNode, format: AVAudioFormat) {
        self.node = node
        self.format = format
    }

    func onAudioChunk(chunk: AudioChunk) {
        // Reinterpret the raw little-endian Float32 bytes as [Float].
        let samples: [Float] = chunk.audioData.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: format,
                  frameCapacity: AVAudioFrameCount(samples.count)
              ) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { ptr in
            buffer.floatChannelData![0].update(from: ptr.baseAddress!, count: ptr.count)
        }
        node.scheduleBuffer(buffer, completionHandler: nil)
    }

    func onProgress(progress: Float) {}
    func onComplete() {}
    func onError(message: String) {
        // ALWAYS log the message — see Error Handling below.
        print("Streaming error: \(message)")
    }
}

let handler = PlaybackHandler(node: playerNode, format: format)
try engine.startTrueStreaming(text: "Hello, world!", handler: handler)
```

> The `PocketTTSSwift` actor wraps this in an `AsyncThrowingStream` via
> `synthesizeStreaming(text:)`, yielding chunks with the same raw-PCM payload.

## Error Handling

All throwing engine methods throw `PocketTtsError`. Every case carries a
`message: String` payload. **Always log that message.** A past on-device field
failure was undiagnosable for weeks because the host app never logged the
message string — the single most useful data point was the exact error text
(see [../APP-SIDE-FINDINGS.md](../APP-SIDE-FINDINGS.md)).

```swift
do {
    let result = try engine.synthesize(text: text)
} catch let error as PocketTtsError {
    switch error {
    case .ModelNotLoaded(let message):
        print("Model not loaded: \(message)")
    case .ModelLoadFailed(let message):
        print("Model load failed: \(message)")
    case .TokenizationFailed(let message):
        print("Tokenization failed: \(message)")
    case .InferenceFailed(let message):
        print("Inference failed: \(message)")
    case .InvalidVoice(let message):
        print("Invalid voice: \(message)")
    case .InvalidConfig(let message):
        print("Invalid config: \(message)")
    case .AudioEncodingFailed(let message):
        print("Audio encoding failed: \(message)")
    case .IoError(let message):
        print("I/O error: \(message)")
    }
} catch {
    print("Unexpected error: \(error)")
}
```

### Error case reference

| Case | Likely cause | What to do |
|------|--------------|------------|
| `ModelNotLoaded` | Called `synthesize`/`configure` before the engine finished loading (or after `unload()`) | Ensure the `PocketTtsEngine(modelPath:)` init succeeded before use |
| `ModelLoadFailed` | Missing/corrupt `model.safetensors`, wrong `modelPath`, or out-of-memory during load (device-only) | Verify the bundle path and files; check memory (see Deployment notes) |
| `TokenizationFailed` | Missing/corrupt `tokenizer.model`, or unencodable input text | Verify `tokenizer.model` is bundled; sanitize input |
| `InferenceFailed` | Runtime failure during transformer/decoder inference | Log the message; capture the input text; file an issue with the string |
| `InvalidVoice` | `voiceIndex` out of range, or voice file format mismatched to the model generation (v1 vs v2) | Use index 0-7; bundle voices from the same generation as the weights |
| `InvalidConfig` | A `TtsConfig` field is out of range | Check `temperature`, `topP`, `speed`, `consistencySteps` |
| `AudioEncodingFailed` | WAV/PCM encoding failed | Log the message; likely an internal bug — report it |
| `IoError` | Filesystem error reading model/voice files | Confirm files exist, are readable, and are not in purgeable storage |

> The `PocketTTSSwift` wrapper surfaces these as
> `PocketTTSSwiftError.synthesisError(message)` (streaming) or rethrows the
> original error — the message is preserved either way, so log it.

## Deployment notes

Three hard-won facts that only surface on physical devices:

1. **Storage location.** Model files must live in the **app bundle** or the app's
   **Documents** directory — **never** `Caches`, `tmp`, or any purgeable /
   iCloud-offloadable location. The engine memory-maps `model.safetensors`; if
   iOS purges or offloads the file while it is mapped, the next access faults
   with **SIGBUS**. This fails **only on device** (the simulator never purges),
   so it will pass every simulator test and crash in the field.

2. **Memory.** Weights are **BF16 on disk (~220-235 MB)** but are loaded as
   **F32**, so the engine reaches **~470 MB resident** after load, with a higher
   transient spike **during** load. Consequences:
   - **Do not use in app extensions.** Keyboard, Share, and Siri extensions have
     ~40-120 MB memory limits — the model cannot load there and will fail
     device-only.
   - On memory-constrained devices, expect `ModelLoadFailed` (device-only). Call
     `unload()` when you no longer need synthesis.

3. **Case sensitivity.** Device filesystems (APFS) are **case-sensitive**; the
   simulator is usually **case-insensitive**. Bundle resource casing must match
   **exactly**: `model.safetensors`, `tokenizer.model`, `voices/`, and lowercase
   voice filenames (`alba.safetensors`, ...). A casing mismatch loads fine in the
   simulator and throws `ModelLoadFailed` / `IoError` on device.

## Reference Integration & Experimentation

The included demo app at `tests/ios-harness/PocketTTSDemo` is the canonical
reference integration and a hands-on experimentation harness — it wires up the
XCFramework, bundles a `Models/` folder, exercises sync and streaming paths, and
includes a Compare tab for the noise-matched correlation gate. Use it as a
working example and to reproduce issues. See
[../tests/ios-harness/README.md](../tests/ios-harness/README.md) for setup.

## Performance Tips

1. **Reuse the engine**: Creating `PocketTtsEngine` loads the model. Do this once at app startup.

2. **Pre-warm**: Call `synthesize` with a short phrase during loading to warm up the model.

3. **Never call synthesis on the main actor**: `synthesize` and
   `startTrueStreaming` are *blocking* calls that run for the full synthesis
   duration (streaming callbacks fire inline from the blocked call). Inside a
   `@MainActor` type, a plain `Task { }` **inherits the main actor** — the
   call will freeze your entire UI and queue every tap until it finishes
   (this bug shipped in our own demo). Use `Task.detached` or an explicitly
   non-main executor:

   ```swift
   let result = try await Task.detached(priority: .userInitiated) {
       try engine.synthesize(text: text)
   }.value
   ```

4. **Memory**: The model reaches ~470 MB resident once loaded (see Deployment notes). Call `unload()` if memory-constrained.

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
