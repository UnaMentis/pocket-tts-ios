import Foundation
import AVFoundation
import Combine
import SwiftUI

// Types from pocket_tts_ios.swift (UniFFI bindings) are compiled directly into the project

@MainActor
class TTSViewModel: ObservableObject {
    @Published var inputText: String = "Hello! This is a test of the Pocket TTS text to speech system."
    @Published var selectedVoice: TTSVoice = .alba
    @Published var status: TTSStatus?
    @Published var lastTiming: SynthesisTiming?
    @Published var audioData: Data?
    @Published var audioSamples: [Float]?
    @Published var isSynthesizing = false
    @Published var isPlaying = false
    @Published var isLoading = false
    @Published var isLoaded = false  // Model loaded successfully
    @Published var savedAudioPath: String?
    @Published var useStreaming = true  // Toggle for streaming vs sync mode

    let resourceMonitor = ResourceMonitor()

    // Expose engine for reference testing
    @Published var engine: PocketTtsEngine?
    private var audioPlayer: AVAudioPlayer?
    private var audioPlayerDelegate: AudioPlayerDelegate?  // Must retain - AVAudioPlayer.delegate is weak
    private var streamingHandler: StreamingEventHandler?  // Must retain for streaming callbacks

    init() {
        resourceMonitor.startMonitoring()
    }

    func loadModel() {
        guard !isLoading else { return }

        isLoading = true
        status = TTSStatus(message: "Loading model...", isLoading: true, type: .info)

        Task {
            do {
                let startTime = CFAbsoluteTimeGetCurrent()

                // Find model path in bundle
                guard let modelPath = Bundle.main.path(forResource: "Models", ofType: nil) else {
                    throw TTSError.modelNotFound
                }

                print("[PocketTTS] Loading model from: \(modelPath)")

                // Initialize the engine
                engine = try PocketTtsEngine(modelPath: modelPath)

                // Configure with default voice
                let config = TtsConfig(
                    voiceIndex: UInt32(selectedVoice.rawValue),
                    temperature: 0.7,
                    topP: 0.9,
                    speed: 1.0,
                    consistencySteps: 2,
                    useFixedSeed: true,
                    seed: 42
                )
                try engine?.configure(config: config)

                let loadTime = CFAbsoluteTimeGetCurrent() - startTime

                status = TTSStatus(
                    message: String(format: "Model loaded in %.2fs", loadTime),
                    isLoading: false,
                    type: .success
                )
                print("[PocketTTS] Model loaded successfully in \(loadTime)s")
                isLoaded = true
            } catch {
                status = TTSStatus(
                    message: "Failed to load model: \(error.localizedDescription)",
                    isLoading: false,
                    type: .error
                )
                print("[PocketTTS] Failed to load model: \(error)")
            }

            isLoading = false
        }
    }

    func synthesize() {
        if useStreaming {
            synthesizeStreaming()
        } else {
            synthesizeSync()
        }
    }

    /// Synchronous synthesis mode
    private func synthesizeSync() {
        guard !isSynthesizing else { return }
        guard !inputText.isEmpty else { return }

        isSynthesizing = true
        status = TTSStatus(message: "Synthesizing (sync)...", isLoading: true, type: .info)
        resourceMonitor.markSynthesisStart()

        Task {
            do {
                let startTime = CFAbsoluteTimeGetCurrent()

                guard let ttsEngine = engine else {
                    throw TTSError.engineNotInitialized
                }

                // Update voice if changed
                let config = TtsConfig(
                    voiceIndex: UInt32(selectedVoice.rawValue),
                    temperature: 0.7,
                    topP: 0.9,
                    speed: 1.0,
                    consistencySteps: 2,
                    useFixedSeed: true,
                    seed: 42
                )
                try ttsEngine.configure(config: config)

                print("[PocketTTS] Synthesizing (sync): \(inputText)")

                // Synthesize
                let result = try ttsEngine.synthesize(text: inputText)

                let synthesisTime = CFAbsoluteTimeGetCurrent() - startTime
                let audioDuration = result.durationSeconds

                // The result already contains WAV-formatted audio data from Rust
                // Use it directly - no need to re-encode
                audioData = result.audioData

                // Extract samples from WAV for waveform display
                audioSamples = extractSamplesFromWav(result.audioData)

                lastTiming = SynthesisTiming(
                    synthesisTime: synthesisTime,
                    audioDuration: audioDuration,
                    realtimeFactor: audioDuration / synthesisTime
                )

                resourceMonitor.markSynthesisEnd()

                // Auto-save audio for validation
                if let wavData = audioData {
                    saveAudioForValidation(wavData)
                }

                status = TTSStatus(
                    message: String(format: "Sync complete! %.2fx realtime", audioDuration / synthesisTime),
                    isLoading: false,
                    type: .success
                )
                print("[PocketTTS] Sync synthesis complete: \(audioSamples?.count ?? 0) samples, \(audioDuration)s audio in \(synthesisTime)s")
            } catch {
                status = TTSStatus(
                    message: "Synthesis failed: \(error.localizedDescription)",
                    isLoading: false,
                    type: .error
                )
                print("[PocketTTS] Synthesis failed: \(error)")
            }

            isSynthesizing = false
        }
    }

    /// Streaming synthesis mode with TTFA measurement
    private func synthesizeStreaming() {
        guard !isSynthesizing else { return }
        guard !inputText.isEmpty else { return }

        isSynthesizing = true
        status = TTSStatus(message: "Synthesizing (streaming)...", isLoading: true, type: .info)
        resourceMonitor.markSynthesisStart()

        Task {
            do {
                guard let ttsEngine = engine else {
                    throw TTSError.engineNotInitialized
                }

                // Update voice if changed
                let config = TtsConfig(
                    voiceIndex: UInt32(selectedVoice.rawValue),
                    temperature: 0.7,
                    topP: 0.9,
                    speed: 1.0,
                    consistencySteps: 2,
                    useFixedSeed: true,
                    seed: 42
                )
                try ttsEngine.configure(config: config)

                print("[PocketTTS] Synthesizing (streaming): \(inputText)")

                let startTime = CFAbsoluteTimeGetCurrent()

                // Create streaming handler with TTFA measurement
                let handler = StreamingEventHandler(
                    startTime: startTime,
                    sampleRate: 24000,
                    onComplete: { [weak self] result in
                        Task { @MainActor in
                            guard let self = self else { return }

                            self.resourceMonitor.markSynthesisEnd()

                            // Build WAV from collected samples
                            let wavData = self.buildWav(samples: result.samples, sampleRate: 24000)
                            self.audioData = wavData
                            self.audioSamples = result.samples

                            self.lastTiming = SynthesisTiming(
                                synthesisTime: result.totalTime,
                                audioDuration: result.audioDuration,
                                realtimeFactor: result.audioDuration / result.totalTime,
                                ttfaMs: result.ttfaMs,
                                chunkCount: result.chunkCount
                            )

                            // Auto-save audio for validation
                            if let wavData = self.audioData {
                                self.saveAudioForValidation(wavData)
                            }

                            let ttfaString = String(format: "%.0fms", result.ttfaMs)
                            self.status = TTSStatus(
                                message: String(format: "Streaming complete! TTFA: %@, %.2fx RTF", ttfaString, result.audioDuration / result.totalTime),
                                isLoading: false,
                                type: .success
                            )
                            print("[PocketTTS] Streaming complete: TTFA=\(ttfaString), \(result.samples.count) samples, \(result.chunkCount) chunks")

                            self.isSynthesizing = false
                        }
                    },
                    onError: { [weak self] error in
                        Task { @MainActor in
                            guard let self = self else { return }
                            self.status = TTSStatus(
                                message: "Streaming failed: \(error)",
                                isLoading: false,
                                type: .error
                            )
                            self.isSynthesizing = false
                        }
                    }
                )

                // Retain handler reference
                self.streamingHandler = handler

                // Start true streaming synthesis (optimized TTFA)
                try ttsEngine.startTrueStreaming(text: inputText, handler: handler)

            } catch {
                status = TTSStatus(
                    message: "Streaming failed: \(error.localizedDescription)",
                    isLoading: false,
                    type: .error
                )
                print("[PocketTTS] Streaming synthesis failed: \(error)")
                isSynthesizing = false
            }
        }
    }

    /// Build a WAV file from raw float samples
    private func buildWav(samples: [Float], sampleRate: UInt32) -> Data {
        var data = Data()

        // WAV header for 32-bit float
        let numSamples = UInt32(samples.count)
        let dataSize = numSamples * 4  // 4 bytes per float
        let fileSize = 36 + dataSize

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })  // chunk size
        data.append(contentsOf: withUnsafeBytes(of: UInt16(3).littleEndian) { Array($0) })   // format: IEEE float
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })   // channels
        data.append(contentsOf: withUnsafeBytes(of: sampleRate.littleEndian) { Array($0) })  // sample rate
        let byteRate = sampleRate * 4  // bytes per second
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(4).littleEndian) { Array($0) })   // block align
        data.append(contentsOf: withUnsafeBytes(of: UInt16(32).littleEndian) { Array($0) })  // bits per sample

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        // Sample data
        for sample in samples {
            data.append(contentsOf: withUnsafeBytes(of: sample) { Array($0) })
        }

        return data
    }

    func playAudio() {
        guard let data = audioData else {
            print("[PocketTTS] Play: No audio data available")
            return
        }
        guard !isPlaying else {
            print("[PocketTTS] Play: Already playing")
            return
        }

        do {
            // Configure audio session
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            audioPlayer = try AVAudioPlayer(data: data)

            // Create and retain the delegate (AVAudioPlayer.delegate is weak!)
            audioPlayerDelegate = AudioPlayerDelegate { [weak self] in
                Task { @MainActor in
                    self?.isPlaying = false
                    print("[PocketTTS] Playback finished")
                }
            }
            audioPlayer?.delegate = audioPlayerDelegate
            audioPlayer?.play()
            isPlaying = true
            print("[PocketTTS] Playback started, duration: \(audioPlayer?.duration ?? 0)s")
        } catch {
            print("[PocketTTS] Playback failed: \(error)")
            status = TTSStatus(
                message: "Playback failed: \(error.localizedDescription)",
                isLoading: false,
                type: .error
            )
        }
    }

    func stopAudio() {
        audioPlayer?.stop()
        isPlaying = false
    }

    /// Save audio to Documents directory for extraction from simulator
    private func saveAudioForValidation(_ wavData: Data) {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let timestamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let filename = "tts_output_\(timestamp).wav"
        let fileURL = documentsPath.appendingPathComponent(filename)

        do {
            try wavData.write(to: fileURL)
            savedAudioPath = fileURL.path
            print("[PocketTTS] Audio saved to: \(fileURL.path)")
            print("[PocketTTS] To extract: xcrun simctl get_app_container booted com.unamentis.PocketTTSDemo data")
        } catch {
            print("[PocketTTS] Failed to save audio: \(error)")
        }
    }

    /// Get the path where audio files are saved (for display in UI)
    func getDocumentsPath() -> String {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path
    }

    /// Extract float samples from WAV data for waveform display
    /// The WAV from Rust uses 32-bit float samples
    private func extractSamplesFromWav(_ data: Data) -> [Float] {
        // WAV header structure:
        // 0-3: "RIFF"
        // 4-7: file size - 8
        // 8-11: "WAVE"
        // 12-15: "fmt "
        // 16-19: fmt chunk size (16 for PCM)
        // 20-21: audio format (3 = IEEE float)
        // 22-23: num channels
        // 24-27: sample rate
        // 28-31: byte rate
        // 32-33: block align
        // 34-35: bits per sample
        // 36-39: "data"
        // 40-43: data size
        // 44+: audio samples

        guard data.count > 44 else {
            print("[PocketTTS] WAV data too short: \(data.count) bytes")
            return []
        }

        // Find the "data" chunk (may not be at fixed offset)
        var dataOffset = 12  // Start after "RIFF" + size + "WAVE"
        while dataOffset < data.count - 8 {
            let chunkId = String(data: data.subdata(in: dataOffset..<dataOffset+4), encoding: .ascii) ?? ""
            let chunkSize = data.subdata(in: dataOffset+4..<dataOffset+8).withUnsafeBytes {
                $0.load(as: UInt32.self).littleEndian
            }

            if chunkId == "data" {
                // Found the data chunk
                let sampleDataOffset = dataOffset + 8
                let sampleData = data.subdata(in: sampleDataOffset..<min(sampleDataOffset + Int(chunkSize), data.count))

                // Convert bytes to Float (32-bit float samples)
                let sampleCount = sampleData.count / MemoryLayout<Float>.size
                var samples = [Float](repeating: 0, count: sampleCount)
                _ = samples.withUnsafeMutableBytes { buffer in
                    sampleData.copyBytes(to: buffer)
                }

                print("[PocketTTS] Extracted \(sampleCount) samples from WAV (data at offset \(sampleDataOffset))")
                return samples
            }

            dataOffset += 8 + Int(chunkSize)
        }

        print("[PocketTTS] Could not find 'data' chunk in WAV")
        return []
    }
}

// MARK: - Audio Player Delegate

class AudioPlayerDelegate: NSObject, AVAudioPlayerDelegate {
    let onFinish: () -> Void

    init(onFinish: @escaping () -> Void) {
        self.onFinish = onFinish
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onFinish()
    }
}

// MARK: - Supporting Types

enum TTSVoice: Int, CaseIterable, Identifiable {
    case alba = 0
    case marius = 1
    case javert = 2
    case jean = 3
    case fantine = 4
    case cosette = 5
    case eponine = 6
    case azelma = 7

    var id: Int { rawValue }

    var displayName: String {
        switch self {
        case .alba: return "Alba"
        case .marius: return "Marius"
        case .javert: return "Javert"
        case .jean: return "Jean"
        case .fantine: return "Fantine"
        case .cosette: return "Cosette"
        case .eponine: return "Eponine"
        case .azelma: return "Azelma"
        }
    }
}

struct TTSStatus {
    let message: String
    let isLoading: Bool
    let type: StatusType

    enum StatusType {
        case info, success, error
    }

    var icon: String {
        switch type {
        case .info: return "info.circle"
        case .success: return "checkmark.circle"
        case .error: return "xmark.circle"
        }
    }

    var color: Color {
        switch type {
        case .info: return .blue
        case .success: return .green
        case .error: return .red
        }
    }

    var backgroundColor: Color {
        switch type {
        case .info: return Color.blue.opacity(0.1)
        case .success: return Color.green.opacity(0.1)
        case .error: return Color.red.opacity(0.1)
        }
    }
}

struct SynthesisTiming {
    let synthesisTime: Double      // Total synthesis time (seconds)
    let audioDuration: Double      // Audio duration (seconds)
    let realtimeFactor: Double     // RTF = audioDuration / synthesisTime
    let ttfaMs: Double?            // Time to first audio (milliseconds) - streaming only
    let chunkCount: Int?           // Number of audio chunks - streaming only

    init(synthesisTime: Double, audioDuration: Double, realtimeFactor: Double, ttfaMs: Double? = nil, chunkCount: Int? = nil) {
        self.synthesisTime = synthesisTime
        self.audioDuration = audioDuration
        self.realtimeFactor = realtimeFactor
        self.ttfaMs = ttfaMs
        self.chunkCount = chunkCount
    }
}

enum TTSError: LocalizedError {
    case modelNotFound
    case engineNotInitialized

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Model files not found in app bundle"
        case .engineNotInitialized:
            return "TTS engine not initialized"
        }
    }
}

// MARK: - Streaming Event Handler

/// Result from streaming synthesis
struct StreamingResult {
    let samples: [Float]
    let ttfaMs: Double           // Time to first audio (milliseconds)
    let totalTime: Double        // Total synthesis time (seconds)
    let audioDuration: Double    // Audio duration (seconds)
    let chunkCount: Int
}

/// Event handler for streaming synthesis with TTFA measurement
class StreamingEventHandler: TtsEventHandler {
    private let startTime: CFAbsoluteTime
    private let sampleRate: UInt32
    private var firstChunkTime: CFAbsoluteTime?
    private var collectedSamples: [Float] = []
    private var chunkCount: Int = 0

    private let completionHandler: (StreamingResult) -> Void
    private let errorHandler: (String) -> Void

    init(
        startTime: CFAbsoluteTime,
        sampleRate: UInt32,
        onComplete: @escaping (StreamingResult) -> Void,
        onError: @escaping (String) -> Void
    ) {
        self.startTime = startTime
        self.sampleRate = sampleRate
        self.completionHandler = onComplete
        self.errorHandler = onError
    }

    func onAudioChunk(chunk: AudioChunk) {
        let now = CFAbsoluteTimeGetCurrent()

        // Record TTFA on first chunk
        if firstChunkTime == nil {
            firstChunkTime = now
            let ttfa = (now - startTime) * 1000.0  // Convert to ms
            print("[PocketTTS] TTFA: \(String(format: "%.1f", ttfa))ms")
        }

        chunkCount += 1

        // Extract float samples from chunk data
        let audioData = chunk.audioData
        let sampleCount = audioData.count / MemoryLayout<Float>.size
        var samples = [Float](repeating: 0, count: sampleCount)
        _ = samples.withUnsafeMutableBytes { buffer in
            audioData.copyBytes(to: buffer)
        }

        collectedSamples.append(contentsOf: samples)

        print("[PocketTTS] Chunk \(chunkCount): \(samples.count) samples, total: \(collectedSamples.count)")
    }

    func onProgress(progress: Float) {
        // Progress updates (optional logging)
    }

    func onComplete() {
        let endTime = CFAbsoluteTimeGetCurrent()
        let totalTime = endTime - startTime
        let audioDuration = Double(collectedSamples.count) / Double(sampleRate)
        let ttfaMs = ((firstChunkTime ?? startTime) - startTime) * 1000.0

        let result = StreamingResult(
            samples: collectedSamples,
            ttfaMs: ttfaMs,
            totalTime: totalTime,
            audioDuration: audioDuration,
            chunkCount: chunkCount
        )

        completionHandler(result)
    }

    func onError(message: String) {
        errorHandler(message)
    }
}
