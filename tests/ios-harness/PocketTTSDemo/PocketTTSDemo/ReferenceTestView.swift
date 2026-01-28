//
//  ReferenceTestView.swift
//  PocketTTSDemo
//
//  AB testing view for comparing reference (Python) vs generated (Rust) audio.
//

import SwiftUI
import AVFoundation

/// Reference phrase for testing
struct ReferencePhrase: Identifiable, Codable {
    let id: String
    let text: String
    let description: String
    let audio_file: String
    let latents_file: String?  // Optional - not needed for direct audio playback testing
}

/// Manifest for reference audio files
struct ReferenceManifest: Codable {
    let sample_rate: Int
    let phrases: [ReferencePhrase]
}

@MainActor
class ReferenceTestViewModel: ObservableObject {
    @Published var phrases: [ReferencePhrase] = []
    @Published var selectedPhrase: ReferencePhrase?
    @Published var referenceAudioData: Data?
    @Published var generatedAudioData: Data?
    @Published var isGenerating = false
    @Published var status: String = "Load a phrase to begin"
    @Published var correlationResult: Double?
    @Published var generationTimeMs: Double?

    private var engine: PocketTtsEngine?
    private var audioPlayer: AVAudioPlayer?
    private var audioPlayerDelegate: AudioPlayerDelegateRef?

    @Published var isPlayingReference = false
    @Published var isPlayingGenerated = false

    init() {
        loadManifest()
    }

    func setEngine(_ engine: PocketTtsEngine?) {
        self.engine = engine
    }

    private func loadManifest() {
        guard let manifestURL = Bundle.main.url(forResource: "manifest", withExtension: "json", subdirectory: "ReferenceAudio") else {
            status = "ERROR: manifest.json not found in bundle"
            print("[ReferenceTest] manifest.json not found")
            return
        }

        do {
            let data = try Data(contentsOf: manifestURL)
            let manifest = try JSONDecoder().decode(ReferenceManifest.self, from: data)
            phrases = manifest.phrases
            status = "Loaded \(phrases.count) reference phrases"
            print("[ReferenceTest] Loaded \(phrases.count) phrases")
        } catch {
            status = "ERROR: Failed to load manifest: \(error)"
            print("[ReferenceTest] Failed to load manifest: \(error)")
        }
    }

    func selectPhrase(_ phrase: ReferencePhrase) {
        selectedPhrase = phrase
        referenceAudioData = nil
        generatedAudioData = nil
        correlationResult = nil
        generationTimeMs = nil
        loadReferenceAudio(phrase)
    }

    private func loadReferenceAudio(_ phrase: ReferencePhrase) {
        guard let audioURL = Bundle.main.url(forResource: phrase.audio_file.replacingOccurrences(of: ".wav", with: ""), withExtension: "wav", subdirectory: "ReferenceAudio") else {
            status = "ERROR: Reference audio not found: \(phrase.audio_file)"
            return
        }

        do {
            referenceAudioData = try Data(contentsOf: audioURL)
            status = "Reference loaded. Tap 'Generate' to compare."
            print("[ReferenceTest] Loaded reference audio: \(referenceAudioData?.count ?? 0) bytes")
        } catch {
            status = "ERROR: Failed to load reference audio: \(error)"
        }
    }

    func generateWithRustTTS() {
        guard let phrase = selectedPhrase else { return }
        guard let engine = engine else {
            status = "ERROR: TTS engine not initialized"
            return
        }

        isGenerating = true
        status = "Generating audio with Rust TTS (streaming)..."

        Task {
            do {
                print("[ReferenceTest] Generating TTS (STREAMING MODE) for: \(phrase.text)")

                let startTime = CFAbsoluteTimeGetCurrent()

                // Use STREAMING mode - this is the only mode we use on-device
                // Sync mode exists but is not for current use (latency is king)
                let handler = ABTestStreamingHandler(
                    startTime: startTime,
                    onComplete: { [weak self] audioData, sampleRate, ttfaMs in
                        guard let self = self else { return }

                        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
                        self.generationTimeMs = elapsed
                        self.ttfaMs = ttfaMs

                        // Convert raw float audio data to WAV format for playback
                        self.generatedAudioData = self.rawAudioToWav(audioData: audioData, sampleRate: sampleRate)

                        let sampleCount = audioData.count / MemoryLayout<Float>.size
                        print("[ReferenceTest] Generated audio: \(sampleCount) samples in \(elapsed)ms (TTFA: \(ttfaMs)ms)")

                        // Compute correlation between Python reference and Rust output
                        if let refData = self.referenceAudioData, let genData = self.generatedAudioData {
                            self.computeCorrelation(reference: refData, generated: genData)
                        }

                        self.status = String(format: "Generated in %.0fms (TTFA: %.0fms). Ready for AB comparison.", elapsed, ttfaMs)
                        self.isGenerating = false
                    },
                    onError: { [weak self] error in
                        self?.status = "ERROR: Streaming failed: \(error)"
                        print("[ReferenceTest] Streaming failed: \(error)")
                        self?.isGenerating = false
                    }
                )

                try engine.startTrueStreaming(text: phrase.text, handler: handler)

            } catch {
                status = "ERROR: Generation failed: \(error)"
                print("[ReferenceTest] Generation failed: \(error)")
                isGenerating = false
            }
        }
    }

    @Published var ttfaMs: Double?

    /// Convert raw float audio data to WAV format
    /// The input audioData is already in 32-bit float format from the TTS engine
    private func rawAudioToWav(audioData: Data, sampleRate: UInt32) -> Data {
        var wavData = Data()

        // WAV header
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 32
        let byteRate = sampleRate * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(audioData.count)
        let fileSize = 36 + dataSize

        // RIFF header
        wavData.append(contentsOf: "RIFF".utf8)
        wavData.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        wavData.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        wavData.append(contentsOf: "fmt ".utf8)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(3).littleEndian) { Array($0) }) // IEEE float
        wavData.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: sampleRate.littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

        // data chunk
        wavData.append(contentsOf: "data".utf8)
        wavData.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        // Sample data (already in correct format)
        wavData.append(audioData)

        return wavData
    }

    private func computeCorrelation(reference: Data, generated: Data) {
        // Extract float samples from both WAV files
        let refSamples = extractSamplesFromWav(reference)
        let genSamples = extractSamplesFromWav(generated)

        guard !refSamples.isEmpty && !genSamples.isEmpty else {
            correlationResult = 0
            return
        }

        // Use minimum length
        let minLen = min(refSamples.count, genSamples.count)
        let ref = Array(refSamples.prefix(minLen))
        let gen = Array(genSamples.prefix(minLen))

        // Compute Pearson correlation
        let n = Double(minLen)
        let sumRef = ref.reduce(0, +)
        let sumGen = gen.reduce(0, +)
        let sumRefSq = ref.map { $0 * $0 }.reduce(0, +)
        let sumGenSq = gen.map { $0 * $0 }.reduce(0, +)
        let sumProd = zip(ref, gen).map { $0 * $1 }.reduce(0, +)

        let numerator = n * Double(sumProd) - Double(sumRef) * Double(sumGen)
        let denominator = sqrt((n * Double(sumRefSq) - Double(sumRef) * Double(sumRef)) *
                              (n * Double(sumGenSq) - Double(sumGen) * Double(sumGen)))

        if denominator > 0 {
            correlationResult = numerator / denominator
        } else {
            correlationResult = 0
        }

        print("[ReferenceTest] Correlation: \(correlationResult ?? 0) (ref: \(refSamples.count), gen: \(genSamples.count) samples)")
    }

    private func extractSamplesFromWav(_ data: Data) -> [Float] {
        guard data.count > 44 else { return [] }

        // Find data chunk
        var offset = 12
        while offset < data.count - 8 {
            let chunkId = String(data: data.subdata(in: offset..<offset+4), encoding: .ascii) ?? ""
            let chunkSize = data.subdata(in: offset+4..<offset+8).withUnsafeBytes {
                $0.load(as: UInt32.self).littleEndian
            }

            if chunkId == "data" {
                let sampleOffset = offset + 8
                let sampleData = data.subdata(in: sampleOffset..<min(sampleOffset + Int(chunkSize), data.count))
                let sampleCount = sampleData.count / MemoryLayout<Float>.size
                var samples = [Float](repeating: 0, count: sampleCount)
                _ = samples.withUnsafeMutableBytes { buffer in
                    sampleData.copyBytes(to: buffer)
                }
                return samples
            }

            offset += 8 + Int(chunkSize)
        }

        return []
    }

    func playReference() {
        guard let data = referenceAudioData else { return }
        playAudio(data: data, isReference: true)
    }

    func playGenerated() {
        guard let data = generatedAudioData else { return }
        playAudio(data: data, isReference: false)
    }

    private func playAudio(data: Data, isReference: Bool) {
        stopPlayback()

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            audioPlayer = try AVAudioPlayer(data: data)
            audioPlayerDelegate = AudioPlayerDelegateRef { [weak self] in
                Task { @MainActor in
                    self?.isPlayingReference = false
                    self?.isPlayingGenerated = false
                }
            }
            audioPlayer?.delegate = audioPlayerDelegate
            audioPlayer?.play()

            if isReference {
                isPlayingReference = true
            } else {
                isPlayingGenerated = true
            }
        } catch {
            print("[ReferenceTest] Playback failed: \(error)")
        }
    }

    func stopPlayback() {
        audioPlayer?.stop()
        isPlayingReference = false
        isPlayingGenerated = false
    }

    /// Export generated audio to Documents for external analysis
    func exportGeneratedAudio() -> URL? {
        guard let data = generatedAudioData, let phrase = selectedPhrase else { return nil }

        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let filename = "rust_\(phrase.id)_generated.wav"
        let fileURL = documentsPath.appendingPathComponent(filename)

        do {
            try data.write(to: fileURL)
            print("[ReferenceTest] Exported to: \(fileURL.path)")
            return fileURL
        } catch {
            print("[ReferenceTest] Export failed: \(error)")
            return nil
        }
    }
}

// Separate delegate class to avoid retain issues
class AudioPlayerDelegateRef: NSObject, AVAudioPlayerDelegate {
    let onFinish: () -> Void

    init(onFinish: @escaping () -> Void) {
        self.onFinish = onFinish
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onFinish()
    }
}

/// Streaming handler for AB test audio generation
/// Collects audio chunks and measures TTFA (Time To First Audio)
class ABTestStreamingHandler: TtsEventHandler {
    private var audioData: Data = Data()
    private let startTime: CFAbsoluteTime
    private var ttfaMs: Double = 0
    private var firstChunkReceived = false
    private var chunkSampleRate: UInt32 = 24000  // Default, updated from first chunk
    private let completionHandler: (Data, UInt32, Double) -> Void
    private let errorHandler: (String) -> Void

    init(
        startTime: CFAbsoluteTime,
        onComplete: @escaping (Data, UInt32, Double) -> Void,
        onError: @escaping (String) -> Void
    ) {
        self.startTime = startTime
        self.completionHandler = onComplete
        self.errorHandler = onError
    }

    func onAudioChunk(chunk: AudioChunk) {
        // Track TTFA
        if !firstChunkReceived {
            ttfaMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
            firstChunkReceived = true
            chunkSampleRate = chunk.sampleRate
            print("[ABTest] First audio chunk received - TTFA: \(String(format: "%.1f", ttfaMs))ms")
        }

        // Collect audio data
        audioData.append(chunk.audioData)
    }

    func onProgress(progress: Float) {
        // Progress tracking (not used for AB test but required by protocol)
    }

    func onError(message: String) {
        print("[ABTest] Error: \(message)")
        DispatchQueue.main.async {
            self.errorHandler(message)
        }
    }

    func onComplete() {
        print("[ABTest] Streaming complete - Total audio data: \(audioData.count) bytes")
        DispatchQueue.main.async {
            self.completionHandler(self.audioData, self.chunkSampleRate, self.ttfaMs)
        }
    }
}

struct ReferenceTestView: View {
    @ObservedObject var viewModel: ReferenceTestViewModel

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                Text("Reference Audio Testing")
                    .font(.headline)

                Text("Compare Rust-generated audio against Python reference")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Divider()

                // Phrase selection
                VStack(alignment: .leading, spacing: 8) {
                    Text("Select Test Phrase")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    ForEach(viewModel.phrases) { phrase in
                        Button(action: { viewModel.selectPhrase(phrase) }) {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(phrase.id.capitalized)
                                        .font(.headline)
                                    Text(phrase.text)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                        .lineLimit(2)
                                }
                                Spacer()
                                if viewModel.selectedPhrase?.id == phrase.id {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.accentColor)
                                }
                            }
                            .padding()
                            .background(viewModel.selectedPhrase?.id == phrase.id ? Color.accentColor.opacity(0.1) : Color(.systemGray6))
                            .cornerRadius(8)
                        }
                        .buttonStyle(.plain)
                    }
                }

                Divider()

                // Status
                Text(viewModel.status)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                // Generate button - uses actual Rust TTS (not latent decoding)
                Button(action: viewModel.generateWithRustTTS) {
                    HStack {
                        if viewModel.isGenerating {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(viewModel.isGenerating ? "Generating..." : "Generate with Rust TTS")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.selectedPhrase == nil || viewModel.isGenerating)

                // Results
                if let correlation = viewModel.correlationResult {
                    VStack(spacing: 12) {
                        // Correlation indicator
                        HStack {
                            Text("Correlation:")
                                .font(.headline)
                            Spacer()
                            Text(String(format: "%.4f", correlation))
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(correlationColor(correlation))
                        }

                        // Visual indicator
                        ProgressView(value: max(0, correlation), total: 1.0)
                            .tint(correlationColor(correlation))

                        // Interpretation
                        Text(correlationInterpretation(correlation))
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if let timeMs = viewModel.generationTimeMs {
                            Text(String(format: "Generation time: %.1fms", timeMs))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                }

                // Playback controls
                if viewModel.referenceAudioData != nil || viewModel.generatedAudioData != nil {
                    VStack(spacing: 12) {
                        Text("AB Testing")
                            .font(.subheadline)
                            .fontWeight(.medium)

                        HStack(spacing: 16) {
                            // Reference playback
                            Button(action: {
                                if viewModel.isPlayingReference {
                                    viewModel.stopPlayback()
                                } else {
                                    viewModel.playReference()
                                }
                            }) {
                                VStack {
                                    Image(systemName: viewModel.isPlayingReference ? "stop.fill" : "play.fill")
                                        .font(.title2)
                                    Text("Reference")
                                        .font(.caption)
                                    Text("(Python)")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue.opacity(0.1))
                                .cornerRadius(8)
                            }
                            .disabled(viewModel.referenceAudioData == nil)

                            // Generated playback
                            Button(action: {
                                if viewModel.isPlayingGenerated {
                                    viewModel.stopPlayback()
                                } else {
                                    viewModel.playGenerated()
                                }
                            }) {
                                VStack {
                                    Image(systemName: viewModel.isPlayingGenerated ? "stop.fill" : "play.fill")
                                        .font(.title2)
                                    Text("Generated")
                                        .font(.caption)
                                    Text("(Rust)")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.green.opacity(0.1))
                                .cornerRadius(8)
                            }
                            .disabled(viewModel.generatedAudioData == nil)
                        }
                    }
                }

                // Export button
                if viewModel.generatedAudioData != nil {
                    Button(action: {
                        if let url = viewModel.exportGeneratedAudio() {
                            viewModel.status = "Exported to: \(url.lastPathComponent)"
                        }
                    }) {
                        Label("Export Generated Audio", systemImage: "square.and.arrow.up")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }

                Spacer()
            }
            .padding()
        }
    }

    private func correlationColor(_ correlation: Double) -> Color {
        if correlation >= 0.99 { return .green }
        if correlation >= 0.95 { return .yellow }
        if correlation >= 0.8 { return .orange }
        return .red
    }

    private func correlationInterpretation(_ correlation: Double) -> String {
        if correlation >= 0.999 { return "Excellent - Nearly identical" }
        if correlation >= 0.99 { return "Very good - Minor differences" }
        if correlation >= 0.95 { return "Good - Some differences" }
        if correlation >= 0.8 { return "Fair - Noticeable differences" }
        if correlation >= 0.5 { return "Poor - Significant differences" }
        return "Very poor - Major mismatch"
    }
}

#Preview {
    ReferenceTestView(viewModel: ReferenceTestViewModel())
}
