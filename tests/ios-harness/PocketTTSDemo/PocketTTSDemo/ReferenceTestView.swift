//
//  ReferenceTestView.swift
//  PocketTTSDemo
//
//  3-way audio comparison: Reference (Python) vs Release (Official) vs Current (Dev Build).
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
    let noise_id: String?      // Captured-noise prefix (e.g. "phrase_00") for noise-matched parity
    let seed: UInt32?          // Seed used when the Python reference + noise were captured
}

/// Manifest for reference audio files
struct ReferenceManifest: Codable {
    let sample_rate: Int
    let phrases: [ReferencePhrase]
}

/// Which audio source is being played
enum AudioSource {
    case reference
    case release
    case current
}

@MainActor
class ReferenceTestViewModel: ObservableObject {
    @Published var phrases: [ReferencePhrase] = []
    @Published var selectedPhrase: ReferencePhrase?
    @Published var referenceAudioData: Data?
    @Published var releaseAudioData: Data?
    @Published var generatedAudioData: Data?
    @Published var isGenerating = false
    @Published var status: String = "Load a phrase to begin"
    @Published var correlationRefVsCurrent: Double?
    @Published var correlationRelVsCurrent: Double?
    @Published var generationTimeMs: Double?
    @Published var ttfaMs: Double?
    @Published var hasReleaseSaved: Bool = false

    private var engine: PocketTtsEngine?
    private var audioPlayer: AVAudioPlayer?
    private var audioPlayerDelegate: AudioPlayerDelegateRef?

    @Published var isPlayingReference = false
    @Published var isPlayingRelease = false
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
        releaseAudioData = nil
        generatedAudioData = nil
        correlationRefVsCurrent = nil
        correlationRelVsCurrent = nil
        generationTimeMs = nil
        ttfaMs = nil
        loadReferenceAudio(phrase)
        loadReleaseAudio(phrase)
    }

    private func loadReferenceAudio(_ phrase: ReferencePhrase) {
        guard let audioURL = Bundle.main.url(forResource: phrase.audio_file.replacingOccurrences(of: ".wav", with: ""), withExtension: "wav", subdirectory: "ReferenceAudio") else {
            status = "ERROR: Reference audio not found: \(phrase.audio_file)"
            return
        }

        do {
            referenceAudioData = try Data(contentsOf: audioURL)
            print("[ReferenceTest] Loaded reference audio: \(referenceAudioData?.count ?? 0) bytes")
        } catch {
            status = "ERROR: Failed to load reference audio: \(error)"
        }
    }

    // MARK: - Release Audio (Documents directory)

    private func releaseAudioURL(for phrase: ReferencePhrase) -> URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsPath.appendingPathComponent("release_\(phrase.id).wav")
    }

    private func loadReleaseAudio(_ phrase: ReferencePhrase) {
        let fileURL = releaseAudioURL(for: phrase)

        if FileManager.default.fileExists(atPath: fileURL.path) {
            do {
                releaseAudioData = try Data(contentsOf: fileURL)
                hasReleaseSaved = true
                print("[ReferenceTest] Loaded release audio from Documents: \(releaseAudioData?.count ?? 0) bytes")
            } catch {
                releaseAudioData = nil
                hasReleaseSaved = false
                print("[ReferenceTest] Failed to load release audio: \(error)")
            }
        } else {
            releaseAudioData = nil
            hasReleaseSaved = false
        }

        updateStatusAfterLoad()
    }

    private func updateStatusAfterLoad() {
        if referenceAudioData != nil && hasReleaseSaved {
            status = "Reference and release loaded. Tap 'Generate' to compare all three."
        } else if referenceAudioData != nil {
            status = "Reference loaded. No release baseline yet. Tap 'Generate' to start."
        } else {
            status = "Select a phrase to begin."
        }
    }

    func saveAsRelease() {
        guard let data = generatedAudioData, let phrase = selectedPhrase else { return }

        let fileURL = releaseAudioURL(for: phrase)

        do {
            try data.write(to: fileURL)
            releaseAudioData = data
            hasReleaseSaved = true
            status = "Saved current audio as release baseline for '\(phrase.id)'"
            print("[ReferenceTest] Saved release baseline to: \(fileURL.path)")
        } catch {
            status = "ERROR: Failed to save release audio: \(error)"
            print("[ReferenceTest] Save failed: \(error)")
        }
    }

    func clearRelease() {
        guard let phrase = selectedPhrase else { return }

        let fileURL = releaseAudioURL(for: phrase)

        do {
            try FileManager.default.removeItem(at: fileURL)
            releaseAudioData = nil
            hasReleaseSaved = false
            correlationRelVsCurrent = nil
            status = "Cleared release baseline for '\(phrase.id)'"
            print("[ReferenceTest] Cleared release baseline")
        } catch {
            status = "ERROR: Failed to clear release audio: \(error)"
            print("[ReferenceTest] Clear failed: \(error)")
        }
    }

    // MARK: - TTS Generation

    func generateWithRustTTS() {
        guard let phrase = selectedPhrase else { return }
        guard let engine = engine else {
            status = "ERROR: TTS engine not initialized"
            return
        }

        // Noise-matched generation: inject the captured Python noise tensors so the on-device
        // output reproduces the reference (the project's standard parity check), instead of
        // sampling fresh random noise (which is ~uncorrelated with the fixed-noise reference).
        guard let refDir = Bundle.main
            .url(forResource: "manifest", withExtension: "json", subdirectory: "ReferenceAudio")?
            .deletingLastPathComponent() else {
            status = "ERROR: reference directory not found in bundle"
            return
        }
        let noiseDir = refDir.appendingPathComponent("noise").path
        let noiseId = phrase.noise_id ?? phrase.id
        let seed = phrase.seed ?? 42
        let text = phrase.text

        isGenerating = true
        status = "Generating (noise-matched) with Rust TTS…"

        Task {
            let startTime = CFAbsoluteTimeGetCurrent()
            do {
                print("[ReferenceTest] Noise-matched generation for \(noiseId) (seed \(seed)): \(text)")
                let result = try engine.synthesizeNoiseMatched(
                    text: text, voiceIndex: 0, noiseDir: noiseDir, phraseId: noiseId, seed: seed)
                let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

                self.generationTimeMs = elapsed
                self.generatedAudioData = Data(result.audioData)

                if let refData = self.referenceAudioData, let gen = self.generatedAudioData {
                    self.correlationRefVsCurrent = self.computeCorrelation(a: refData, b: gen)
                }
                if let relData = self.releaseAudioData, let gen = self.generatedAudioData {
                    self.correlationRelVsCurrent = self.computeCorrelation(a: relData, b: gen)
                }

                let corr = self.correlationRefVsCurrent.map { String(format: "%.4f", $0) } ?? "n/a"
                self.status = "Noise-matched in \(Int(elapsed))ms — corr vs Python: \(corr)"
                print("[ReferenceTest] corr vs Python (noise-matched) = \(corr)")
                self.isGenerating = false
            } catch {
                self.status = "ERROR: Generation failed: \(error)"
                print("[ReferenceTest] Generation failed: \(error)")
                self.isGenerating = false
            }
        }
    }

    // MARK: - Audio Format Helpers

    /// Convert raw float audio data to WAV format
    private func rawAudioToWav(audioData: Data, sampleRate: UInt32) -> Data {
        var wavData = Data()

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
        wavData.append(audioData)

        return wavData
    }

    /// Compute Pearson correlation between two WAV audio buffers
    private func computeCorrelation(a: Data, b: Data) -> Double? {
        let samplesA = extractSamplesFromWav(a)
        let samplesB = extractSamplesFromWav(b)

        guard !samplesA.isEmpty && !samplesB.isEmpty else { return nil }

        let minLen = min(samplesA.count, samplesB.count)
        let arrA = Array(samplesA.prefix(minLen))
        let arrB = Array(samplesB.prefix(minLen))

        let n = Double(minLen)
        let sumA = arrA.reduce(0.0) { $0 + Double($1) }
        let sumB = arrB.reduce(0.0) { $0 + Double($1) }
        let sumASq = arrA.reduce(0.0) { $0 + Double($1) * Double($1) }
        let sumBSq = arrB.reduce(0.0) { $0 + Double($1) * Double($1) }
        let sumProd = zip(arrA, arrB).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }

        let numerator = n * sumProd - sumA * sumB
        let denominator = sqrt((n * sumASq - sumA * sumA) * (n * sumBSq - sumB * sumB))

        guard denominator > 0 else { return 0 }

        let result = numerator / denominator
        print("[ReferenceTest] Correlation: \(result) (a: \(samplesA.count), b: \(samplesB.count) samples)")
        return result
    }

    /// Extract float samples from a WAV file, handling both Int16 PCM and Float32 IEEE formats
    private func extractSamplesFromWav(_ data: Data) -> [Float] {
        guard data.count > 44 else { return [] }

        var formatCode: UInt16 = 3  // Default to float
        var bitsPerSample: UInt16 = 32
        var offset = 12

        while offset < data.count - 8 {
            let chunkId = String(data: data.subdata(in: offset..<offset+4), encoding: .ascii) ?? ""
            let chunkSize = data.subdata(in: offset+4..<offset+8).withUnsafeBytes {
                $0.load(as: UInt32.self).littleEndian
            }

            if chunkId == "fmt " && chunkSize >= 16 {
                formatCode = data.subdata(in: offset+8..<offset+10).withUnsafeBytes {
                    $0.load(as: UInt16.self).littleEndian
                }
                bitsPerSample = data.subdata(in: offset+22..<offset+24).withUnsafeBytes {
                    $0.load(as: UInt16.self).littleEndian
                }
            }

            if chunkId == "data" {
                let sampleOffset = offset + 8
                let sampleData = data.subdata(in: sampleOffset..<min(sampleOffset + Int(chunkSize), data.count))

                if formatCode == 1 && bitsPerSample == 16 {
                    // Int16 PCM -> Float conversion
                    let sampleCount = sampleData.count / MemoryLayout<Int16>.size
                    var int16Samples = [Int16](repeating: 0, count: sampleCount)
                    _ = int16Samples.withUnsafeMutableBytes { buffer in
                        sampleData.copyBytes(to: buffer)
                    }
                    return int16Samples.map { Float($0) / 32768.0 }
                } else {
                    // Float32 IEEE
                    let sampleCount = sampleData.count / MemoryLayout<Float>.size
                    var samples = [Float](repeating: 0, count: sampleCount)
                    _ = samples.withUnsafeMutableBytes { buffer in
                        sampleData.copyBytes(to: buffer)
                    }
                    return samples
                }
            }

            offset += 8 + Int(chunkSize)
        }

        return []
    }

    // MARK: - Playback

    func playReference() {
        guard let data = referenceAudioData else { return }
        playAudio(data: data, source: .reference)
    }

    func playRelease() {
        guard let data = releaseAudioData else { return }
        playAudio(data: data, source: .release)
    }

    func playGenerated() {
        guard let data = generatedAudioData else { return }
        playAudio(data: data, source: .current)
    }

    private func playAudio(data: Data, source: AudioSource) {
        stopPlayback()

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            audioPlayer = try AVAudioPlayer(data: data)
            audioPlayerDelegate = AudioPlayerDelegateRef { [weak self] in
                Task { @MainActor in
                    self?.isPlayingReference = false
                    self?.isPlayingRelease = false
                    self?.isPlayingGenerated = false
                }
            }
            audioPlayer?.delegate = audioPlayerDelegate
            audioPlayer?.play()

            switch source {
            case .reference: isPlayingReference = true
            case .release: isPlayingRelease = true
            case .current: isPlayingGenerated = true
            }
        } catch {
            print("[ReferenceTest] Playback failed: \(error)")
        }
    }

    func stopPlayback() {
        audioPlayer?.stop()
        isPlayingReference = false
        isPlayingRelease = false
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
        if !firstChunkReceived {
            ttfaMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
            firstChunkReceived = true
            chunkSampleRate = chunk.sampleRate
            print("[ABTest] First audio chunk received - TTFA: \(String(format: "%.1f", ttfaMs))ms")
        }
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

// MARK: - Reusable Playback Card

struct AudioPlaybackCard: View {
    let title: String
    let subtitle: String
    let color: Color
    let isPlaying: Bool
    let isDisabled: Bool
    let statusText: String?
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack {
                Image(systemName: isPlaying ? "stop.fill" : "play.fill")
                    .font(.title2)
                    .frame(width: 44)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                Spacer()
                if let statusText = statusText {
                    Text(statusText)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .background(color.opacity(isDisabled ? 0.05 : 0.1))
            .cornerRadius(8)
            .opacity(isDisabled ? 0.5 : 1.0)
        }
        .disabled(isDisabled)
        .buttonStyle(.plain)
    }
}

// MARK: - Correlation Row

struct CorrelationRow: View {
    let label: String
    let value: Double

    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Text(label)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text(String(format: "%.4f", value))
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundColor(correlationColor(value))
            }

            ProgressView(value: max(0, value), total: 1.0)
                .tint(correlationColor(value))

            Text(correlationInterpretation(value))
                .font(.caption2)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
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

// MARK: - Main View

struct ReferenceTestView: View {
    @ObservedObject var viewModel: ReferenceTestViewModel

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                Text("Audio Comparison")
                    .font(.headline)

                Text("Compare reference, release, and current build audio")
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

                // Generate button
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

                // Correlation results
                if viewModel.correlationRefVsCurrent != nil || viewModel.correlationRelVsCurrent != nil {
                    VStack(spacing: 12) {
                        if let refCorr = viewModel.correlationRefVsCurrent {
                            CorrelationRow(label: "vs Reference (Python)", value: refCorr)
                        }
                        if let relCorr = viewModel.correlationRelVsCurrent {
                            CorrelationRow(label: "vs Release (Official)", value: relCorr)
                        }
                        if let timeMs = viewModel.generationTimeMs {
                            HStack {
                                Text(String(format: "Generated in %.0fms", timeMs))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                if let ttfa = viewModel.ttfaMs {
                                    Text(String(format: "(TTFA: %.0fms)", ttfa))
                                        .font(.caption)
                                        .foregroundColor(ttfa <= 300 ? .green : .orange)
                                }
                            }
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                }

                // Playback controls - 3 cards
                if viewModel.referenceAudioData != nil || viewModel.releaseAudioData != nil || viewModel.generatedAudioData != nil {
                    VStack(spacing: 8) {
                        Text("Listen & Compare")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        AudioPlaybackCard(
                            title: "Reference (Python)",
                            subtitle: "Pre-recorded gold standard",
                            color: .blue,
                            isPlaying: viewModel.isPlayingReference,
                            isDisabled: viewModel.referenceAudioData == nil,
                            statusText: viewModel.referenceAudioData != nil ? "Bundled" : nil,
                            onTap: {
                                if viewModel.isPlayingReference {
                                    viewModel.stopPlayback()
                                } else {
                                    viewModel.playReference()
                                }
                            }
                        )

                        AudioPlaybackCard(
                            title: "Release (Official)",
                            subtitle: viewModel.hasReleaseSaved ? "Previously saved on-device" : "No baseline yet",
                            color: .purple,
                            isPlaying: viewModel.isPlayingRelease,
                            isDisabled: viewModel.releaseAudioData == nil,
                            statusText: viewModel.hasReleaseSaved ? "Saved" : nil,
                            onTap: {
                                if viewModel.isPlayingRelease {
                                    viewModel.stopPlayback()
                                } else {
                                    viewModel.playRelease()
                                }
                            }
                        )

                        AudioPlaybackCard(
                            title: "Current (Dev Build)",
                            subtitle: "Fresh on-device generation",
                            color: .green,
                            isPlaying: viewModel.isPlayingGenerated,
                            isDisabled: viewModel.generatedAudioData == nil,
                            statusText: viewModel.generationTimeMs.map { String(format: "%.0fms", $0) },
                            onTap: {
                                if viewModel.isPlayingGenerated {
                                    viewModel.stopPlayback()
                                } else {
                                    viewModel.playGenerated()
                                }
                            }
                        )
                    }
                }

                // Action buttons
                if viewModel.generatedAudioData != nil {
                    VStack(spacing: 8) {
                        Button(action: {
                            viewModel.saveAsRelease()
                        }) {
                            Label("Save as Release Baseline", systemImage: "checkmark.seal.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .tint(.purple)

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
                }

                if viewModel.hasReleaseSaved {
                    Button(action: {
                        viewModel.clearRelease()
                    }) {
                        Label("Clear Release Baseline", systemImage: "trash")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }

                Spacer()
            }
            .padding()
        }
    }
}

#Preview {
    ReferenceTestView(viewModel: ReferenceTestViewModel())
}
