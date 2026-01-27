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
    let latents_file: String
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

    func generateFromLatents() {
        guard let phrase = selectedPhrase else { return }
        guard let engine = engine else {
            status = "ERROR: TTS engine not initialized"
            return
        }

        // Load latents from file
        guard let latentsURL = Bundle.main.url(forResource: phrase.latents_file.replacingOccurrences(of: ".f32", with: ""), withExtension: "f32", subdirectory: "ReferenceAudio") else {
            status = "ERROR: Latents file not found: \(phrase.latents_file)"
            return
        }

        isGenerating = true
        status = "Generating audio from latents..."

        Task {
            do {
                let latentsData = try Data(contentsOf: latentsURL)

                // Calculate dimensions from file size (each float is 4 bytes)
                let floatCount = latentsData.count / 4
                let latentDim = 32
                let numFrames = floatCount / latentDim

                print("[ReferenceTest] Loaded latents: \(numFrames) frames x \(latentDim) dim")

                let startTime = CFAbsoluteTimeGetCurrent()

                // Call Rust engine to decode latents
                let result = try engine.decodeLatents(latentsData: latentsData, numFrames: UInt32(numFrames))

                let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
                generationTimeMs = elapsed

                // Result is already WAV-encoded from Rust
                generatedAudioData = result.audioData

                print("[ReferenceTest] Generated audio: \(result.audioData.count) bytes in \(elapsed)ms")

                // Compute correlation
                if let refData = referenceAudioData, let genData = generatedAudioData {
                    computeCorrelation(reference: refData, generated: genData)
                }

                status = String(format: "Generated in %.1fms. Correlation: %.4f", elapsed, correlationResult ?? 0)

            } catch {
                status = "ERROR: Generation failed: \(error)"
                print("[ReferenceTest] Generation failed: \(error)")
            }

            isGenerating = false
        }
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

                // Generate button
                Button(action: viewModel.generateFromLatents) {
                    HStack {
                        if viewModel.isGenerating {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(viewModel.isGenerating ? "Generating..." : "Generate from Latents")
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
