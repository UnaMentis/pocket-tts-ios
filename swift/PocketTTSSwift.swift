//
//  PocketTTSSwift.swift
//  UnaMentis
//
//  Swift wrapper for the Rust Pocket TTS implementation.
//  Provides an async/await API for SwiftUI integration.
//

import Foundation

// MARK: - Swift Wrapper

/// Swift-native wrapper for Pocket TTS with async/await support
@available(iOS 17.0, *)
public actor PocketTTSSwift {

    // MARK: - Types

    /// Audio chunk from streaming synthesis
    public struct AudioChunk: Sendable {
        public let audioData: Data
        public let sampleRate: UInt32
        public let isFinal: Bool
    }

    /// Synthesis result
    public struct SynthesisResult: Sendable {
        public let audioData: Data
        public let sampleRate: UInt32
        public let channels: UInt32
        public let durationSeconds: Double
    }

    /// Voice information
    public struct Voice: Sendable, Identifiable {
        public let id: UInt32
        public let name: String
        public let gender: String
        public let description: String

        public init(index: UInt32, name: String, gender: String, description: String) {
            self.id = index
            self.name = name
            self.gender = gender
            self.description = description
        }
    }

    /// Configuration options
    public struct Config: Sendable {
        public var voiceIndex: UInt32
        public var temperature: Float
        public var topP: Float
        public var speed: Float
        public var consistencySteps: UInt32
        public var useFixedSeed: Bool
        public var seed: UInt32

        public init(
            voiceIndex: UInt32 = 0,
            temperature: Float = 0.7,
            topP: Float = 0.9,
            speed: Float = 1.0,
            consistencySteps: UInt32 = 2,
            useFixedSeed: Bool = false,
            seed: UInt32 = 42
        ) {
            self.voiceIndex = voiceIndex
            self.temperature = temperature
            self.topP = topP
            self.speed = speed
            self.consistencySteps = consistencySteps
            self.useFixedSeed = useFixedSeed
            self.seed = seed
        }

        public static let `default` = Config()
        public static let lowLatency = Config(consistencySteps: 1)
        public static let highQuality = Config(consistencySteps: 4, temperature: 0.5)
    }

    // MARK: - Properties

    private var engine: PocketTTSEngine?
    private let modelPath: String

    // MARK: - Initialization

    public init(modelPath: String) {
        self.modelPath = modelPath
    }

    // MARK: - Public API

    /// Load the TTS model
    public func load() async throws {
        engine = try PocketTTSEngine(modelPath: modelPath)
    }

    /// Check if model is loaded
    public var isLoaded: Bool {
        engine?.isReady() ?? false
    }

    /// Unload model to free memory
    public func unload() {
        engine?.unload()
        engine = nil
    }

    /// Configure synthesis parameters
    public func configure(_ config: Config) throws {
        guard let engine else {
            throw PocketTTSSwiftError.modelNotLoaded
        }

        let rustConfig = TTSConfig(
            voiceIndex: config.voiceIndex,
            temperature: config.temperature,
            topP: config.topP,
            speed: config.speed,
            consistencySteps: config.consistencySteps,
            useFixedSeed: config.useFixedSeed,
            seed: config.seed
        )

        try engine.configure(config: rustConfig)
    }

    /// Get available voices
    public static var availableVoices: [Voice] {
        pocket_tts_ios.availableVoices().map { info in
            Voice(
                index: info.index,
                name: info.name,
                gender: info.gender,
                description: info.description
            )
        }
    }

    /// Synthesize text to audio
    public func synthesize(text: String) async throws -> SynthesisResult {
        guard let engine else {
            throw PocketTTSSwiftError.modelNotLoaded
        }

        return try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try engine.synthesize(text: text)
                continuation.resume(returning: SynthesisResult(
                    audioData: Data(result.audioData),
                    sampleRate: result.sampleRate,
                    channels: result.channels,
                    durationSeconds: result.durationSeconds
                ))
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    /// Synthesize with specific voice
    public func synthesize(text: String, voice: UInt32) async throws -> SynthesisResult {
        guard let engine else {
            throw PocketTTSSwiftError.modelNotLoaded
        }

        return try await withCheckedThrowingContinuation { continuation in
            do {
                let result = try engine.synthesizeWithVoice(text: text, voiceIndex: voice)
                continuation.resume(returning: SynthesisResult(
                    audioData: Data(result.audioData),
                    sampleRate: result.sampleRate,
                    channels: result.channels,
                    durationSeconds: result.durationSeconds
                ))
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    /// Streaming synthesis
    public func synthesizeStreaming(text: String) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            guard let engine = self.engine else {
                continuation.finish(throwing: PocketTTSSwiftError.modelNotLoaded)
                return
            }

            let handler = StreamingHandler(continuation: continuation)

            Task {
                do {
                    try engine.startStreaming(text: text, handler: handler)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Cancel ongoing synthesis
    public func cancel() {
        engine?.cancel()
    }

    /// Model version
    public var version: String {
        engine?.modelVersion() ?? "unknown"
    }

    /// Parameter count
    public var parameterCount: UInt64 {
        engine?.parameterCount() ?? 0
    }
}

// MARK: - Streaming Handler

@available(iOS 17.0, *)
private class StreamingHandler: TTSEventHandler {
    private let continuation: AsyncThrowingStream<PocketTTSSwift.AudioChunk, Error>.Continuation

    init(continuation: AsyncThrowingStream<PocketTTSSwift.AudioChunk, Error>.Continuation) {
        self.continuation = continuation
    }

    func onAudioChunk(chunk: pocket_tts_ios.AudioChunk) {
        continuation.yield(PocketTTSSwift.AudioChunk(
            audioData: Data(chunk.audioData),
            sampleRate: chunk.sampleRate,
            isFinal: chunk.isFinal
        ))
    }

    func onProgress(progress: Float) {
        // Could emit progress events if needed
    }

    func onComplete() {
        continuation.finish()
    }

    func onError(message: String) {
        continuation.finish(throwing: PocketTTSSwiftError.synthesisError(message))
    }
}

// MARK: - Errors

@available(iOS 17.0, *)
public enum PocketTTSSwiftError: Error, LocalizedError {
    case modelNotLoaded
    case synthesisError(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Pocket TTS model is not loaded"
        case .synthesisError(let message):
            return "Synthesis failed: \(message)"
        }
    }
}
