import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var viewModel = TTSViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Resource Monitor
                    ResourceMonitorView(monitor: viewModel.resourceMonitor)

                    Divider()

                    // Text Input
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Text to Synthesize")
                            .font(.headline)

                        TextEditor(text: $viewModel.inputText)
                            .frame(minHeight: 100)
                            .padding(8)
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                    }

                    // Voice Selection
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Voice")
                            .font(.headline)

                        Picker("Voice", selection: $viewModel.selectedVoice) {
                            ForEach(TTSVoice.allCases) { voice in
                                Text(voice.displayName).tag(voice)
                            }
                        }
                        .pickerStyle(.segmented)
                    }

                    // Status
                    if let status = viewModel.status {
                        StatusView(status: status)
                    }

                    // Timing Info
                    if let timing = viewModel.lastTiming {
                        TimingView(timing: timing)
                    }

                    // Controls
                    HStack(spacing: 16) {
                        Button(action: viewModel.synthesize) {
                            Label("Synthesize", systemImage: "waveform")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isSynthesizing || viewModel.inputText.isEmpty)

                        Button(action: {
                            if viewModel.isPlaying {
                                viewModel.stopAudio()
                            } else {
                                viewModel.playAudio()
                            }
                        }) {
                            Label(viewModel.isPlaying ? "Stop" : "Play",
                                  systemImage: viewModel.isPlaying ? "stop.fill" : "play.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .tint(viewModel.isPlaying ? .red : .accentColor)
                        .disabled(viewModel.audioData == nil)
                    }

                    // Audio Waveform Preview
                    if let samples = viewModel.audioSamples {
                        WaveformView(samples: samples)
                            .frame(height: 100)
                    }

                    Spacer()
                }
                .padding()
            }
            .navigationTitle("Pocket TTS Demo")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: viewModel.loadModel) {
                        Label("Reload Model", systemImage: "arrow.clockwise")
                    }
                    .disabled(viewModel.isLoading)
                }
            }
        }
        .onAppear {
            viewModel.loadModel()
        }
    }
}

// MARK: - Resource Monitor View

struct ResourceMonitorView: View {
    @ObservedObject var monitor: ResourceMonitor

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("System Resources")
                    .font(.headline)
                Spacer()
                Circle()
                    .fill(monitor.thermalState.color)
                    .frame(width: 12, height: 12)
                Text(monitor.thermalState.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            HStack(spacing: 20) {
                MetricCard(
                    title: "Memory",
                    value: monitor.memoryUsageMB,
                    unit: "MB",
                    icon: "memorychip"
                )

                MetricCard(
                    title: "CPU",
                    value: monitor.cpuUsage,
                    unit: "%",
                    icon: "cpu"
                )

                MetricCard(
                    title: "App Size",
                    value: monitor.appMemoryMB,
                    unit: "MB",
                    icon: "app.badge"
                )
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct MetricCard: View {
    let title: String
    let value: Double
    let unit: String
    let icon: String

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.accentColor)

            Text(String(format: "%.1f", value))
                .font(.title3)
                .fontWeight(.semibold)

            Text(unit)
                .font(.caption2)
                .foregroundColor(.secondary)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Status View

struct StatusView: View {
    let status: TTSStatus

    var body: some View {
        HStack {
            if status.isLoading {
                ProgressView()
                    .scaleEffect(0.8)
            }

            Image(systemName: status.icon)
                .foregroundColor(status.color)

            Text(status.message)
                .font(.subheadline)

            Spacer()
        }
        .padding()
        .background(status.backgroundColor)
        .cornerRadius(8)
    }
}

// MARK: - Timing View

struct TimingView: View {
    let timing: SynthesisTiming

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Performance")
                .font(.headline)

            HStack {
                TimingItem(label: "Synthesis", value: timing.synthesisTime, unit: "s")
                TimingItem(label: "Audio", value: timing.audioDuration, unit: "s")
                TimingItem(label: "RTF", value: timing.realtimeFactor, unit: "x")
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

struct TimingItem: View {
    let label: String
    let value: Double
    let unit: String

    var body: some View {
        VStack {
            Text(String(format: "%.2f%@", value, unit))
                .font(.title3)
                .fontWeight(.medium)
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Waveform View

struct WaveformView: View {
    let samples: [Float]

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height
            let midY = height / 2
            let step = max(1, samples.count / Int(width))

            Path { path in
                path.move(to: CGPoint(x: 0, y: midY))

                for i in stride(from: 0, to: samples.count, by: step) {
                    let x = CGFloat(i) / CGFloat(samples.count) * width
                    let sample = samples[i]
                    let y = midY - CGFloat(sample) * midY * 0.9
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
            .stroke(Color.accentColor, lineWidth: 1)
        }
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

#Preview {
    ContentView()
}
