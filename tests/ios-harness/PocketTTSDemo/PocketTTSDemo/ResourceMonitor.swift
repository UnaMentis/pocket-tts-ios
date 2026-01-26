import Foundation
import SwiftUI
import Combine

class ResourceMonitor: ObservableObject {
    @Published var memoryUsageMB: Double = 0
    @Published var cpuUsage: Double = 0
    @Published var appMemoryMB: Double = 0
    @Published var thermalState: ThermalState = .nominal

    @Published var peakMemoryMB: Double = 0
    @Published var peakCPU: Double = 0

    private var timer: Timer?
    private var synthesisStartMemory: Double = 0

    func startMonitoring() {
        // Initial update
        updateMetrics()

        // Start periodic monitoring
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.updateMetrics()
        }

        // Monitor thermal state changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(thermalStateChanged),
            name: ProcessInfo.thermalStateDidChangeNotification,
            object: nil
        )
    }

    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }

    func markSynthesisStart() {
        synthesisStartMemory = appMemoryMB
        peakMemoryMB = appMemoryMB
        peakCPU = 0
    }

    func markSynthesisEnd() {
        // Peak values are tracked during monitoring
    }

    private func updateMetrics() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            self.memoryUsageMB = self.getSystemMemoryUsage()
            self.cpuUsage = self.getCPUUsage()
            self.appMemoryMB = self.getAppMemoryUsage()
            self.thermalState = ThermalState(from: ProcessInfo.processInfo.thermalState)

            // Track peaks
            if self.appMemoryMB > self.peakMemoryMB {
                self.peakMemoryMB = self.appMemoryMB
            }
            if self.cpuUsage > self.peakCPU {
                self.peakCPU = self.cpuUsage
            }
        }
    }

    @objc private func thermalStateChanged() {
        DispatchQueue.main.async { [weak self] in
            self?.thermalState = ThermalState(from: ProcessInfo.processInfo.thermalState)
        }
    }

    private func getAppMemoryUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / 1024.0 / 1024.0
    }

    private func getSystemMemoryUsage() -> Double {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)

        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }

        guard result == KERN_SUCCESS else { return 0 }

        let pageSize = UInt64(vm_kernel_page_size)
        let activeMemory = UInt64(stats.active_count) * pageSize
        let wiredMemory = UInt64(stats.wire_count) * pageSize
        let compressedMemory = UInt64(stats.compressor_page_count) * pageSize

        let usedMemory = activeMemory + wiredMemory + compressedMemory
        return Double(usedMemory) / 1024.0 / 1024.0
    }

    private func getCPUUsage() -> Double {
        var threadsList: thread_act_array_t?
        var threadsCount = mach_msg_type_number_t()

        let result = task_threads(mach_task_self_, &threadsList, &threadsCount)
        guard result == KERN_SUCCESS, let threads = threadsList else { return 0 }

        var totalUsage: Double = 0

        for i in 0..<Int(threadsCount) {
            var info = thread_basic_info()
            var count = mach_msg_type_number_t(THREAD_INFO_MAX)

            let infoResult = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    thread_info(threads[i], thread_flavor_t(THREAD_BASIC_INFO), $0, &count)
                }
            }

            guard infoResult == KERN_SUCCESS else { continue }

            if info.flags & TH_FLAGS_IDLE == 0 {
                totalUsage += Double(info.cpu_usage) / Double(TH_USAGE_SCALE) * 100.0
            }
        }

        // Deallocate thread list
        let size = vm_size_t(Int(threadsCount) * MemoryLayout<thread_t>.stride)
        vm_deallocate(mach_task_self_, vm_address_t(bitPattern: threads), size)

        return min(totalUsage, 100.0)
    }

    deinit {
        stopMonitoring()
    }
}

// MARK: - Thermal State

enum ThermalState: CustomStringConvertible {
    case nominal
    case fair
    case serious
    case critical

    init(from state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal: self = .nominal
        case .fair: self = .fair
        case .serious: self = .serious
        case .critical: self = .critical
        @unknown default: self = .nominal
        }
    }

    var description: String {
        switch self {
        case .nominal: return "Normal"
        case .fair: return "Warm"
        case .serious: return "Hot"
        case .critical: return "Critical"
        }
    }

    var color: Color {
        switch self {
        case .nominal: return .green
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        }
    }
}
