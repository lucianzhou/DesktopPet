import AppKit
import DesktopPetCore

@MainActor
final class PetWindowController: NSWindowController {
    private static let windowSize = NSSize(width: 268, height: 291)
    private static let sleepDelay: TimeInterval = 5 * 60
    private static let standDelay: TimeInterval = 5

    private var model = PetInteractionModel()
    private let petView: PetView
    private lazy var animator = SpriteAnimator(view: petView)
    private var sleepTimer: Timer?
    private var standTimer: Timer?
    private var gazeTimer: Timer?
    private var lastGazeDirection: Int?
    private var gazeIsActive = false
    private var sleepObserver: NSObjectProtocol?

    init(assetDirectory: URL?) throws {
        let assets = try AssetLocator.loadImages(assetDirectory: assetDirectory)
        petView = PetView(interactionImage: assets.interaction, standardImage: assets.standard)

        let panel = NSPanel(
            contentRect: NSRect(origin: .zero, size: Self.windowSize),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.hidesOnDeactivate = false
        panel.isMovableByWindowBackground = false
        panel.contentView = petView

        super.init(window: panel)

        petView.onSingleClick = { [weak self] in self?.singleClick() }
        petView.onDoubleClick = { [weak self] in self?.doubleClick() }
        installGazeTracking()
        sleepObserver = NotificationCenter.default.addObserver(
            forName: .desktopPetSleepNow,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.sleepNow() }
        }
        showSleeping()
        restoreOrPlaceWindow()
    }

    required init?(coder: NSCoder) { nil }

    func shutdown() {
        cancelTimers()
        animator.stop()
        gazeTimer?.invalidate()
        gazeTimer = nil
        if let sleepObserver { NotificationCenter.default.removeObserver(sleepObserver) }
        sleepObserver = nil
    }

    override func close() {
        persistWindowPosition()
        super.close()
    }

    private func singleClick() {
        let transition = model.singleClick()
        apply(transition)
    }

    private func doubleClick() {
        let transition = model.doubleClick()
        apply(transition)
    }

    private func apply(_ transition: PetTransition) {
        for action in transition.actions {
            switch action {
            case .playStretch:
                showStretching()
            case .startStandTimer:
                showStanding()
                restartStandTimer()
            case .startSleepTimer:
                restartSleepTimer()
            case .cancelStandTimer:
                standTimer?.invalidate()
                standTimer = nil
            case .cancelAllTimers:
                cancelTimers()
            case .none:
                break
            }
        }
    }

    private func showSleeping() {
        resetGazeTracking()
        animator.play(SpriteSequence(
            atlas: .interaction,
            row: 0,
            durations: [0.70, 0.55, 0.55, 0.65, 0.55, 0.85]
        ))
    }

    private func showStretching() {
        resetGazeTracking()
        animator.play(SpriteSequence(atlas: .interaction, row: 1, frameCount: 8, duration: 0.15, loops: false)) { [weak self] in
            guard let self else { return }
            let transition = self.model.stretchCompleted()
            self.showSitting()
            self.apply(transition)
        }
    }

    private func showSitting() {
        resetGazeTracking()
        animator.play(SpriteSequence(
            atlas: .interaction,
            row: 2,
            durations: [0.52, 0.42, 0.42, 0.50, 0.42, 0.70]
        ))
    }

    private func showStanding() {
        resetGazeTracking()
        animator.play(SpriteSequence(atlas: .standard, row: 0, frameCount: 6, duration: 0.18))
    }

    private func sleepNow() {
        _ = model.sleepTimerFired()
        cancelTimers()
        showSleeping()
    }

    private func restartSleepTimer() {
        sleepTimer?.invalidate()
        sleepTimer = Timer.scheduledTimer(withTimeInterval: Self.sleepDelay, repeats: false) { [weak self] _ in
            MainActor.assumeIsolated { self?.sleepNow() }
        }
    }

    private func restartStandTimer() {
        standTimer?.invalidate()
        standTimer = Timer.scheduledTimer(withTimeInterval: Self.standDelay, repeats: false) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                let transition = self.model.standTimerFired()
                if transition.posture == .sitting { self.showSitting() }
            }
        }
    }

    private func cancelTimers() {
        sleepTimer?.invalidate()
        sleepTimer = nil
        standTimer?.invalidate()
        standTimer = nil
    }

    private func installGazeTracking() {
        gazeTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.updateGaze() }
        }
    }

    private func updateGaze() {
        guard model.posture == .sitting, let window else { return }
        let center = CGPoint(x: window.frame.midX, y: window.frame.midY)
        let direction = PetInteractionModel.gazeDirection(
            pointer: NSEvent.mouseLocation,
            petCenter: center
        )

        guard let direction else {
            if gazeIsActive {
                gazeIsActive = false
                lastGazeDirection = nil
                animator.play(SpriteSequence(
                    atlas: .interaction,
                    row: 2,
                    durations: [0.52, 0.42, 0.42, 0.50, 0.42, 0.70]
                ))
            }
            return
        }
        guard !gazeIsActive || direction != lastGazeDirection else { return }
        gazeIsActive = true
        lastGazeDirection = direction
        if direction < 8 {
            animator.show(atlas: .interaction, row: 3, frame: direction)
        } else {
            animator.show(atlas: .interaction, row: 4, frame: direction - 8)
        }
    }

    private func resetGazeTracking() {
        gazeIsActive = false
        lastGazeDirection = nil
    }

    private func restoreOrPlaceWindow() {
        guard let window else { return }
        let defaults = UserDefaults.standard
        if defaults.object(forKey: "DesktopPetWindowX") != nil {
            let origin = NSPoint(
                x: defaults.double(forKey: "DesktopPetWindowX"),
                y: defaults.double(forKey: "DesktopPetWindowY")
            )
            window.setFrameOrigin(origin)
        } else if let screen = NSScreen.main {
            let visible = screen.visibleFrame
            window.setFrameOrigin(NSPoint(
                x: visible.maxX - Self.windowSize.width - 24,
                y: visible.minY + 32
            ))
        }
    }

    func persistWindowPosition() {
        guard let origin = window?.frame.origin else { return }
        UserDefaults.standard.set(origin.x, forKey: "DesktopPetWindowX")
        UserDefaults.standard.set(origin.y, forKey: "DesktopPetWindowY")
    }
}

private enum AssetLocator {
    struct Images {
        let interaction: NSImage
        let standard: NSImage
    }

    static func loadImages(assetDirectory: URL?) throws -> Images {
        let roots = candidateRoots(assetDirectory: assetDirectory)
        guard let assetsRoot = roots.first(where: {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("baomihua-interaction.png").path)
        }) else {
            throw AssetError.missingAssets(roots.map(\.path))
        }
        guard let interaction = NSImage(contentsOf: assetsRoot.appendingPathComponent("baomihua-interaction.png")),
              let standard = NSImage(contentsOf: assetsRoot.appendingPathComponent("baomihua-standard.png")) else {
            throw AssetError.unreadableAssets(assetsRoot.path)
        }
        return Images(interaction: interaction, standard: standard)
    }

    private static func candidateRoots(assetDirectory: URL?) -> [URL] {
        var roots: [URL] = []
        if let assetDirectory { roots.append(assetDirectory) }
        if let override = ProcessInfo.processInfo.environment["DESKTOP_PET_ASSETS"] {
            roots.append(URL(fileURLWithPath: override, isDirectory: true))
        }
        if let resourceURL = Bundle.main.resourceURL {
            roots.append(resourceURL.appendingPathComponent("Assets", isDirectory: true))
        }
        roots.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("Assets", isDirectory: true))
        return roots
    }

    enum AssetError: LocalizedError {
        case missingAssets([String])
        case unreadableAssets(String)

        var errorDescription: String? {
            switch self {
            case .missingAssets(let searched):
                return "Missing pet assets. Searched: \(searched.joined(separator: ", "))"
            case .unreadableAssets(let root):
                return "Unable to read pet assets in \(root)"
            }
        }
    }
}
