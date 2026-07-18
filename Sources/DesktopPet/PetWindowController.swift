import AppKit
import DesktopPetCore

@MainActor
final class PetWindowController: NSWindowController {
    // Exactly 75% of the previous footprint. The 192x208 source cells now
    // render close to native size instead of being enlarged by about 40%.
    private static let windowSize = NSSize(width: 201, height: 218.25)
    private static let sleepDelay: TimeInterval = 5 * 60
    private static let interactionDuration: TimeInterval = 5

    private var model = PetInteractionModel()
    private let petView: PetView
    private let interactionLayout: AssetLocator.InteractionAtlasLayout
    private let gazeLayout: AssetLocator.GazeAtlasLayout
    private lazy var animator = SpriteAnimator(view: petView)
    private var sleepTimer: Timer?
    private var gazeTimer: Timer?
    private var lastGazeDirection: Int?
    private var targetGazeDirection: Int?
    private var filteredGazeAngle: Double?
    private var lastGazeStepAt: TimeInterval = 0
    private var gazeIsActive = false
    private var sleepObserver: NSObjectProtocol?

    init(assetDirectory: URL?) throws {
        let assets = try AssetLocator.loadImages(assetDirectory: assetDirectory)
        interactionLayout = assets.interactionLayout
        gazeLayout = assets.gazeLayout
        petView = PetView(
            interactionImage: assets.interaction,
            sleepImage: assets.sleep,
            wakeImage: assets.wake,
            neutralImage: assets.neutral,
            gazeImage: assets.gaze
        )

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
            case .playInteraction:
                showInteraction()
            case .startSleepTimer:
                restartSleepTimer()
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
            atlas: .sleep,
            row: 0,
            frameCount: 1,
            duration: 4.4,
            motion: .breathing(cycle: 4.4)
        ))
    }

    private func showStretching() {
        resetGazeTracking()
        let sequence = SpriteSequence(
            atlas: .wake,
            row: 0,
            durations: [0.25, 0.22, 0.24, 0.26, 0.34, 0.24, 0.22, 0.23],
            loops: false
        )
        animator.settlePresentation(duration: 0.15) { [weak self] in
            guard let self else { return }
            self.animator.play(sequence) { [weak self] in
                guard let self else { return }
                let transition = self.model.stretchCompleted()
                self.showSitting()
                self.apply(transition)
            }
        }
    }

    private func showSitting() {
        resetGazeTracking()
        animator.show(atlas: .neutral, row: 0, frame: 0)
    }

    private func showInteraction() {
        resetGazeTracking()
        animator.play(SpriteSequence(
            frames: interactionFrames(),
            duration: Self.interactionDuration / Double(interactionLayout.frameCount),
            loops: false
        )) { [weak self] in
            guard let self, self.model.posture == .standing else { return }
            let transition = self.model.interactionCompleted()
            self.showSitting()
            self.apply(transition)
        }
    }

    private func interactionFrames() -> [SpriteFrame] {
        (0..<interactionLayout.rows).flatMap { row in
            (0..<interactionLayout.columns).map { column in
                SpriteFrame(atlas: .interaction, row: row, column: column)
            }
        }
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

    private func cancelTimers() {
        sleepTimer?.invalidate()
        sleepTimer = nil
    }

    private func installGazeTracking() {
        let timer = Timer(timeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.updateGaze() }
        }
        gazeTimer = timer
        RunLoop.main.add(timer, forMode: .common)
    }

    private func updateGaze() {
        guard model.posture == .sitting, let window else { return }
        // The neutral target is the face/eye line, not the transparent cell's
        // geometric center.  The sprite is bottom-registered, so the face is
        // roughly three quarters of the way up from the window's lower edge.
        // This keeps a pointer placed directly in front of the face on the
        // canonical front pose; a pointer below the paws still produces a
        // natural downward look.
        let center = CGPoint(
            x: window.frame.midX,
            y: window.frame.minY + window.frame.height * 0.75
        )
        let pointer = NSEvent.mouseLocation
        let dx = pointer.x - center.x
        let dy = pointer.y - center.y

        // The center of the cat is the neutral front-facing pose. Return to
        // that canonical frame when the pointer is directly in front rather
        // than leaving the last diagonal direction frozen on screen.
        guard hypot(dx, dy) > 18 else {
            if gazeIsActive {
                resetGazeTracking()
                animator.show(atlas: .neutral, row: 0, frame: 0)
            }
            return
        }

        var rawAngle = atan2(dx, dy) * 180 / .pi
        if rawAngle < 0 { rawAngle += 360 }
        if let filteredGazeAngle {
            var delta = rawAngle - filteredGazeAngle
            if delta > 180 { delta -= 360 }
            if delta < -180 { delta += 360 }
            self.filteredGazeAngle = (filteredGazeAngle + delta * 0.24 + 360)
                .truncatingRemainder(dividingBy: 360)
        } else {
            filteredGazeAngle = rawAngle
        }

        guard let filteredGazeAngle else { return }
        let directionCount = gazeLayout.frameCount
        let directionStep = 360 / Double(directionCount)
        targetGazeDirection = Int((filteredGazeAngle / directionStep).rounded()) % directionCount
        guard let targetGazeDirection else { return }

        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastGazeStepAt >= 0.05 else { return }
        let direction = lastGazeDirection.map {
            PetInteractionModel.nextGazeDirection(
                from: $0,
                toward: targetGazeDirection,
                directionCount: directionCount
            )
        } ?? targetGazeDirection
        guard !gazeIsActive || direction != lastGazeDirection else { return }
        gazeIsActive = true
        lastGazeDirection = direction
        lastGazeStepAt = now
        animator.show(
            atlas: .gaze,
            row: direction / gazeLayout.columns,
            frame: direction % gazeLayout.columns
        )
    }

    private func resetGazeTracking() {
        gazeIsActive = false
        lastGazeDirection = nil
        targetGazeDirection = nil
        filteredGazeAngle = nil
        lastGazeStepAt = 0
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
    private static let interactionFilename = "baomihua-interaction-v5.png"
    private static let interactionCellWidth = 192
    private static let interactionCellHeight = 208
    private static let interactionColumns = 8
    private static let interactionRows = 6

    struct InteractionAtlasLayout: Equatable {
        let columns: Int
        let rows: Int

        var frameCount: Int { columns * rows }
    }

    struct GazeAtlasLayout: Equatable {
        let columns: Int
        let rows: Int

        var frameCount: Int { columns * rows }
    }

    struct Images {
        let interaction: NSImage
        let interactionLayout: InteractionAtlasLayout
        let sleep: NSImage
        let wake: NSImage
        let neutral: NSImage
        let gaze: NSImage
        let gazeLayout: GazeAtlasLayout
    }

    static func loadImages(assetDirectory: URL?) throws -> Images {
        let roots = candidateRoots(assetDirectory: assetDirectory)
        guard let assetsRoot = roots.first(where: {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent(interactionFilename).path)
                && FileManager.default.fileExists(atPath: $0.appendingPathComponent("baomihua-sleep-v2.png").path)
                && FileManager.default.fileExists(atPath: $0.appendingPathComponent("baomihua-wake.png").path)
                && FileManager.default.fileExists(atPath: $0.appendingPathComponent("baomihua-neutral.png").path)
                && FileManager.default.fileExists(atPath: $0.appendingPathComponent("baomihua-gaze-v8-uniform.png").path)
        }) else {
            throw AssetError.missingAssets(roots.map(\.path))
        }
        guard let interaction = NSImage(contentsOf: assetsRoot.appendingPathComponent(interactionFilename)),
              let sleep = NSImage(contentsOf: assetsRoot.appendingPathComponent("baomihua-sleep-v2.png")),
              let wake = NSImage(contentsOf: assetsRoot.appendingPathComponent("baomihua-wake.png")),
              let neutral = NSImage(contentsOf: assetsRoot.appendingPathComponent("baomihua-neutral.png")),
              let gaze = NSImage(contentsOf: assetsRoot.appendingPathComponent("baomihua-gaze-v8-uniform.png")) else {
            throw AssetError.unreadableAssets(assetsRoot.path)
        }
        let interactionSize = pixelSize(of: interaction)
        guard let interactionLayout = interactionLayout(for: interactionSize) else {
            let actual = interactionSize.map { "\($0.width) × \($0.height)" } ?? "unavailable"
            throw AssetError.invalidInteractionAtlasDimensions(assetsRoot.path, actual)
        }
        let gazeSize = pixelSize(of: gaze)
        guard let gazeLayout = gazeLayout(for: gazeSize) else {
            let actual = gazeSize.map { "\($0.width) × \($0.height)" } ?? "unavailable"
            throw AssetError.invalidGazeAtlasDimensions(assetsRoot.path, actual)
        }
        return Images(
            interaction: interaction,
            interactionLayout: interactionLayout,
            sleep: sleep,
            wake: wake,
            neutral: neutral,
            gaze: gaze,
            gazeLayout: gazeLayout
        )
    }

    private static func interactionLayout(
        for size: (width: Int, height: Int)?
    ) -> InteractionAtlasLayout? {
        guard let size,
              size.width == interactionCellWidth * interactionColumns,
              size.height == interactionCellHeight * interactionRows else {
            return nil
        }
        return InteractionAtlasLayout(columns: interactionColumns, rows: interactionRows)
    }

    private static func gazeLayout(for size: (width: Int, height: Int)?) -> GazeAtlasLayout? {
        guard let size,
              size.width == interactionCellWidth * interactionColumns,
              size.height.isMultiple(of: interactionCellHeight) else {
            return nil
        }
        let rows = size.height / interactionCellHeight
        guard rows > 0 else { return nil }
        return GazeAtlasLayout(columns: interactionColumns, rows: rows)
    }

    private static func pixelSize(of image: NSImage) -> (width: Int, height: Int)? {
        var proposedRect = NSRect(origin: .zero, size: image.size)
        guard let cgImage = image.cgImage(
            forProposedRect: &proposedRect,
            context: nil,
            hints: nil
        ) else {
            return nil
        }
        return (cgImage.width, cgImage.height)
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
        case invalidInteractionAtlasDimensions(String, String)
        case invalidGazeAtlasDimensions(String, String)

        var errorDescription: String? {
            switch self {
            case .missingAssets(let searched):
                return "Missing pet assets (including the QA-approved baomihua-interaction-v5.png atlas). Searched: \(searched.joined(separator: ", "))"
            case .unreadableAssets(let root):
                return "Unable to read pet assets in \(root)"
            case .invalidInteractionAtlasDimensions(let root, let actual):
                return "Invalid baomihua-interaction-v5.png dimensions in \(root). Expected 1536 × 1248 pixels (8 × 6 / 48 frames of 192 × 208); found \(actual)."
            case .invalidGazeAtlasDimensions(let root, let actual):
                return "Invalid baomihua gaze atlas dimensions in \(root). Expected 8 columns of 192 × 208 cells; found \(actual)."
            }
        }
    }
}
