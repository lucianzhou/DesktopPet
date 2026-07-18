import AppKit
import QuartzCore

@MainActor
final class PetView: NSView {
    static let cellSize = CGSize(width: 192, height: 208)
    // gaze-v8 retains the preferred original gaze poses and is registered at
    // a 136px body width.  The neutral and interaction masters are 145px
    // wide.  Draw the latter pair at this common isotropic factor so a
    // sitting/gaze/double-click transition does not make Baomihua grow.
    // 0.948 is sqrt(mean(gaze-v8 visible area) / neutral visible area).
    private static let seatedRegistrationScale: CGFloat = 0.948

    private struct DisplayFrame: Equatable {
        let atlas: SpriteAtlas
        let row: Int
        let column: Int
    }

    var onSingleClick: (() -> Void)?
    var onDoubleClick: (() -> Void)?

    private let interactionImage: NSImage
    private let sleepImage: NSImage
    private let wakeImage: NSImage
    private let neutralImage: NSImage
    private let gazeImage: NSImage
    // Keep decoded Core Graphics representations alive for the lifetime of
    // the pet. NSImage.draw otherwise may lazily decompress a PNG on the
    // first frame that references it, which can show up as a hitch during a
    // state transition. Rendering still goes through NSImage so AppKit keeps
    // its established flipped-coordinate and interpolation behavior.
    private let decodedInteractionImage: CGImage
    private let decodedSleepImage: CGImage
    private let decodedWakeImage: CGImage
    private let decodedNeutralImage: CGImage
    private let decodedGazeImage: CGImage
    private var currentFrame = DisplayFrame(atlas: .interaction, row: 0, column: 0)
    private var presentationScaleX: CGFloat = 1
    private var presentationScaleY: CGFloat = 1
    private var pendingSingleClick: Timer?
    private var dragStart: NSPoint?
    private var dragged = false

    init(
        interactionImage: NSImage,
        sleepImage: NSImage,
        wakeImage: NSImage,
        neutralImage: NSImage,
        gazeImage: NSImage
    ) {
        self.interactionImage = interactionImage
        self.sleepImage = sleepImage
        self.wakeImage = wakeImage
        self.neutralImage = neutralImage
        self.gazeImage = gazeImage
        self.decodedInteractionImage = Self.decode(interactionImage)
        self.decodedSleepImage = Self.decode(sleepImage)
        self.decodedWakeImage = Self.decode(wakeImage)
        self.decodedNeutralImage = Self.decode(neutralImage)
        self.decodedGazeImage = Self.decode(gazeImage)
        super.init(frame: .zero)
        self.interactionImage.cacheMode = .always
        self.sleepImage.cacheMode = .always
        self.wakeImage.cacheMode = .always
        self.neutralImage.cacheMode = .always
        self.gazeImage.cacheMode = .always
        wantsLayer = true
        if let layer {
            layer.backgroundColor = NSColor.clear.cgColor
            // The view is still zero-sized during init.  Deferring the anchor
            // configuration until layout prevents a zero-bounds anchor-point
            // change from permanently shifting the layer when AppKit assigns
            // its real frame.
            configureBottomAnchoredPresentationLayer(layer)
        }
    }

    required init?(coder: NSCoder) { nil }

    override func layout() {
        super.layout()
        guard let layer, configureBottomAnchoredPresentationLayer(layer) else { return }

        // A sleep sequence can begin before the panel has acquired its final
        // size. Apply the latest sampled transform only after the real bottom
        // anchor has been installed.
        applyPresentationTransform()
    }

    private static func decode(_ image: NSImage) -> CGImage {
        var proposedRect = NSRect(origin: .zero, size: image.size)
        guard let decoded = image.cgImage(
            forProposedRect: &proposedRect,
            context: nil,
            hints: nil
        ) else {
            preconditionFailure("Unable to decode pet sprite atlas")
        }
        return decoded
    }

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    func setFrame(atlas: SpriteAtlas, row: Int, column: Int) {
        let next = DisplayFrame(atlas: atlas, row: row, column: column)
        guard next != currentFrame else { return }
        currentFrame = next
        needsDisplay = true
    }

    func setPresentation(scaleX: CGFloat, scaleY: CGFloat) {
        guard presentationScaleX != scaleX || presentationScaleY != scaleY else { return }
        presentationScaleX = scaleX
        presentationScaleY = scaleY
        applyPresentationTransform()
    }

    func resetPresentation() {
        guard presentationScaleX != 1 || presentationScaleY != 1 else { return }
        presentationScaleX = 1
        presentationScaleY = 1
        applyPresentationTransform()
    }

    func resetPresentation(animatedOver duration: TimeInterval, completion: @escaping () -> Void) {
        guard presentationScaleX != 1 || presentationScaleY != 1,
              duration > 0,
              let layer,
              configureBottomAnchoredPresentationLayer(layer) else {
            resetPresentation()
            completion()
            return
        }

        presentationScaleX = 1
        presentationScaleY = 1
        CATransaction.begin()
        CATransaction.setDisableActions(false)
        CATransaction.setAnimationDuration(duration)
        CATransaction.setAnimationTimingFunction(CAMediaTimingFunction(name: .easeInEaseOut))
        CATransaction.setCompletionBlock(completion)
        layer.transform = CATransform3DIdentity
        CATransaction.commit()
    }

    /// The breathing loop updates at display-link cadence.  Scaling the
    /// already-rendered backing layer keeps those updates on Core Animation's
    /// compositing path instead of asking AppKit to redraw and re-decode the
    /// sprite cell sixty times per second.  The anchor is the visual bottom
    /// of the view in either possible layer coordinate system, matching the
    /// bottom-registered destination rect used by `draw(_:)`.
    @discardableResult
    private func configureBottomAnchoredPresentationLayer(_ layer: CALayer) -> Bool {
        // During init AppKit has not assigned the content view's actual size.
        // Changing anchorPoint against a zero-sized bounds cannot compensate
        // layer.position, which makes a later scale drift upward instead of
        // keeping the paws fixed. Wait for layout(), where the compensation is
        // expressed in real panel points.
        guard !layer.bounds.isEmpty else { return false }

        // Sprite cells keep four transparent pixels beneath the registered
        // y=204 contact line. Anchor the transform to that real contact line,
        // not to the 208px cell boundary, so breathing never lifts the cat.
        let baselineRatio = CGFloat(204) / Self.cellSize.height
        let bottomAnchor = CGPoint(
            x: 0.5,
            y: layer.isGeometryFlipped ? baselineRatio : 1 - baselineRatio
        )
        guard layer.anchorPoint != bottomAnchor else { return true }

        // Changing a CALayer's anchor point otherwise changes its frame.  At
        // this point the transform is still identity; compensate its position
        // so the panel and its NSView event geometry remain exactly where
        // AppKit laid them out.
        let previousAnchor = layer.anchorPoint
        let bounds = layer.bounds
        let position = layer.position
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        layer.anchorPoint = bottomAnchor
        layer.position = CGPoint(
            x: position.x + (bottomAnchor.x - previousAnchor.x) * bounds.width,
            y: position.y + (bottomAnchor.y - previousAnchor.y) * bounds.height
        )
        CATransaction.commit()
        return true
    }

    private func applyPresentationTransform() {
        guard let layer else { return }
        guard configureBottomAnchoredPresentationLayer(layer) else { return }

        // Disable implicit animation: the display link supplies the smooth
        // samples, and Core Animation should present each sampled transform
        // directly rather than enqueueing an additional interpolation.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        layer.transform = CATransform3DMakeScale(presentationScaleX, presentationScaleY, 1)
        CATransaction.commit()
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        NSGraphicsContext.current?.cgContext.clear(dirtyRect)
        draw(currentFrame)
    }

    private func draw(_ frame: DisplayFrame) {
        let image: NSImage
        switch frame.atlas {
        case .sleep: image = sleepImage
        case .wake: image = wakeImage
        case .neutral: image = neutralImage
        case .gaze: image = gazeImage
        case .interaction: image = interactionImage
        }
        let sourceY = image.size.height - CGFloat(frame.row + 1) * Self.cellSize.height
        let source = NSRect(
            x: CGFloat(frame.column) * Self.cellSize.width,
            y: sourceY,
            width: Self.cellSize.width,
            height: Self.cellSize.height
        )
        let atlasScale: CGFloat
        switch frame.atlas {
        case .neutral, .interaction:
            atlasScale = Self.seatedRegistrationScale
        case .sleep, .wake, .gaze:
            atlasScale = 1
        }
        // Presentation-scale breathing is deliberately not part of this draw
        // pass.  It is applied to the backing CALayer above, so a 60 Hz sleep
        // cycle does not invalidate this view.  Sprite-frame changes still
        // take this normal single-frame `.copy` rendering path.
        let width = bounds.width * atlasScale
        let height = bounds.height * atlasScale
        let destination = NSRect(
            x: bounds.midX - width / 2,
            y: bounds.maxY - height,
            width: width,
            height: height
        )
        image.draw(
            in: destination,
            from: source,
            operation: .copy,
            fraction: 1,
            respectFlipped: true,
            hints: [.interpolation: NSImageInterpolation.high]
        )
    }

    override func mouseDown(with event: NSEvent) {
        dragStart = convert(event.locationInWindow, from: nil)
        dragged = false
    }

    override func mouseDragged(with event: NSEvent) {
        guard let dragStart else { return }
        let current = convert(event.locationInWindow, from: nil)
        if hypot(current.x - dragStart.x, current.y - dragStart.y) > 4 {
            dragged = true
            pendingSingleClick?.invalidate()
            pendingSingleClick = nil
            window?.performDrag(with: event)
        }
    }

    override func mouseUp(with event: NSEvent) {
        defer { dragStart = nil }
        guard !dragged else { return }

        if event.clickCount >= 2 {
            pendingSingleClick?.invalidate()
            pendingSingleClick = nil
            onDoubleClick?()
        } else {
            pendingSingleClick?.invalidate()
            pendingSingleClick = Timer.scheduledTimer(withTimeInterval: 0.28, repeats: false) { [weak self] _ in
                MainActor.assumeIsolated {
                    self?.pendingSingleClick = nil
                    self?.onSingleClick?()
                }
            }
        }
    }

    override func rightMouseDown(with event: NSEvent) {
        let menu = NSMenu()
        menu.addItem(withTitle: "让爆米花睡觉", action: #selector(sleepNow), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "退出 DesktopPet", action: #selector(quitApp), keyEquivalent: "q")
        NSMenu.popUpContextMenu(menu, with: event, for: self)
    }

    @objc private func sleepNow() {
        NotificationCenter.default.post(name: .desktopPetSleepNow, object: nil)
    }

    @objc private func quitApp() {
        NSApplication.shared.terminate(nil)
    }
}

extension Notification.Name {
    static let desktopPetSleepNow = Notification.Name("DesktopPetSleepNow")
}
