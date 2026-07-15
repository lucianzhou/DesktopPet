import AppKit

@MainActor
final class PetView: NSView {
    static let cellSize = CGSize(width: 192, height: 208)

    var onSingleClick: (() -> Void)?
    var onDoubleClick: (() -> Void)?

    private let interactionImage: NSImage
    private let standardImage: NSImage
    private var atlas: SpriteAtlas = .interaction
    private var row = 0
    private var column = 0
    private var pendingSingleClick: Timer?
    private var dragStart: NSPoint?
    private var dragged = false

    init(interactionImage: NSImage, standardImage: NSImage) {
        self.interactionImage = interactionImage
        self.standardImage = standardImage
        super.init(frame: .zero)
        wantsLayer = true
        layer?.backgroundColor = NSColor.clear.cgColor
    }

    required init?(coder: NSCoder) { nil }

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    func setFrame(atlas: SpriteAtlas, row: Int, column: Int) {
        self.atlas = atlas
        self.row = row
        self.column = column
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        let image = atlas == .interaction ? interactionImage : standardImage
        let sourceY = image.size.height - CGFloat(row + 1) * Self.cellSize.height
        let source = NSRect(
            x: CGFloat(column) * Self.cellSize.width,
            y: sourceY,
            width: Self.cellSize.width,
            height: Self.cellSize.height
        )
        image.draw(
            in: bounds,
            from: source,
            operation: .sourceOver,
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
