import AppKit
import CoreVideo

/// Bridges Core Video's display-link thread to the main-actor animator.
///
/// A display link can continue producing callbacks while the main run loop is
/// busy. Coalescing them keeps that backlog to one pending main-actor update;
/// `SpriteAnimator` then advances its existing monotonic timeline using the
/// current uptime, so no elapsed time is lost.
private final class DisplayLinkCallbackContext: @unchecked Sendable {
    private let lock = NSLock()
    private weak var animator: SpriteAnimator?
    private var tickScheduled = false

    init(animator: SpriteAnimator) {
        self.animator = animator
    }

    func scheduleTick() {
        lock.lock()
        guard !tickScheduled else {
            lock.unlock()
            return
        }
        tickScheduled = true
        lock.unlock()

        Task { @MainActor [weak self] in
            self?.deliverTick()
        }
    }

    @MainActor
    private func deliverTick() {
        lock.lock()
        tickScheduled = false
        let animator = animator
        lock.unlock()
        animator?.tick()
    }
}

private func desktopPetDisplayLinkCallback(
    _: CVDisplayLink,
    _: UnsafePointer<CVTimeStamp>,
    _: UnsafePointer<CVTimeStamp>,
    _: CVOptionFlags,
    _: UnsafeMutablePointer<CVOptionFlags>,
    context: UnsafeMutableRawPointer?
) -> CVReturn {
    guard let context else { return kCVReturnError }
    let callbackContext = Unmanaged<DisplayLinkCallbackContext>
        .fromOpaque(context)
        .takeUnretainedValue()
    callbackContext.scheduleTick()
    return kCVReturnSuccess
}

enum SpriteAtlas: Equatable {
    case sleep
    case wake
    case neutral
    case gaze
    case interaction
}

struct SpriteFrame: Equatable {
    let atlas: SpriteAtlas
    let row: Int
    let column: Int
}

enum SpriteMotion {
    case none
    case breathing(cycle: TimeInterval)
}

struct SpriteSequence {
    let frames: [SpriteFrame]
    let durations: [TimeInterval]
    let loops: Bool
    let motion: SpriteMotion

    init(
        atlas: SpriteAtlas,
        row: Int,
        frameCount: Int,
        duration: TimeInterval,
        loops: Bool = true,
        motion: SpriteMotion = .none
    ) {
        frames = (0..<frameCount).map { SpriteFrame(atlas: atlas, row: row, column: $0) }
        durations = Array(repeating: duration, count: frameCount)
        self.loops = loops
        self.motion = motion
    }

    init(
        atlas: SpriteAtlas,
        row: Int,
        durations: [TimeInterval],
        loops: Bool = true,
        frameIndices: [Int]? = nil,
        motion: SpriteMotion = .none
    ) {
        let indices = frameIndices ?? Array(0..<durations.count)
        frames = indices.map { SpriteFrame(atlas: atlas, row: row, column: $0) }
        self.durations = durations
        self.loops = loops
        self.motion = motion
    }

    init(
        frames: [SpriteFrame],
        duration: TimeInterval,
        loops: Bool = true,
        motion: SpriteMotion = .none
    ) {
        self.frames = frames
        durations = Array(repeating: duration, count: frames.count)
        self.loops = loops
        self.motion = motion
    }

    init(
        frames: [SpriteFrame],
        durations: [TimeInterval],
        loops: Bool = true,
        motion: SpriteMotion = .none
    ) {
        precondition(frames.count == durations.count)
        self.frames = frames
        self.durations = durations
        self.loops = loops
        self.motion = motion
    }
}

@MainActor
final class SpriteAnimator {
    private weak var view: PetView?
    private var displayLink: CVDisplayLink?
    private lazy var displayLinkContext = DisplayLinkCallbackContext(animator: self)
    private var sequence: SpriteSequence?
    private var frame = 0
    private var completion: (() -> Void)?
    private var frameStartedAt: TimeInterval = 0
    private var sequenceStartedAt: TimeInterval = 0

    init(view: PetView) {
        self.view = view
    }

    func play(_ sequence: SpriteSequence, completion: (() -> Void)? = nil) {
        precondition(!sequence.frames.isEmpty)
        precondition(sequence.frames.count == sequence.durations.count)
        precondition(sequence.durations.allSatisfy { $0.isFinite && $0 > 0 })
        stopDisplayLink()
        self.sequence = sequence
        self.completion = completion
        frame = 0
        let now = ProcessInfo.processInfo.systemUptime
        frameStartedAt = now
        sequenceStartedAt = now
        view?.resetPresentation()
        renderCurrentFrame()
        startDisplayLink()
    }

    func show(atlas: SpriteAtlas, row: Int, frame: Int) {
        stopDisplayLink()
        sequence = nil
        completion = nil
        view?.resetPresentation()
        view?.setFrame(atlas: atlas, row: row, column: frame)
    }

    func stop() {
        stopDisplayLink()
        sequence = nil
        completion = nil
        view?.resetPresentation()
    }

    private func startDisplayLink() {
        if displayLink == nil {
            var createdLink: CVDisplayLink?
            let createResult = CVDisplayLinkCreateWithActiveCGDisplays(&createdLink)
            precondition(
                createResult == kCVReturnSuccess && createdLink != nil,
                "Unable to create a display-synchronised animation clock"
            )
            displayLink = createdLink
            let callbackResult = CVDisplayLinkSetOutputCallback(
                createdLink!,
                desktopPetDisplayLinkCallback,
                Unmanaged.passUnretained(displayLinkContext).toOpaque()
            )
            precondition(
                callbackResult == kCVReturnSuccess,
                "Unable to configure the display-synchronised animation clock"
            )
        }

        guard let displayLink else { return }
        let startResult = CVDisplayLinkStart(displayLink)
        precondition(
            startResult == kCVReturnSuccess,
            "Unable to start the display-synchronised animation clock"
        )
    }

    private func stopDisplayLink() {
        guard let displayLink else { return }
        CVDisplayLinkStop(displayLink)
    }

    fileprivate func tick() {
        guard let sequence else { return }
        let now = ProcessInfo.processInfo.systemUptime
        updateMotion(sequence.motion, elapsed: now - sequenceStartedAt)

        // Advance against the monotonic timeline instead of resetting the
        // frame clock to `now`. A busy main run loop can delay a timer tick;
        // carrying the original deadline forward prevents that delay from
        // accumulating into visibly slower, uneven animation.
        var advanced = false
        var catchUpCount = 0
        while now - frameStartedAt >= sequence.durations[frame] {
            frameStartedAt += sequence.durations[frame]
            catchUpCount += 1

            if frame + 1 < sequence.frames.count {
                frame += 1
                advanced = true
            } else if sequence.loops {
                frame = 0
                advanced = true
            } else {
                finishSequence()
                return
            }

            // A malformed sequence must not monopolize the main thread.
            if catchUpCount > sequence.frames.count * 2 { break }
        }
        if advanced {
            renderCurrentFrame()
        }
    }

    private func finishSequence() {
        stopDisplayLink()
        sequence = nil
        view?.resetPresentation()
        let callback = completion
        completion = nil
        callback?()
    }

    private func renderCurrentFrame() {
        guard let sequence else { return }
        let current = sequence.frames[frame]
        view?.setFrame(atlas: current.atlas, row: current.row, column: current.column)
    }

    private func updateMotion(_ motion: SpriteMotion, elapsed: TimeInterval) {
        switch motion {
        case .none:
            break
        case .breathing(let cycle):
            let phase = (1 - cos(2 * Double.pi * elapsed / cycle)) / 2
            view?.setPresentation(
                scaleX: 1 + 0.004 * phase,
                scaleY: 1 + 0.012 * phase
            )
        }
    }
}
