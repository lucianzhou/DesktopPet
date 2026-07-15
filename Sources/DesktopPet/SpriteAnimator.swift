import AppKit

enum SpriteAtlas {
    case interaction
    case standard
}

struct SpriteSequence {
    let atlas: SpriteAtlas
    let row: Int
    let frameCount: Int
    let durations: [TimeInterval]
    let loops: Bool

    init(atlas: SpriteAtlas, row: Int, frameCount: Int, duration: TimeInterval, loops: Bool = true) {
        self.atlas = atlas
        self.row = row
        self.frameCount = frameCount
        self.durations = Array(repeating: duration, count: frameCount)
        self.loops = loops
    }

    init(atlas: SpriteAtlas, row: Int, durations: [TimeInterval], loops: Bool = true) {
        self.atlas = atlas
        self.row = row
        self.frameCount = durations.count
        self.durations = durations
        self.loops = loops
    }
}

@MainActor
final class SpriteAnimator {
    private weak var view: PetView?
    private var timer: Timer?
    private var sequence: SpriteSequence?
    private var frame = 0
    private var completion: (() -> Void)?

    init(view: PetView) {
        self.view = view
    }

    func play(_ sequence: SpriteSequence, completion: (() -> Void)? = nil) {
        timer?.invalidate()
        self.sequence = sequence
        self.completion = completion
        frame = 0
        render()
        scheduleNextFrame()
    }

    func show(atlas: SpriteAtlas, row: Int, frame: Int) {
        timer?.invalidate()
        sequence = nil
        completion = nil
        view?.setFrame(atlas: atlas, row: row, column: frame)
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    private func render() {
        guard let sequence else { return }
        view?.setFrame(atlas: sequence.atlas, row: sequence.row, column: frame)
    }

    private func scheduleNextFrame() {
        guard let sequence else { return }
        let delay = sequence.durations[min(frame, sequence.durations.count - 1)]
        timer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.advance()
            }
        }
    }

    private func advance() {
        guard let sequence else { return }
        if frame + 1 < sequence.frameCount {
            frame += 1
            render()
            scheduleNextFrame()
        } else if sequence.loops {
            frame = 0
            render()
            scheduleNextFrame()
        } else {
            timer = nil
            let callback = completion
            completion = nil
            callback?()
        }
    }
}
