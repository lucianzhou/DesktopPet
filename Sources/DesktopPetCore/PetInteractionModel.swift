import Foundation

public enum PetPosture: String, Equatable, Sendable {
    case sleeping
    case stretching
    case sitting
    case standing
}

public enum PetAction: Equatable, Sendable {
    case none
    case playStretch
    case playInteraction
    case startSleepTimer
    case cancelAllTimers
}

public struct PetTransition: Equatable, Sendable {
    public let posture: PetPosture
    public let actions: [PetAction]

    public init(posture: PetPosture, actions: [PetAction]) {
        self.posture = posture
        self.actions = actions
    }
}

public struct PetInteractionModel: Sendable {
    public private(set) var posture: PetPosture = .sleeping

    public init() {}

    @discardableResult
    public mutating func singleClick() -> PetTransition {
        switch posture {
        case .sleeping:
            posture = .stretching
            return PetTransition(posture: posture, actions: [.cancelAllTimers, .playStretch])
        case .sitting:
            return PetTransition(posture: posture, actions: [.startSleepTimer])
        case .stretching, .standing:
            return PetTransition(posture: posture, actions: [.none])
        }
    }

    @discardableResult
    public mutating func doubleClick() -> PetTransition {
        guard posture == .sitting else {
            return PetTransition(posture: posture, actions: [.none])
        }
        posture = .standing
        // The controller owns one finite, 48-frame interaction sequence.  Do
        // not start a second timer here: its completion is the only event
        // allowed to return the pet to sitting.
        return PetTransition(posture: posture, actions: [.cancelAllTimers, .playInteraction])
    }

    @discardableResult
    public mutating func stretchCompleted() -> PetTransition {
        guard posture == .stretching else {
            return PetTransition(posture: posture, actions: [.none])
        }
        posture = .sitting
        return PetTransition(posture: posture, actions: [.startSleepTimer])
    }

    @discardableResult
    public mutating func interactionCompleted() -> PetTransition {
        guard posture == .standing else {
            return PetTransition(posture: posture, actions: [.none])
        }
        posture = .sitting
        return PetTransition(posture: posture, actions: [.startSleepTimer])
    }

    @discardableResult
    public mutating func sleepTimerFired() -> PetTransition {
        posture = .sleeping
        return PetTransition(posture: posture, actions: [.cancelAllTimers])
    }

    public static func gazeDirection(
        pointer: CGPoint,
        petCenter: CGPoint,
        deadzone: CGFloat = 8,
        directionCount: Int = 16
    ) -> Int? {
        precondition(directionCount > 1)
        let dx = pointer.x - petCenter.x
        let dy = pointer.y - petCenter.y
        guard hypot(dx, dy) > deadzone else { return nil }

        var degrees = atan2(dx, dy) * 180 / .pi
        if degrees < 0 { degrees += 360 }
        let step = 360 / Double(directionCount)
        return Int((degrees / step).rounded()) % directionCount
    }

    public static func nextGazeDirection(from current: Int, toward target: Int, directionCount: Int = 16) -> Int {
        precondition(directionCount > 1)
        let normalizedCurrent = ((current % directionCount) + directionCount) % directionCount
        let normalizedTarget = ((target % directionCount) + directionCount) % directionCount
        guard normalizedCurrent != normalizedTarget else { return normalizedCurrent }

        let clockwise = (normalizedTarget - normalizedCurrent + directionCount) % directionCount
        let counterclockwise = (normalizedCurrent - normalizedTarget + directionCount) % directionCount
        return clockwise <= counterclockwise
            ? (normalizedCurrent + 1) % directionCount
            : (normalizedCurrent + directionCount - 1) % directionCount
    }

    public static func breathingAmplitude(elapsed: TimeInterval, cycle: TimeInterval) -> Double {
        precondition(cycle.isFinite && cycle > 0)
        let progress = (elapsed / cycle).truncatingRemainder(dividingBy: 1)
        if progress < 0.42 {
            return smootherStep(progress / 0.42)
        }
        if progress < 0.48 {
            return 1
        }
        return 1 - smootherStep((progress - 0.48) / 0.52)
    }

    private static func smootherStep(_ value: Double) -> Double {
        let x = min(max(value, 0), 1)
        return x * x * x * (x * (x * 6 - 15) + 10)
    }
}
