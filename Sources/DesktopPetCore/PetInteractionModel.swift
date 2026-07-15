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
    case startStandTimer
    case startSleepTimer
    case cancelStandTimer
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
        case .sitting, .standing:
            return PetTransition(posture: posture, actions: [.startSleepTimer])
        case .stretching:
            return PetTransition(posture: posture, actions: [.none])
        }
    }

    @discardableResult
    public mutating func doubleClick() -> PetTransition {
        guard posture == .sitting else {
            return PetTransition(posture: posture, actions: [.none])
        }
        posture = .standing
        return PetTransition(posture: posture, actions: [.startStandTimer, .startSleepTimer])
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
    public mutating func standTimerFired() -> PetTransition {
        guard posture == .standing else {
            return PetTransition(posture: posture, actions: [.none])
        }
        posture = .sitting
        return PetTransition(posture: posture, actions: [.none])
    }

    @discardableResult
    public mutating func sleepTimerFired() -> PetTransition {
        posture = .sleeping
        return PetTransition(posture: posture, actions: [.cancelAllTimers])
    }

    public static func gazeDirection(pointer: CGPoint, petCenter: CGPoint, deadzone: CGFloat = 8) -> Int? {
        let dx = pointer.x - petCenter.x
        let dy = pointer.y - petCenter.y
        guard hypot(dx, dy) > deadzone else { return nil }

        var degrees = atan2(dx, dy) * 180 / .pi
        if degrees < 0 { degrees += 360 }
        return Int((degrees / 22.5).rounded()) % 16
    }
}
