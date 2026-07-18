import CoreGraphics
import DesktopPetCore

func check(_ condition: @autoclosure () -> Bool, _ message: String) {
    guard condition() else {
        fatalError("Core check failed: \(message)")
    }
}

var model = PetInteractionModel()
check(model.posture == .sleeping, "initial posture")
check(model.singleClick().posture == .stretching, "single-click wake")
check(model.stretchCompleted().posture == .sitting, "stretch completion")
let interaction = model.doubleClick()
check(interaction.posture == .standing, "double-click starts interaction")
check(interaction.actions == [.cancelAllTimers, .playInteraction], "interaction has one completion path")
check(model.singleClick().actions == [.none], "ignore clicks during interaction")
let interactionCompletion = model.interactionCompleted()
check(interactionCompletion.posture == .sitting, "interaction returns to sitting")
check(interactionCompletion.actions == [.startSleepTimer], "restart idle timer after interaction")
check(model.interactionCompleted().actions == [.none], "interaction completion is one-shot")
check(model.sleepTimerFired().posture == .sleeping, "sleep timeout")

let center = CGPoint(x: 100, y: 100)
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 100, y: 200), petCenter: center) == 0, "up gaze")
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 200, y: 100), petCenter: center) == 4, "right gaze")
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 100, y: 0), petCenter: center) == 8, "down gaze")
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 0, y: 100), petCenter: center) == 12, "left gaze")
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 103, y: 103), petCenter: center) == nil, "deadzone")
check(PetInteractionModel.nextGazeDirection(from: 15, toward: 1) == 0, "clockwise wrap step")
check(PetInteractionModel.nextGazeDirection(from: 1, toward: 15) == 0, "counterclockwise wrap step")
check(
    PetInteractionModel.gazeDirection(
        pointer: CGPoint(x: 200, y: 100),
        petCenter: center,
        directionCount: 32
    ) == 8,
    "32-direction right gaze"
)
check(
    PetInteractionModel.nextGazeDirection(from: 31, toward: 1, directionCount: 32) == 0,
    "32-direction clockwise wrap step"
)
check(PetInteractionModel.breathingAmplitude(elapsed: 0, cycle: 4.4) == 0, "breathing starts at rest")
check(
    abs(PetInteractionModel.breathingAmplitude(elapsed: 4.4 * 0.42, cycle: 4.4) - 1) < 0.000_001,
    "breathing reaches inhale peak"
)
check(
    abs(PetInteractionModel.breathingAmplitude(elapsed: 4.4 * 0.46, cycle: 4.4) - 1) < 0.000_001,
    "breathing holds briefly"
)
check(
    abs(PetInteractionModel.breathingAmplitude(elapsed: 4.4, cycle: 4.4)) < 0.000_001,
    "breathing loop closes at rest"
)

print("DesktopPetCoreChecks: all checks passed")
