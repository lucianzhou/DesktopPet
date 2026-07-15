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
check(model.doubleClick().posture == .standing, "double-click stand")
check(model.standTimerFired().posture == .sitting, "five-second return")
check(model.sleepTimerFired().posture == .sleeping, "sleep timeout")

let center = CGPoint(x: 100, y: 100)
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 100, y: 200), petCenter: center) == 0, "up gaze")
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 200, y: 100), petCenter: center) == 4, "right gaze")
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 100, y: 0), petCenter: center) == 8, "down gaze")
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 0, y: 100), petCenter: center) == 12, "left gaze")
check(PetInteractionModel.gazeDirection(pointer: CGPoint(x: 103, y: 103), petCenter: center) == nil, "deadzone")

print("DesktopPetCoreChecks: all checks passed")
