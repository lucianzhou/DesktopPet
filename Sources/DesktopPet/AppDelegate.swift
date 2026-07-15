import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var petController: PetWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        do {
            let controller = try PetWindowController(assetDirectory: nil)
            petController = controller
            controller.showWindow(nil)
            controller.window?.orderFrontRegardless()
        } catch {
            let alert = NSAlert(error: error)
            alert.runModal()
            NSApp.terminate(nil)
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        petController?.persistWindowPosition()
        petController?.shutdown()
    }
}
