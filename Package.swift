// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "DesktopPet",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "DesktopPet", targets: ["DesktopPet"]),
        .executable(name: "DesktopPetCoreChecks", targets: ["DesktopPetCoreChecks"])
    ],
    targets: [
        .target(name: "DesktopPetCore"),
        .executableTarget(
            name: "DesktopPet",
            dependencies: ["DesktopPetCore"]
        ),
        .executableTarget(
            name: "DesktopPetCoreChecks",
            dependencies: ["DesktopPetCore"]
        )
    ]
)
