// swift-tools-version: 5.6
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "picograd",
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "picograd",
            targets: ["picograd"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", branch: "main")
    ],
    targets: [
        .target(
            name: "picograd",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics")
            ]),
        .testTarget(
            name: "picogradTests",
            dependencies: [
                "picograd",
                .product(name: "Numerics", package: "swift-numerics"),
            ]),
    ]
)
