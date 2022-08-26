import XCTest
@testable import picograd

final class picogradTests: XCTestCase {
    func testExample() throws {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(picograd().text, "Hello, World!")
    }
}
