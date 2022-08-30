import XCTest
import Numerics
@testable import picograd

final class picogradTests: XCTestCase {
    func testSumBackward() throws {
        let a = Value(2.0, label: "a")
        let b = Value(3.0, label: "b")

        let l = a + b
        l.backward()

        assert(l.grad.isApproximatelyEqual(to: 1.0))
        assert(l.data.isApproximatelyEqual(to: 5.0), "`l.data(\(l.data)` shuold be 5.0")
        assert(a.grad.isApproximatelyEqual(to: 1.0), "`a.grad(\(a.grad)` should be 1.0 ")
        assert(b.grad.isApproximatelyEqual(to: 1.0), "`b.grad(\(a.grad)` should be 1.0 ")
    }

    func testMulBackward() throws {
        let a = Value(2.0, label: "a")
        let b = Value(3.0, label: "b")
        let l = a * b

        l.backward()

        assert(l.grad.isApproximatelyEqual(to: 1.0))
        assert(l.data.isApproximatelyEqual(to: 6.0), "`l.data(\(l.data)` shuold be 5.0")
        assert(a.grad.isApproximatelyEqual(to: b.data), "`a.grad(\(a.grad)` should be 1.0 ")
        assert(b.grad.isApproximatelyEqual(to: a.data), "`b.grad(\(a.grad)` should be 1.0 ")
    }

    func testOneValueOperandInSum() throws {
        let a = 3.0 + Value(2.0)
        let b = Value(2.0) + 3.0

        assert(a.data.isApproximatelyEqual(to: 5.0))
        assert(b.data.isApproximatelyEqual(to: 5.0))
    }

    func testOneValueOperandInMul() throws {
        let a = 3.0 * Value(2.0)
        let b = Value(2.0) * 3.0

        assert(a.data.isApproximatelyEqual(to: 6.0))
        assert(b.data.isApproximatelyEqual(to: 6.0))
    }

}