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
        assert(l.data.isApproximatelyEqual(to: 5.0), "`l.data(\(l.data))` shuold be 5.0")
        assert(a.grad.isApproximatelyEqual(to: 1.0), "`a.grad(\(a.grad))` should be 1.0 ")
        assert(b.grad.isApproximatelyEqual(to: 1.0), "`b.grad(\(a.grad))` should be 1.0 ")
    }

    func testMulBackward() throws {
        let a = Value(2.0, label: "a")
        let b = Value(3.0, label: "b")
        let l = a * b

        l.backward()

        assert(l.grad.isApproximatelyEqual(to: 1.0))
        assert(l.data.isApproximatelyEqual(to: 6.0), "`l.data(\(l.data))` shuold be 5.0")
        assert(a.grad.isApproximatelyEqual(to: b.data), "`a.grad(\(a.grad))` should be 1.0 ")
        assert(b.grad.isApproximatelyEqual(to: a.data), "`b.grad(\(a.grad))` should be 1.0 ")
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

    func testPowBackward() throws {
        let a = Value(2.0, label: "a")
        let l = a ** 4.0
        l.backward()

        let dlda = 4.0 * pow(2.0, 3.0)

        assert(a.grad.isApproximatelyEqual(to: dlda), "`a.grad(\(a.grad))` shuold be \(dlda)")
    }

    func testSumsAndMultsCombined() throws {
        let a = Value(2.0, label: "a")
        let b = Value(3.0, label: "b")
        let c = Value(5.0, label: "c")
        let d = Value(8.0, label: "d")
        let e = Value(2.0, label: "l")

        let l = ((a*b*c) + d)*e
        l.backward()

        let dlde = (a*b*c) + d
        let dldd = e
        let dlda = e*b*c
        let dldb = e*a*c
        let dldc = e*a*b

        assert(e.grad.isApproximatelyEqual(to: dlde.data), "e.grad(\(e.grad)) should be \(dlde.data)")
        assert(d.grad.isApproximatelyEqual(to: dldd.data), "d.grad(\(d.grad)) should be \(dldd.data)")
        assert(a.grad.isApproximatelyEqual(to: dlda.data), "a.grad(\(a.grad)) should be \(dlda.data)")
        assert(b.grad.isApproximatelyEqual(to: dldb.data), "b.grad(\(b.grad)) should be \(dldb.data)")
        assert(c.grad.isApproximatelyEqual(to: dldc.data), "c.grad(\(c.grad)) should be \(dldc.data)")
    }

    func testUnaryMinus() throws {
        let a = Value(2.0, label: "a")
        let b = Value(3.0, label: "b")
        let c = -a
        let d = b + c
        d.backward()


        assert(c.data.isApproximatelyEqual(to: -a.data), "c.data(\(c.data) should be \(-a.data)")
        assert(a.grad.isApproximatelyEqual(to: -1.0), "a.grad(\(a.grad) should be -1.0")
    }

}
