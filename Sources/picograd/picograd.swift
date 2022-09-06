import Numerics

typealias DiffT = Real

// MARK: Sum operations on `Value`

/// Forward and backward pass for `+` operation
func +<T>(lhs: Value<T>, rhs: Value<T>) -> Value<T> where T: DiffT {
    // Forward pass
    let sumValue = Value(lhs.data + rhs.data, inputs: [lhs, rhs], op: "+")

    // Backward pass
    sumValue._backward = {
        lhs.grad += sumValue.grad
        rhs.grad += sumValue.grad
    }
    return sumValue
}

func +<T>(lhs: T, rhs: Value<T>) -> Value<T> where T: DiffT {
    let lhsValue = Value(lhs)
    return lhsValue + rhs
}

func +<T>(lhs: Value<T>, rhs: T) -> Value<T> where T: DiffT {
    let rhsValue = Value(rhs)
    return lhs + rhsValue
}

// MARK: Mul operations on `Value`

/// Forward and backward pass for `-` operation

func *<T>(lhs: Value<T>, rhs: Value<T>) -> Value<T> where T: DiffT {
    // Forward pass
    let mulValue = Value(lhs.data * rhs.data, inputs: [lhs, rhs], op: "*")

    // Backward pass
    mulValue._backward = {
        lhs.grad += rhs.data * mulValue.grad
        rhs.grad += lhs.data * mulValue.grad
    }
    return mulValue
}

func *<T>(lhs: T, rhs: Value<T>) -> Value<T> where T: DiffT {
    // Forward pass
    let lhsValue = Value(lhs)
    return lhsValue * rhs;
}

func *<T>(lhs: Value<T>, rhs: T) -> Value<T> where T: DiffT {
    // Forward pass
    let rhsValue = Value(rhs)
    return lhs * rhsValue;
}

//MARK: Pow operation on `Value`

infix operator ** : MultiplicationPrecedence
func **<T>(lhs: Value<T>, rhs: T) -> Value<T> where T: DiffT {
    // Forward pass
    let powValue = Value(T.pow(lhs.data, rhs), inputs: [lhs], op: "**\(rhs)")

    // Backward pass
    powValue._backward = {
        lhs.grad += rhs * (T.pow(lhs.data, rhs - 1)) * powValue.grad
    }

    return powValue
}


// MARK: `Value` definition
class Value<T: DiffT> : CustomStringConvertible {
    /// String representation of `Value`
    var description: String {
        return "Value(data=\(self.data), grad=\(self.grad), label=\(self.label))"
    }

    /// Pretty name
    let label: String

    /// Underlaying data
    let data: T

    /// Computed gradient
    var grad: T = T.zero

    /// Operation that triggered the generation of this value
    let op: String

    /// Set of `Value` objects that contribute to compute `self`
    let inputs: Set<Value>

    /// Closure that computes the gradient
    var _backward: () -> Void

    init(_ data: T, label: String = "", inputs: Set<Value> = [], op: String = "") {
        self.data = data
        self.label = label
        self.op = op
        self.inputs = inputs
        self._backward = {}
    }

    /// Triggers backward propagation of gradients using as origin node `self`
    func backward() -> Void {
        var topo = Array<Value>()
        var visited = Set<Value>()

        // Compute topological sort on the graph that leads to this `Value`
        func build_topo(value: Value) {
            guard !visited.contains(value) else { return }
            visited.update(with: value)
            value.inputs.forEach { build_topo(value: $0) }
            topo.append(value)
        }
        build_topo(value: self)

        self.grad = 1
        topo.reversed().forEach { $0._backward() }
    }
}

// MARK: Extentions on `Value`
extension Value: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

extension Value: Equatable {
    static func == (lhs: Value<T>, rhs: Value<T>) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }
}
