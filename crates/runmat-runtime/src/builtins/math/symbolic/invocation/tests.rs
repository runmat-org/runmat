use super::*;
use runmat_value::{LogicalArray, SymbolicArray, SymbolicExpr, Tensor};
#[test]
fn representation_preflight_preserves_scalar_carriers_and_rejects_array_lookalikes() {
    let array = Value::SymbolicArray(
        SymbolicArray::new(vec![SymbolicExpr::variable("x")], vec![1, 1]).unwrap(),
    );
    let logical = Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap());
    for value in [
        array,
        logical,
        Value::Complex(1., 0.),
        Value::Tensor(Tensor::zeros(vec![0, 1])),
    ] {
        for operation in [
            SymbolicScalarOperation::Integrate,
            SymbolicScalarOperation::Limit,
            SymbolicScalarOperation::Decimal,
            SymbolicScalarOperation::Piecewise,
        ] {
            assert!(!accepts(&value, operation.operand(0, 2)));
        }
    }
    assert!(accepts(
        &Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        SymbolicScalarOperand::Algebra
    ));
    assert!(!accepts(&Value::Num(0.), SymbolicScalarOperand::Variable));
    assert!(!accepts(
        &Value::Symbolic(SymbolicExpr::variable("left")),
        SymbolicScalarOperand::Direction
    ));
    assert!(accepts(
        &Value::from("left"),
        SymbolicScalarOperand::Direction
    ));
}
