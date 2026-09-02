use runmat_builtins::RelationalOperator;
use runmat_value::{SymbolicArray, SymbolicExpr, Value};

use super::compare_symbolic;

#[test]
fn named_symbolic_relations_use_the_typed_operator_identity() {
    let lhs = Value::Symbolic(SymbolicExpr::variable("x"));
    let rhs = Value::Num(0.0);
    let cases = [
        (RelationalOperator::NotEqual, "ne"),
        (RelationalOperator::LessThan, "lt"),
        (RelationalOperator::LessThanOrEqual, "le"),
        (RelationalOperator::GreaterThan, "gt"),
        (RelationalOperator::GreaterThanOrEqual, "ge"),
    ];

    for (operator, expected_name) in cases {
        let result = compare_symbolic(&lhs, &rhs, operator)
            .expect("valid symbolic relation")
            .expect("symbolic output");
        let Value::Symbolic(SymbolicExpr::FunctionCall(name, arguments)) = result else {
            panic!("expected a named symbolic relation");
        };
        assert_eq!(name, expected_name);
        assert_eq!(arguments.len(), 2);
    }
}

#[test]
fn named_symbolic_relations_broadcast_arrays() {
    let lhs = SymbolicArray::new(
        vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
        vec![2, 1],
    )
    .expect("symbolic column");
    let result = compare_symbolic(
        &Value::SymbolicArray(lhs),
        &Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        RelationalOperator::GreaterThan,
    )
    .expect("broadcast relation")
    .expect("symbolic output");

    let Value::SymbolicArray(result) = result else {
        panic!("expected a symbolic array");
    };
    assert_eq!(result.shape, vec![2, 3]);
    assert_eq!(result.data.len(), 6);
}
