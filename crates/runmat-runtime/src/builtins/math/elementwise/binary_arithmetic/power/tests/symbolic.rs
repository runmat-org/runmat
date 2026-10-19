use super::*;
use runmat_builtins::POWER_ERROR_SIZE_MISMATCH;
#[test]
fn power_symbolic_array_with_scalar_builds_symbolic_array() {
    let array = SymbolicArray::new(
        vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
        vec![1, 2],
    )
    .unwrap();

    let result =
        power_builtin(Value::SymbolicArray(array), Value::Num(2.0), Vec::new()).expect("power");

    match result {
        Value::SymbolicArray(array) => {
            assert_eq!(array.shape, vec![1, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x^2", "y^2"]
            );
        }
        other => panic!("expected symbolic array, got {other:?}"),
    }
}

#[test]
fn power_scalar_with_symbolic_array_builds_symbolic_array() {
    let array = SymbolicArray::new(
        vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
        vec![1, 2],
    )
    .unwrap();

    let result =
        power_builtin(Value::Num(2.0), Value::SymbolicArray(array), Vec::new()).expect("power");

    match result {
        Value::SymbolicArray(array) => {
            assert_eq!(array.shape, vec![1, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["2^x", "2^y"]
            );
        }
        other => panic!("expected symbolic array, got {other:?}"),
    }
}

#[test]
fn power_compatible_symbolic_arrays_builds_symbolic_array() {
    let lhs = SymbolicArray::new(
        vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
        vec![1, 2],
    )
    .unwrap();
    let rhs = SymbolicArray::new(
        vec![SymbolicExpr::constant(2.0), SymbolicExpr::constant(3.0)],
        vec![1, 2],
    )
    .unwrap();

    let result = power_builtin(
        Value::SymbolicArray(lhs),
        Value::SymbolicArray(rhs),
        Vec::new(),
    )
    .expect("power");

    match result {
        Value::SymbolicArray(array) => {
            assert_eq!(array.shape, vec![1, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x^2", "y^3"]
            );
        }
        other => panic!("expected symbolic array, got {other:?}"),
    }
}

#[test]
fn power_symbolic_arrays_support_singleton_expansion() {
    let lhs = SymbolicArray::new(
        vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
        vec![2, 1],
    )
    .unwrap();
    let rhs = SymbolicArray::new(
        vec![SymbolicExpr::constant(2.0), SymbolicExpr::constant(3.0)],
        vec![1, 2],
    )
    .unwrap();

    let result = power_builtin(
        Value::SymbolicArray(lhs),
        Value::SymbolicArray(rhs),
        Vec::new(),
    )
    .expect("power");

    match result {
        Value::SymbolicArray(array) => {
            assert_eq!(array.shape, vec![2, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x^2", "y^2", "x^3", "y^3"]
            );
        }
        other => panic!("expected symbolic array, got {other:?}"),
    }
}

#[test]
fn power_symbolic_array_shape_mismatch_errors() {
    let lhs = SymbolicArray::new(
        vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
        vec![1, 2],
    )
    .unwrap();
    let rhs = SymbolicArray::new(
        vec![
            SymbolicExpr::constant(1.0),
            SymbolicExpr::constant(2.0),
            SymbolicExpr::constant(3.0),
        ],
        vec![1, 3],
    )
    .unwrap();

    let err = power_builtin(
        Value::SymbolicArray(lhs),
        Value::SymbolicArray(rhs),
        Vec::new(),
    )
    .expect_err("shape mismatch should fail");

    assert_eq!(err.identifier(), POWER_ERROR_SIZE_MISMATCH.identifier);
}
