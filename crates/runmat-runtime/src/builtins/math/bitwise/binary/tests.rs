use super::{bitand_builtin, bitor_builtin, bitxor_builtin};
use futures::executor::block_on;
use runmat_value::{IntegerStorage, Tensor, Value};

#[test]
fn operators_preserve_typed_storage_and_broadcast() {
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U16(vec![1, 3, 7]), vec![1, 3]).expect("input"),
    );
    let mask = Value::Int(runmat_value::IntValue::U16(3));

    for (result, expected) in [
        (
            block_on(bitand_builtin(vec![input.clone(), mask.clone()])),
            vec![1, 3, 3],
        ),
        (
            block_on(bitor_builtin(vec![input.clone(), mask.clone()])),
            vec![3, 3, 7],
        ),
        (
            block_on(bitxor_builtin(vec![input.clone(), mask.clone()])),
            vec![2, 0, 4],
        ),
    ] {
        let Value::Tensor(result) = result.expect("binary bitwise result") else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U16(expected))
        );
        assert_eq!(result.shape, vec![1, 3]);
    }
}

#[test]
fn logical_inputs_produce_logical_outputs() {
    let result =
        block_on(bitxor_builtin(vec![Value::Bool(true), Value::Bool(false)])).expect("logical xor");
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn incompatible_integer_classes_are_rejected() {
    let error = block_on(bitand_builtin(vec![
        Value::Int(runmat_value::IntValue::U16(1)),
        Value::Int(runmat_value::IntValue::U32(1)),
    ]))
    .expect_err("mixed classes must fail");
    assert_eq!(error.identifier(), Some("RunMat:bitwise:InvalidInput"));
}
