use super::*;

#[test]
fn hypot_rejects_typed_complex_integer_inputs() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let complex = ComplexTensor::new_integer(
        IntegerComplexStorage::new(IntegerStorage::I64(vec![1]), IntegerStorage::I64(vec![-2]))
            .expect("storage"),
        vec![1, 1],
    )
    .expect("tensor");

    let left = hypot_builtin(Value::ComplexTensor(complex.clone()), Value::Num(1.0))
        .expect_err("typed complex integer input must reject");
    assert!(left
        .message()
        .contains("complex numbers with integer types"));

    let right = hypot_builtin(Value::Num(1.0), Value::ComplexTensor(complex))
        .expect_err("typed complex integer input must reject");
    assert!(right
        .message()
        .contains("complex numbers with integer types"));
}

#[test]
fn hypot_rejects_excess_outputs() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = hypot_builtin(Value::Num(3.0), Value::Num(4.0))
        .expect_err("second output must be rejected");
    assert_eq!(error.identifier(), HYPOT_ERROR_TOO_MANY_OUTPUTS.identifier);
}

#[test]
fn hypot_rejects_inexact_wide_integer_extension_values() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let value = Value::Int(IntValue::U64(9_007_199_254_740_993));
    let err = hypot_builtin(value, Value::Num(1.0)).expect_err("inexact integer rejects");
    assert_eq!(err.identifier(), Some("RunMat:hypot:InvalidInput"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_string_input_has_stable_identifier() {
    let err = hypot_builtin(Value::from("bad"), Value::Num(1.0)).expect_err("expected error");
    assert_eq!(err.identifier(), HYPOT_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_dimension_mismatch_errors() {
    let lhs = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let err = hypot_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).unwrap_err();
    assert!(
        err.message().contains("dimension"),
        "unexpected error: {err}"
    );
}

#[test]
fn hypot_runmat_extensions_follow_compatibility_mode() {
    for (value, identifier) in [
        (
            Value::Int(IntValue::I32(3)),
            "RunMat:compatibility:HypotIntegerInputExtension",
        ),
        (
            Value::Bool(true),
            "RunMat:compatibility:HypotLogicalInputExtension",
        ),
        (
            Value::CharArray(CharArray::new(vec!['A'], 1, 1).unwrap()),
            "RunMat:compatibility:HypotCharacterInputExtension",
        ),
    ] {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = hypot_builtin(value, Value::Num(4.0)).expect_err("strict mode rejects extension");
        assert_eq!(err.identifier(), Some(identifier));
        assert_eq!(err.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    }
}
