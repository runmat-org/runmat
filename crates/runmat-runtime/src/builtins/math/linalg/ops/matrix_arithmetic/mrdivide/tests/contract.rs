use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn divides_scalar_by_scalar() {
    let result = mrdivide_builtin(Value::Num(6.0), Value::Num(2.0)).expect("mrdivide");
    match result {
        Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[test]
fn mrdivide_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = MRDIVIDE_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert!(labels.contains(&"X = mrdivide(A, B)"));
}

#[test]
fn mrdivide_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = MRDIVIDE_DESCRIPTOR
        .errors
        .iter()
        .map(|err| err.code)
        .collect();
    assert!(codes.contains(&"RM.MRDIVIDE.INVALID_INPUT"));
    assert!(codes.contains(&"RM.MRDIVIDE.INTERNAL"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn divides_matrix_by_scalar() {
    let tensor = Tensor::new(vec![2.0, 4.0, 6.0], vec![1, 3]).expect("tensor");
    let result = mrdivide_builtin(Value::Tensor(tensor), Value::Num(2.0)).expect("mrdivide");
    match result {
        Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![1.0, 2.0, 3.0]),
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn host_single_solve_preserves_single_storage() {
    let lhs = Tensor::from_f32(vec![8.0, 12.0], vec![1, 2]).expect("single A");
    let rhs = Tensor::from_f32(vec![2.0, 0.0, 0.0, 4.0], vec![2, 2]).expect("single B");
    let Value::Tensor(result) =
        mrdivide_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("single solve")
    else {
        panic!("expected tensor")
    };
    assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(result.materialize_f64(), vec![4.0, 3.0]);
}

#[test]
fn ambient_f64_provider_does_not_widen_host_single_solve() {
    test_support::with_test_provider(|_| {
        let lhs = Tensor::from_f32(vec![8.0, 12.0], vec![1, 2]).expect("A");
        let rhs = Tensor::from_f32(vec![2.0, 0.0, 0.0, 4.0], vec![2, 2]).expect("B");
        let Value::Tensor(result) =
            mrdivide_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("solve")
        else {
            panic!("precision mismatch must use host fallback")
        };
        assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
    });
}

#[test]
fn mrdivide_declares_documented_integer_scalar_capability() {
    let metadata = runmat_builtins::builtin_integer_metadata_by_name(NAME).expect("mrdivide");
    assert_eq!(metadata.capabilities.len(), 1);
    let capability = &metadata.capabilities[0];
    assert_eq!(capability.inputs.len(), 2);
    assert_eq!(capability.inputs[0].classes.len(), 8);
    assert_eq!(capability.inputs[1].classes.len(), 8);
    assert_eq!(
        capability.output_class,
        BuiltinIntegerOutputClassRule::PreserveNondoubleInput
    );
    assert_eq!(
        capability.backend,
        BuiltinIntegerBackendRule::GatherFallback
    );
}
