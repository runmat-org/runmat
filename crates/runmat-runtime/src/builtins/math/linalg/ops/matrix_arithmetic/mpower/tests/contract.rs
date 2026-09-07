use super::*;

#[test]
fn matrix_single_mpower_preserves_native_storage() {
    let matrix = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result =
        mpower_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("single matrix power");
    let Value::Tensor(result) = result else {
        panic!("expected single matrix");
    };
    assert_eq!(
        result.into_numeric_storage().unwrap(),
        runmat_value::NumericStorage::F32(vec![7.0, 10.0, 15.0, 22.0])
    );
}

#[test]
fn mpower_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = MPOWER_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert!(labels.contains(&"C = mpower(A, B)"));
}

#[test]
fn mpower_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = MPOWER_DESCRIPTOR
        .errors
        .iter()
        .map(|err| err.code)
        .collect();
    assert!(codes.contains(&"RM.MPOWER.INVALID_ARGUMENT"));
    assert!(codes.contains(&"RM.MPOWER.INVALID_INPUT"));
    assert!(codes.contains(&"RM.MPOWER.INTERNAL"));
}
