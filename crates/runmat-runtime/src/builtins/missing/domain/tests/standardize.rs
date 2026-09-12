use super::super::entrypoints::standardize_missing_builtin;
use super::*;

#[test]
fn standardize_missing_replaces_indicators() {
    let result = block_on(standardize_missing_builtin(
        tensor(vec![-99.0, 2.0], vec![1, 2]),
        vec![Value::Num(-99.0)],
    ))
    .unwrap();
    assert!(
        matches!(result, Value::Tensor(t) if t.materialize_f64()[0].is_nan() && t.materialize_f64()[1] == 2.0)
    );
}

#[test]
fn standardize_missing_reads_indicator_storage_and_does_not_nan_integer_targets() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let marker =
        Tensor::new_integer(IntegerStorage::I16(vec![-99]), vec![1, 1]).expect("integer marker");
    let expected = IntegerStorage::I16(vec![-99, 2]);
    let input = Tensor::new_integer(expected.clone(), vec![1, 2]).expect("integer input");

    let result = block_on(standardize_missing_builtin(
        Value::Tensor(input),
        vec![Value::Tensor(marker)],
    ))
    .unwrap();

    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.integer_storage(), Some(&expected));
            assert_eq!(tensor.materialize_f64(), vec![-99.0, 2.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn standardize_missing_integer_data_is_mode_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let input =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![-99, 2]), vec![1, 2]).unwrap());
    let error = block_on(standardize_missing_builtin(input, vec![Value::Num(-99.0)]))
        .expect_err("MATLAB-compatible mode must reject direct integer data");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:StandardizeMissingIntegerDataExtension")
    );
}

#[test]
fn standardize_missing_documented_integer_indicator_needs_no_extension() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let marker =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![-99]), vec![1, 1]).unwrap());
    let result = block_on(standardize_missing_builtin(
        tensor(vec![-99.0, 2.0], vec![1, 2]),
        vec![marker],
    ))
    .expect("documented integer indicator");
    assert!(
        matches!(result, Value::Tensor(t) if t.materialize_f64()[0].is_nan() && t.materialize_f64()[1] == 2.0)
    );
}

#[test]
fn standardize_missing_explicit_gpu_indicator_is_gated_before_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let handle = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 0,
        buffer_id: 9_451_002,
        descriptor: Default::default(),
    };
    let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
    let error = block_on(standardize_missing_builtin(
        tensor(vec![-99.0, 2.0], vec![1, 2]),
        vec![Value::GpuTensor(handle)],
    ))
    .expect_err("explicit GPU indicator must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:StandardizeMissingExplicitGpuIndicatorExtension")
    );
}
