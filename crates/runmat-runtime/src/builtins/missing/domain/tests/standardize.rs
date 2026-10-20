use super::super::entrypoints::standardize_missing_builtin;
use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn standardize_missing_retains_single_class_and_low_rank_shape() {
    for (shape, values) in [
        (Vec::new(), vec![-99.0_f32]),
        (vec![2], vec![-99.0_f32, 2.0]),
    ] {
        let input = Tensor::from_f32(values, shape.clone()).unwrap();
        let output = block_on(standardize_missing_builtin(
            None,
            Value::Tensor(input),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(-99.0)]),
        ))
        .unwrap();
        let Value::Tensor(output) = output else {
            panic!("missing-value conversion must retain tensor representation");
        };
        assert_eq!(output.shape(), shape);
        let values = output.as_f32_slice().expect("single output storage");
        assert!(values[0].is_nan());
        if values.len() == 2 {
            assert_eq!(values[1], 2.0);
        }
    }
}

#[test]
fn standardize_missing_replaces_indicators() {
    let result = block_on(standardize_missing_builtin(
        None,
        tensor(vec![-99.0, 2.0], vec![1, 2]),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(-99.0)]),
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
        None,
        Value::Tensor(input),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(marker)]),
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
    let error = block_on(standardize_missing_builtin(
        None,
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(-99.0)]),
    ))
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
        None,
        tensor(vec![-99.0, 2.0], vec![1, 2]),
        crate::builtins::common::test_support::builtin_arguments(vec![marker]),
    ))
    .expect("documented integer indicator");
    assert!(
        matches!(result, Value::Tensor(t) if t.materialize_f64()[0].is_nan() && t.materialize_f64()[1] == 2.0)
    );
}

#[test]
fn standardize_missing_wide_integer_indicators_do_not_round_into_false_matches() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let represented = (1_u64 << 53) as f64;
    for marker in [
        Value::Int(IntValue::U64((1_u64 << 53) + 1)),
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap(),
        ),
    ] {
        let result = block_on(standardize_missing_builtin(
            None,
            tensor(vec![represented], vec![1, 1]),
            crate::builtins::common::test_support::builtin_arguments(vec![marker]),
        ))
        .expect("documented integer marker");
        assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![represented]));
    }

    let aligned = Value::Int(IntValue::U64(1_u64 << 54));
    let result = block_on(standardize_missing_builtin(
        None,
        tensor(vec![(1_u64 << 54) as f64], vec![1, 1]),
        crate::builtins::common::test_support::builtin_arguments(vec![aligned]),
    ))
    .expect("exact wide marker");
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64()[0].is_nan()));
}

#[test]
fn standardize_missing_explicit_gpu_indicator_is_gated_before_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let handle = allocation
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    assert!(gpu_helpers::exact_provider_for_handle(&handle).is_none());
    let error = block_on(standardize_missing_builtin(
        None,
        tensor(vec![-99.0, 2.0], vec![1, 2]),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(handle)]),
    ))
    .expect_err("explicit GPU indicator must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:StandardizeMissingExplicitGpuIndicatorExtension")
    );
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn standardize_missing_rejects_host_indicator_before_resident_input_download() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let error = block_on(standardize_missing_builtin(
        None,
        Value::GpuTensor(allocation.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Bool(true)]),
    ))
    .expect_err("invalid indicator must be rejected before input download");
    assert_eq!(error.identifier(), Some("RunMat:missing:UnsupportedType"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}
