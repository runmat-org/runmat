#[test]
fn nnz_rejected_result_consumes_only_its_owned_alias() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .unwrap();
        let output = input.with_view(input.view().clone()).unwrap();
        let error = super::validate_provider_result(provider, &input, output, &[1, 1]).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        input.allocation_lease().validate_live().unwrap();
        let host = futures::executor::block_on(provider.download(&input)).unwrap();
        assert_eq!(host.data, vec![1.0]);
    });
}

use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, LogicalArray};

fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
    error.identifier()
}

#[test]
fn nnz_descriptor_signatures_and_errors() {
    let labels: Vec<&str> = NNZ_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"N = nnz(A)"));
    assert!(labels.contains(&"N = nnz(A, dim)"));
    assert!(NNZ_DESCRIPTOR
        .errors
        .iter()
        .any(|err| err.code == NNZ_ERROR_INVALID_ARGUMENT.code));
    assert!(NNZ_DESCRIPTOR
        .errors
        .iter()
        .any(|err| err.code == NNZ_ERROR_INVALID_INPUT.code));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_scalar_zero() {
    let result = nnz_host_value(Value::Num(0.0), None).expect("nnz");
    assert_eq!(result, Value::Num(0.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_scalar_negative_zero_is_zero() {
    let result = nnz_host_value(Value::Num(-0.0), None).expect("nnz");
    assert_eq!(result, Value::Num(0.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_scalar_nonzero() {
    let result = nnz_host_value(Value::Int(IntValue::I32(-5)), None).expect("nnz");
    assert_eq!(result, Value::Num(1.0));
}

#[test]
fn nnz_scalar_wide_uint64_is_nonzero() {
    let result = nnz_host_value(Value::Int(IntValue::U64(u64::MAX)), None).expect("nnz");
    assert_eq!(result, Value::Num(1.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_tensor_counts_entries() {
    let tensor = Tensor::new(vec![1.0, 0.0, -3.0, f64::NAN], vec![2, 2]).unwrap();
    let result = nnz_host_value(Value::Tensor(tensor), None).expect("nnz");
    assert_eq!(result, Value::Num(3.0));
}

#[test]
fn nnz_reads_typed_integer_storage_exactly() {
    let tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![1, 0, -3, 0]), vec![2, 2]).expect("tensor");
    let result = nnz_host_value(Value::Tensor(tensor), None).expect("nnz");
    assert_eq!(result, Value::Num(2.0));
}

#[test]
fn nnz_counts_every_integer_class_from_authoritative_storage() {
    let cases = [
        IntegerStorage::I8(vec![0, i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![0, i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![0, i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![0, i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, 1, u8::MAX]),
        IntegerStorage::U16(vec![0, 1, u16::MAX]),
        IntegerStorage::U32(vec![0, 1, u32::MAX]),
        IntegerStorage::U64(vec![0, 1, u64::MAX]),
    ];
    for storage in cases {
        let tensor = Tensor::new_integer(storage, vec![3, 1]).expect("integer tensor");
        assert_eq!(
            nnz_host_value(Value::Tensor(tensor), None).expect("integer nnz"),
            Value::Num(2.0)
        );
    }
}

#[test]
fn nnz_dimension_extension_gates_before_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident reduction admission fixture");
    let source = Value::GpuTensor(allocation.handle.clone());
    let error = block_on(nnz_builtin(
        None,
        source,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U8(1))]),
    ))
    .expect_err("undocumented dimension overload must gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:NnzDimensionExtension")
    );
    assert_eq!(allocation.probe.retirements(), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    drop(error);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn nnz_no_attempt_is_typed_and_unsupported_text_cannot_authorize_replay() {
    let kind = runmat_builtins::NumericalReductionKind::CountNonzero;
    assert_eq!(
        super::super::numerical_execution::completed::<usize>(kind, Ok(None)).unwrap(),
        None
    );
    let source = std::io::Error::new(
        std::io::ErrorKind::BrokenPipe,
        "reduce_nnz not supported by provider",
    );
    let error = super::super::numerical_execution::completed::<usize>(
        kind,
        Err(runmat_accelerate_api::ProviderExecutionError::from(source)),
    )
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<std::io::Error>() {
        cause = cause.source().expect("original provider cause");
    }
    assert_eq!(
        cause.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::BrokenPipe
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_sparse_ignores_explicit_stored_zeros() {
    let sparse = SparseTensor::new(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 2, 1, 2],
        vec![0.0, 5.0, 0.0, f64::NAN],
    )
    .unwrap();
    let result = nnz_host_value(Value::SparseTensor(sparse), None).expect("nnz");
    assert_eq!(result, Value::Num(2.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_matrix_dimension_one() {
    let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 5.0], vec![2, 2]).unwrap();
    let result = nnz_host_value(Value::Tensor(tensor), Some(1)).expect("nnz");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 2]);
            assert_eq!(out.materialize_f64(), vec![1.0, 2.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn nnz_dimension_reads_typed_integer_storage_exactly() {
    let tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![1, 0, 2, 5]), vec![2, 2]).expect("tensor");
    let result = nnz_host_value(Value::Tensor(tensor), Some(1)).expect("nnz");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 2]);
            assert_eq!(out.materialize_f64(), vec![1.0, 2.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_matrix_dimension_two() {
    let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 5.0], vec![2, 2]).unwrap();
    let result = nnz_host_value(Value::Tensor(tensor), Some(2)).expect("nnz");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [2, 1]);
            assert_eq!(out.materialize_f64(), vec![2.0, 1.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_sparse_dimension_ignores_explicit_stored_zeros() {
    let sparse = SparseTensor::new(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 2, 1, 2],
        vec![0.0, 5.0, 0.0, f64::NAN],
    )
    .unwrap();

    let per_column = nnz_host_value(Value::SparseTensor(sparse.clone()), Some(1)).expect("nnz");
    match per_column {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 2]);
            assert_eq!(out.materialize_f64(), vec![1.0, 1.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }

    let per_row = nnz_host_value(Value::SparseTensor(sparse), Some(2)).expect("nnz");
    match per_row {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [3, 1]);
            assert_eq!(out.materialize_f64(), vec![0.0, 0.0, 2.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn nnz_sparse_dimension_one_avoids_dense_mask_allocation() {
    let sparse =
        SparseTensor::new(usize::MAX, 2, vec![0, 1, 1], vec![0], vec![7.0]).expect("sparse");

    let result = nnz_host_value(Value::SparseTensor(sparse), Some(1)).expect("nnz");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 2]);
            assert_eq!(out.materialize_f64(), vec![1.0, 0.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_empty_matrix_dimension_returns_zero_counts() {
    let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let result = nnz_host_value(Value::Tensor(tensor), Some(1)).expect("nnz");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(out.materialize_f64(), vec![0.0, 0.0, 0.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_scalar_with_dimension_returns_scalar() {
    let tensor = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
    let result = nnz_host_value(Value::Tensor(tensor), Some(1)).expect("nnz");
    assert_eq!(result, Value::Num(1.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_three_dimensional_reduction_counts_per_slice() {
    let data = vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2, 2]).unwrap();
    let result = nnz_host_value(Value::Tensor(tensor), Some(3)).expect("nnz");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [2, 2, 1]);
            assert_eq!(out.materialize_f64(), vec![2.0, 0.0, 0.0, 2.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_dimension_greater_than_ndims_returns_mask() {
    let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
    let result = nnz_host_value(Value::Tensor(tensor), Some(3)).expect("nnz");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [2, 2]);
            assert_eq!(out.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn mask_from_sparse_rejects_overflowing_logical_size() {
    let sparse = SparseTensor::zeros(usize::MAX, 2);

    let err = match mask_from_sparse(&sparse) {
        Ok(_) => panic!("expected sparse mask overflow error"),
        Err(err) => err,
    };
    assert_eq!(err.identifier(), NNZ_ERROR_INTERNAL.identifier);
    assert!(err.message().contains("sparse dimensions overflow usize"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_empty_tensor_is_zero() {
    let tensor = Tensor::new(vec![], vec![0, 3]).unwrap();
    let result = nnz_host_value(Value::Tensor(tensor), None).expect("nnz");
    assert_eq!(result, Value::Num(0.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_complex_tensor_counts() {
    let data = vec![(0.0, 0.0), (2.0, 0.0), (0.0, -4.0), (0.0, 0.0)];
    let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
    let result = nnz_host_value(Value::ComplexTensor(tensor), None).expect("nnz");
    assert_eq!(result, Value::Num(2.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_logical_array_counts_true() {
    let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![4]).unwrap();
    let result = nnz_host_value(Value::LogicalArray(logical), None).expect("nnz");
    assert_eq!(result, Value::Num(2.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_char_array_counts_nonzero_codepoints() {
    let chars = CharArray::from_unicode_scalars(vec!['a', '\0', 'c'], 1, 3).unwrap();
    let result = nnz_host_value(Value::CharArray(chars), None).expect("nnz");
    assert_eq!(result, Value::Num(2.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_complex_scalar_nan_counts() {
    let result = nnz_host_value(Value::Complex(f64::NAN, 0.0), None).expect("nnz");
    assert_eq!(result, Value::Num(1.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 0.0, 3.0], vec![5, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = futures::executor::block_on(provider::execute(
            &handle,
            None,
            &super::super::numerical_execution::NumericalCall::new(
                runmat_builtins::NumericalReductionKind::CountNonzero,
                None,
            )
            .unwrap(),
        ))
        .expect("nnz");
        assert_eq!(result, Value::Num(3.0));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn nnz_wgpu_dim_matches_cpu() {
    let _guard = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("register WGPU provider");
    let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 5.0], vec![2, 2]).unwrap();
    let cpu = nnz_host_value(Value::Tensor(tensor.clone()), Some(1)).expect("nnz");
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let gpu = futures::executor::block_on(provider::execute(
        &handle,
        Some(1),
        &super::super::numerical_execution::NumericalCall::new(
            runmat_builtins::NumericalReductionKind::CountNonzero,
            None,
        )
        .unwrap(),
    ))
    .expect("nnz");
    match (cpu, gpu) {
        (Value::Tensor(ct), Value::Tensor(gt)) => {
            assert_eq!(gt.shape(), ct.shape());
            assert_eq!(gt.materialize_f64(), ct.materialize_f64());
        }
        other => panic!("unexpected comparison result {other:?}"),
    }
}

#[test]
#[cfg(feature = "wgpu")]
fn nnz_wgpu_integer_fallback_is_exact_and_preserves_source() {
    let _guard = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("register WGPU provider");
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
        vec![3, 1],
    )
    .expect("integer tensor");
    let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer");
    let result = block_on(provider::execute(
        &handle.clone(),
        None,
        &super::super::numerical_execution::NumericalCall::new(
            runmat_builtins::NumericalReductionKind::CountNonzero,
            None,
        )
        .unwrap(),
    ))
    .expect("integer nnz fallback");
    assert_eq!(result, Value::Num(2.0));
    let source = block_on(gpu_helpers::download_value_preserving_residency_async(
        provider, &handle,
    ))
    .expect("source remains live");
    assert!(
        matches!(source, Value::Tensor(tensor) if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX])))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_rejects_strings() {
    let err = nnz_host_value(Value::from("hello"), None).unwrap_err();
    assert_eq!(error_identifier(&err), NNZ_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains(NNZ_ERROR_INVALID_INPUT.message));
    assert!(
        err.message()
            .contains("expected numeric, logical, complex, or char array"),
        "unexpected error: {err}"
    );
    assert!(
        err.message().contains("string scalar"),
        "unexpected error: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nnz_too_many_args_sets_invalid_argument_identifier() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let err = block_on(nnz_builtin(
        None,
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::Num(2.0),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        error_identifier(&err),
        NNZ_ERROR_INVALID_ARGUMENT.identifier
    );
    assert!(err.message().contains(NNZ_ERROR_INVALID_ARGUMENT.message));
}
