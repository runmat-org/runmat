use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{IntegerStorage, NumericDType, NumericStorage, Tensor};

fn gradient_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
    block_on(super::gradient_builtin(None, value, rest))
}

#[test]
fn gradient_integer_extensions_and_capabilities_are_declared() {
    assert_eq!(GRADIENT_EXTENSIONS.len(), 4);
    assert_eq!(GRADIENT_INTEGER_CAPABILITIES.len(), 2);
    assert!(GRADIENT_EXTENSIONS
        .iter()
        .any(|extension| extension.id == "gradient-integer-data"));
    assert!(GRADIENT_EXTENSIONS
        .iter()
        .any(|extension| extension.id == "gradient-integer-spacing"));
}

#[test]
fn gradient_strict_mode_rejects_integer_data_before_computation() {
    let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
    let input =
        Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3]).expect("integer input");
    let error = block_on(super::gradient_builtin(
        None,
        Value::Tensor(input),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("integer data is an extension");
    assert_eq!(
        error.identifier(),
        GRADIENT_INTEGER_DATA_EXTENSION.error_identifier
    );
}

#[test]
fn gradient_rejects_inexact_integer_data_and_spacing() {
    let inexact = 9_007_199_254_740_993_u64;
    let data = Tensor::new_integer(IntegerStorage::U64(vec![inexact, inexact + 2]), vec![1, 2])
        .expect("integer data");
    let error = gradient_builtin(
        Value::Tensor(data),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect_err("inexact integer data must reject");
    assert!(error.message().contains("exactly representable as double"));

    let data = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("double data");
    let spacing = Tensor::new_integer(IntegerStorage::U64(vec![inexact]), vec![1, 1])
        .expect("integer spacing");
    let error = gradient_builtin(
        Value::Tensor(data),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
    )
    .expect_err("inexact integer spacing must reject");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn gradient_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = GRADIENT_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"G = gradient(F)"));
    assert!(labels.contains(&"G = gradient(F, h)"));
    assert!(labels.contains(&"[G1, G2, ...] = gradient(F)"));
    assert!(labels.contains(&"[G1, G2, ...] = gradient(F, h1, h2, ...)"));
}

#[test]
fn gradient_descriptor_errors_have_stable_codes() {
    assert!(GRADIENT_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == GRADIENT_ERROR_INVALID_ARGUMENT.code));
    assert!(GRADIENT_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == GRADIENT_ERROR_INVALID_INPUT.code));
    assert!(GRADIENT_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == GRADIENT_ERROR_INTERNAL.code));
}

#[test]
fn gradient_row_vector_returns_horizontal_derivative() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("gradient");
    assert_eq!(
        result,
        Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap())
    );
}

#[test]
fn gradient_one_dimensional_tensor_is_treated_as_row_vector() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3]).unwrap();
    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    )
    .expect("gradient");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(out.materialize_f64(), vec![1.5, 2.0, 2.5]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn gradient_matrix_outputs_follow_matlab_order() {
    let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("gradient");
    match result {
        Value::OutputList(outputs) => {
            let fx = test_support::gather(outputs[0].clone()).expect("fx");
            let fy = test_support::gather(outputs[1].clone()).expect("fy");
            assert_eq!(fx.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0]);
            assert_eq!(fy.materialize_f64(), vec![2.0, 2.0, 2.0, 2.0]);
        }
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn gradient_scalar_spacing_scales_output() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    )
    .expect("gradient");
    match result {
        Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![1.5, 2.0, 2.5]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn gradient_scalar_spacing_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let spacing = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("spacing");

    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
    )
    .expect("gradient");
    match result {
        Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![1.5, 2.0, 2.5]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn gradient_input_reads_typed_integer_storage_without_mirror() {
    let tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![1, 4, 9]), vec![1, 3]).expect("input");

    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("gradient");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(out.numeric_dtype(), NumericDType::F64);
            assert!(out.integer_storage().is_none());
            assert_eq!(out.materialize_f64(), vec![3.0, 4.0, 5.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn gradient_multi_output_uses_typed_integer_storage_length_without_mirror() {
    let tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2]).expect("input");
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("gradient");
    match result {
        Value::OutputList(outputs) => {
            let fx = test_support::gather(outputs[0].clone()).expect("fx");
            let fy = test_support::gather(outputs[1].clone()).expect("fy");
            assert_eq!(fx.shape(), [2, 2]);
            assert_eq!(fx.numeric_dtype(), NumericDType::F64);
            assert!(fx.integer_storage().is_none());
            assert_eq!(fx.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0]);
            assert_eq!(fy.shape(), [2, 2]);
            assert_eq!(fy.numeric_dtype(), NumericDType::F64);
            assert!(fy.integer_storage().is_none());
            assert_eq!(fy.materialize_f64(), vec![2.0, 2.0, 2.0, 2.0]);
        }
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn gradient_preserves_single_precision_host_tensor() {
    let tensor =
        Tensor::new_with_dtype(vec![1.0, 4.0, 9.0], vec![1, 3], NumericDType::F32).unwrap();
    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("gradient");
    match result {
        Value::Tensor(out) => assert_eq!(
            out.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![3.0, 4.0, 5.0])
        ),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn gradient_complex_host_supported() {
    let tensor = ComplexTensor::new(vec![(1.0, 1.0), (4.0, 3.0), (9.0, 6.0)], vec![1, 3]).unwrap();
    let result = gradient_builtin(
        Value::ComplexTensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("gradient");
    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(
                out.materialize_f64(),
                vec![(3.0, 2.0), (4.0, 2.5), (5.0, 3.0)]
            );
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn gradient_coordinate_vector_spacing_for_row_vector() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let spacing = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
    )
    .expect("gradient");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(out.materialize_f64(), vec![3.0, 8.0 / 3.0, 2.5]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn gradient_coordinate_vector_spacing_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let spacing =
        Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3]).expect("spacing");

    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
    )
    .expect("gradient");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(out.materialize_f64(), vec![3.0, 8.0 / 3.0, 2.5]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn gradient_mixed_scalar_and_coordinate_vector_spacing_for_matrix() {
    let tensor = Tensor::new(vec![0.0, 20.0, 1.0, 21.0, 9.0, 29.0], vec![2, 3]).unwrap();
    let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(x),
            Value::Num(2.0),
        ]),
    )
    .expect("gradient");
    match result {
        Value::OutputList(outputs) => {
            let fx = test_support::gather(outputs[0].clone()).expect("fx");
            let fy = test_support::gather(outputs[1].clone()).expect("fy");
            assert_eq!(fx.shape(), [2, 3]);
            assert_eq!(fx.materialize_f64(), vec![1.0, 1.0, 3.0, 3.0, 4.0, 4.0]);
            assert_eq!(fy.shape(), [2, 3]);
            assert_eq!(
                fy.materialize_f64(),
                vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
            );
        }
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn gradient_complex_coordinate_vector_spacing() {
    let tensor = ComplexTensor::new(vec![(1.0, 1.0), (4.0, 3.0), (9.0, 7.0)], vec![1, 3]).unwrap();
    let spacing = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
    let result = gradient_builtin(
        Value::ComplexTensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
    )
    .expect("gradient");
    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(out.shape(), [1, 3]);
            assert_eq!(
                out.materialize_f64(),
                vec![(3.0, 2.0), (8.0 / 3.0, 2.0), (2.5, 2.0)]
            );
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn gradient_rejects_coordinate_vector_length_mismatch() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let spacing = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
    let err = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
    )
    .unwrap_err();
    assert_eq!(err.identifier(), GRADIENT_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("length"));
}

#[test]
fn gradient_allows_nonmonotonic_coordinate_vector_spacing() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let spacing = Tensor::new(vec![0.0, 1.0, 0.5], vec![1, 3]).unwrap();
    let result = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
    )
    .expect("gradient");
    match result {
        Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![3.0, 16.0, -10.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn gradient_rejects_zero_coordinate_denominator() {
    let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
    let spacing = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
    let err = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
    )
    .unwrap_err();
    assert_eq!(err.identifier(), GRADIENT_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("denominator"));
}

#[test]
fn gradient_rejects_too_many_outputs() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let err = gradient_builtin(
        Value::Tensor(tensor),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .unwrap_err();
    assert_eq!(err.identifier(), GRADIENT_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("requested 2 outputs"));
}

#[test]
#[cfg(feature = "wgpu")]
fn gradient_gpu_scalar_spacing_matches_cpu_and_stays_resident() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let host = Tensor::new_with_dtype(vec![1.0, 4.0, 9.0], vec![1, 3], NumericDType::F32).unwrap();
    let view = HostTensorView {
        data: &host.materialize_f64(),
        shape: host.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let result = gradient_builtin(
        Value::GpuTensor(handle),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    )
    .expect("gradient");
    match result {
        Value::GpuTensor(out) => {
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![1.5, 2.0, 2.5]);
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
        }
        other => panic!("expected gpu tensor, got {other:?}"),
    }
}

#[test]
#[cfg(feature = "wgpu")]
fn gradient_gpu_coordinate_spacing_matches_cpu_and_stays_resident() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let host = Tensor::new_with_dtype(vec![1.0, 4.0, 9.0], vec![1, 3], NumericDType::F32).unwrap();
    let view = HostTensorView {
        data: &host.materialize_f64(),
        shape: host.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let spacing = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
    let result = gradient_builtin(
        Value::GpuTensor(handle),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
    )
    .expect("gradient");
    match result {
        Value::GpuTensor(out) => {
            let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
            assert_eq!(gathered.shape(), [1, 3]);
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            let expected = [3.0, 8.0 / 3.0, 2.5];
            for (idx, (actual, expected)) in
                gathered.materialize_f64().iter().zip(expected).enumerate()
            {
                assert!(
                    (*actual - expected).abs() < 1.0e-5,
                    "gradient mismatch at {idx}: actual={actual} expected={expected}"
                );
            }
        }
        other => panic!("expected gpu tensor, got {other:?}"),
    }
}

#[test]
#[cfg(feature = "wgpu")]
fn gradient_gpu_one_dimensional_shape_matches_matlab_row_vector_semantics() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let data = [1.0, 4.0, 9.0];
    let shape = [3usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    let handle = provider.upload(&view).expect("upload");
    let result = gradient_builtin(
        Value::GpuTensor(handle),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    )
    .expect("gradient");
    let gathered = test_support::gather(result).expect("gather");
    assert_eq!(gathered.shape(), [1, 3]);
    assert_eq!(gathered.materialize_f64(), vec![1.5, 2.0, 2.5]);
}

#[test]
#[cfg(feature = "wgpu")]
fn gradient_gpu_multi_output_uses_output_list() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let host = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let view = HostTensorView {
        data: &host.materialize_f64(),
        shape: host.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let _out_guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = gradient_builtin(
        Value::GpuTensor(handle),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("gradient");
    match result {
        Value::OutputList(outputs) => {
            assert!(matches!(outputs[0], Value::GpuTensor(_)));
            assert!(matches!(outputs[1], Value::GpuTensor(_)));
        }
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn gradient_gpu_coordinate_vector_spacing_stays_resident() {
    test_support::with_test_provider(|provider| {
        let host = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let view = HostTensorView {
            data: &host.materialize_f64(),
            shape: host.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let spacing = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let result = gradient_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(spacing)]),
        )
        .expect("gradient");
        match result {
            Value::GpuTensor(out_handle) => {
                let out = test_support::gather(Value::GpuTensor(out_handle)).expect("gather");
                assert_eq!(out.shape(), [1, 3]);
                assert_eq!(out.materialize_f64(), vec![3.0, 8.0 / 3.0, 2.5]);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    });
}

#[test]
fn gradient_gpu_mixed_scalar_and_coordinate_outputs_stay_resident() {
    test_support::with_test_provider(|provider| {
        let host = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &host.materialize_f64(),
            shape: host.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let spacing = Tensor::new(vec![0.0, 2.0], vec![2, 1]).unwrap();
        let _out_guard =
            crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = gradient_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(spacing),
                Value::Num(2.0),
            ]),
        )
        .expect("gradient");
        match result {
            Value::OutputList(outputs) => {
                assert!(matches!(outputs[0], Value::GpuTensor(_)));
                assert!(matches!(outputs[1], Value::GpuTensor(_)));
                let first = test_support::gather(outputs[0].clone()).expect("gather first");
                let second = test_support::gather(outputs[1].clone()).expect("gather second");
                assert_eq!(first.shape(), [2, 2]);
                assert_eq!(first.materialize_f64(), vec![0.5, 0.5, 0.5, 0.5]);
                assert_eq!(second.shape(), [2, 2]);
                assert_eq!(second.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0]);
            }
            other => panic!("expected output list, got {other:?}"),
        }
    });
}

#[test]
fn gradient_inprocess_complex_gpu_matches_cpu_and_stays_resident() {
    test_support::with_test_provider(|provider| {
        let host = ComplexTensor::new(
            vec![
                (1.0, 1.0),
                (2.0, -1.0),
                (4.0, 3.0),
                (6.0, 2.0),
                (9.0, 6.0),
                (12.0, 4.0),
            ],
            vec![2, 3],
        )
        .unwrap();
        let expected = gradient_complex_tensor_host(host.clone(), 2, 2.0).expect("cpu gradient");
        let handle = gpu_helpers::upload_complex_tensor(provider, &host).expect("upload");
        let result = gradient_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
        )
        .expect("gradient");
        let Value::GpuTensor(out_handle) = result else {
            panic!("expected complex gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(
            crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, NAME),
        )
        .expect("gather complex gradient");
        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn gradient_gpu_complex_matches_cpu_and_stays_resident() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let host = ComplexTensor::new(
        vec![
            (1.0, 1.0),
            (2.0, -1.0),
            (4.0, 3.0),
            (6.0, 2.0),
            (9.0, 6.0),
            (12.0, 4.0),
        ],
        vec![2, 3],
    )
    .unwrap();
    let expected = gradient_complex_tensor_host(host.clone(), 2, 2.0).expect("cpu gradient");
    let handle = gpu_helpers::upload_complex_tensor(provider, &host).expect("upload");
    let result = gradient_builtin(
        Value::GpuTensor(handle),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    )
    .expect("gradient");
    let Value::GpuTensor(out_handle) = result else {
        panic!("expected complex gpu tensor");
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&out_handle),
        GpuTensorStorage::ComplexInterleaved
    );
    let gathered =
        block_on(crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, NAME))
            .expect("gather complex gradient");
    assert_eq!(gathered.shape(), expected.shape());
    for (idx, (actual, expected)) in gathered
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64().iter())
        .enumerate()
    {
        assert!(
            (actual.0 - expected.0).abs() <= 1.0e-5,
            "real mismatch at {idx}: actual={} expected={}",
            actual.0,
            expected.0
        );
        assert!(
            (actual.1 - expected.1).abs() <= 1.0e-5,
            "imag mismatch at {idx}: actual={} expected={}",
            actual.1,
            expected.1
        );
    }
}
