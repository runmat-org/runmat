use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_accelerate_api::HostTensorView;

use runmat_value::{IntValue, IntegerStorage};

fn run_trapz(first: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    block_on(super::trapz_builtin(None, first, rest))
}

#[test]
fn trapz_scalar_is_zero() {
    let value = run_trapz(
        Value::Num(5.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("trapz");
    assert_eq!(value, Value::Num(0.0));
}

#[test]
fn trapz_row_vector_unit_spacing() {
    let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let value = run_trapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("trapz");
    assert_eq!(value, Value::Num(4.0));
}

#[test]
fn trapz_nonuniform_x_vector() {
    let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
    let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
    let value = run_trapz(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(y)]),
    )
    .expect("trapz");
    assert_eq!(value, Value::Num(3.5));
}

#[test]
fn trapz_reads_typed_integer_values_and_spacing_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3]).expect("x");
    let y = Tensor::new_integer(IntegerStorage::I16(vec![0, 1, 2]), vec![1, 3]).expect("y");

    let value = run_trapz(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(y)]),
    )
    .expect("trapz");

    assert_eq!(value, Value::Num(3.5));
}

#[test]
fn trapz_preserves_documented_single_output_class() {
    let y = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("single input");
    let value = run_trapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("trapz");
    let Value::Tensor(output) = value else {
        panic!("expected tensor output");
    };
    assert_eq!(
        output.into_numeric_storage().expect("single output"),
        runmat_value::NumericStorage::F32(vec![4.0])
    );
}

#[test]
fn trapz_integer_data_and_spacing_are_independently_compatibility_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let y = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3])
        .expect("integer samples");
    let data_error = run_trapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect_err("typed sample data is an extension");
    assert_eq!(
        data_error.identifier(),
        TRAPZ_INTEGER_DATA_EXTENSION.error_identifier
    );

    let x = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3])
        .expect("integer spacing");
    let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).expect("double samples");
    let spacing_error = run_trapz(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(y)]),
    )
    .expect_err("typed spacing is an extension");
    assert_eq!(
        spacing_error.identifier(),
        TRAPZ_INTEGER_SPACING_EXTENSION.error_identifier
    );
}

#[test]
fn trapz_rejects_inexact_wide_integer_before_floating_integration() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
        vec![1, 2],
    )
    .expect("wide samples");
    let error = run_trapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect_err("inexact binary64 conversion must reject");
    assert!(error
        .message
        .contains("must be exactly representable as double"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
}

#[test]
fn trapz_matrix_dimension_two() {
    let y = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let value = run_trapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            2,
        ))]),
    )
    .expect("trapz");
    let Value::Tensor(out) = value else {
        panic!("expected tensor result");
    };
    assert_eq!(out.shape(), &[2, 1]);
    assert_eq!(out.materialize_f64(), vec![4.0, 10.0]);
}

#[test]
fn trapz_complex_values() {
    let y = ComplexTensor::new(vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], vec![1, 3]).unwrap();
    let value = run_trapz(
        Value::ComplexTensor(y),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    )
    .expect("trapz");
    assert_eq!(value, Value::Complex(4.0, 4.0));
}

#[test]
fn trapz_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = TRAPZ_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Q = trapz(Y)"));
    assert!(labels.contains(&"Q = trapz(Y, dim)"));
    assert!(labels.contains(&"Q = trapz(X, Y)"));
    assert!(labels.contains(&"Q = trapz(X, Y, dim)"));
}

#[test]
fn trapz_descriptor_errors_have_stable_codes() {
    assert!(TRAPZ_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == TRAPZ_ERROR_INVALID_ARGUMENT.code));
    assert!(TRAPZ_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == TRAPZ_ERROR_INVALID_INPUT.code));
    assert!(TRAPZ_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == TRAPZ_ERROR_INTERNAL.code));
}

#[test]
fn trapz_invalid_dim_uses_descriptor_identifier() {
    let y = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let err = run_trapz(
        Value::Tensor(y),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::I32(
            0,
        ))]),
    )
    .expect_err("trapz");
    assert_eq!(err.identifier(), TRAPZ_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn trapz_too_many_inputs_uses_descriptor_identifier() {
    let err = run_trapz(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ]),
    )
    .expect_err("trapz");
    assert_eq!(err.identifier(), TRAPZ_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn trapz_gpu_input_preserves_real_result_residency() {
    test_support::with_test_provider(|provider| {
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &y.materialize_f64(),
                shape: y.shape(),
            })
            .expect("upload");
        let result = run_trapz(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("trapz gpu");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.shape(), &[1, 1]);
        assert_eq!(gathered.materialize_f64(), vec![4.0]);
    });
}

#[test]
fn trapz_complex_gpu_input_preserves_result_residency() {
    test_support::with_test_provider(|provider| {
        let y = ComplexTensor::new(vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], vec![1, 3]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &y)
            .expect("upload complex");
        provider.reset_telemetry();

        let result = run_trapz(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        )
        .expect("trapz complex gpu");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu result");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = block_on(provider.download(&out)).expect("download");
        assert_eq!(
            gathered.storage,
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(gathered.shape, vec![1, 1]);
        assert_eq!(gathered.data, vec![4.0, 4.0]);
        handle.release().unwrap();
        out.release().unwrap();
    });
}

#[test]
fn trapz_gpu_input_uses_vector_spacing_on_provider() {
    test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let handle = provider
            .upload(&HostTensorView {
                data: &y.materialize_f64(),
                shape: y.shape(),
            })
            .expect("upload y");
        let result = run_trapz(
            Value::Tensor(x),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle,
            )]),
        )
        .expect("trapz gpu");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.shape(), &[1, 1]);
        assert_eq!(gathered.materialize_f64(), vec![3.5]);
    });
}

#[test]
fn trapz_gpu_input_uses_tensor_spacing_on_provider() {
    test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![0.0, 0.0, 1.0, 1.0, 3.0, 3.0], vec![2, 3]).unwrap();
        let y = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let y_handle = provider
            .upload(&HostTensorView {
                data: &y.materialize_f64(),
                shape: y.shape(),
            })
            .expect("upload y");
        let x_handle = provider
            .upload(&HostTensorView {
                data: &x.materialize_f64(),
                shape: x.shape(),
            })
            .expect("upload x");
        let result = run_trapz(
            Value::GpuTensor(x_handle),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(y_handle),
                Value::Int(IntValue::I32(2)),
            ]),
        )
        .expect("trapz gpu");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu result");
        };
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.shape(), &[2, 1]);
        assert_eq!(gathered.materialize_f64(), vec![6.5, 15.5]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn trapz_wgpu_matches_cpu_vector_spacing() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    runmat_accelerate_api::set_thread_provider(Some(provider));
    let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
    let y = Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
    let cpu = run_trapz(
        Value::Tensor(x.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(y.clone())]),
    )
    .unwrap();
    let y_handle = provider
        .upload(&HostTensorView {
            data: &y.materialize_f64(),
            shape: y.shape(),
        })
        .unwrap();
    let gpu = run_trapz(
        Value::Tensor(x),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(y_handle)]),
    )
    .unwrap();
    let gathered = test_support::gather(gpu).expect("gather gpu");
    let expected = match cpu {
        Value::Num(value) => value,
        other => panic!("unexpected cpu result {other:?}"),
    };
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    assert_eq!(gathered.shape(), &[1, 1]);
    assert!((gathered.materialize_f64()[0] - expected).abs() < tol);
}
