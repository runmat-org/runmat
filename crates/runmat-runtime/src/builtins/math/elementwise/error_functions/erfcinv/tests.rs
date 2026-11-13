use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{GpuHandleProvenance, HostTensorView};
use runmat_value::{CharArray, ComplexTensor, IntValue, LogicalArray, SparseTensor};

fn erfcinv_builtin(value: Value) -> BuiltinResult<Value> {
    block_on(super::erfcinv_builtin(
        value,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
}

fn assert_close(actual: f64, expected: f64, tol: f64) {
    if actual.is_nan() && expected.is_nan() {
        return;
    }
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn scalar_values_match_reference_points() {
    match erfcinv_builtin(Value::Num(1.0)).unwrap() {
        Value::Num(value) => assert_eq!(value, 0.0),
        other => panic!("expected scalar, got {other:?}"),
    }
    let cases = [
        (0.3, 0.732_869_077_959_216_6, 2e-14),
        (0.5, 0.476_936_276_204_469_8, 2e-14),
        (1.5, -0.476_936_276_204_469_8, 2e-14),
        (0.999_999_999_999, 8.862_073_205_887_489e-13, 2e-16),
        (1.000_000_000_001, -8.863_057_115_425_171e-13, 2e-16),
        (1e-100, 15.065_574_702_592_645, 5e-13),
        (f64::MIN_POSITIVE, 26.543_258_454_250_98, 5e-13),
    ];
    for (input, expected, tol) in cases {
        assert_close(erfcinv_scalar(input), expected, tol);
    }
}

#[test]
fn tiny_tail_inputs_remain_ordered_and_finite() {
    let realmin = erfcinv_scalar(f64::MIN_POSITIVE);
    let subnormal = erfcinv_scalar(f64::from_bits(1));
    assert!(realmin.is_finite());
    assert!(subnormal.is_finite());
    assert!(subnormal > realmin);
    assert!(subnormal < MAX_POSITIVE_RESULT);
}

#[test]
fn endpoints_and_out_of_domain_match_matlab_shape() {
    assert!(erfcinv_scalar(0.0).is_infinite() && erfcinv_scalar(0.0).is_sign_positive());
    assert!(erfcinv_scalar(2.0).is_infinite() && erfcinv_scalar(2.0).is_sign_negative());
    assert!(erfcinv_scalar(-0.1).is_nan());
    assert!(erfcinv_scalar(2.1).is_nan());
    assert!(erfcinv_scalar(f64::NAN).is_nan());
}

#[test]
fn tensor_preserves_shape_and_single_dtype() {
    let tensor =
        Tensor::new_with_dtype(vec![0.5, 1.0, 1.5, 2.5], vec![2, 2], NumericDType::F32).unwrap();
    match erfcinv_builtin(Value::Tensor(tensor)).unwrap() {
        Value::Tensor(out) => {
            assert_eq!(out.shape(), &[2, 2]);
            assert_eq!(out.numeric_dtype(), NumericDType::F32);
            let NumericStorage::F32(values) = out.into_numeric_storage().expect("single storage")
            else {
                panic!("expected native single storage");
            };
            assert_close(libm::erfc(f64::from(values[0])), 0.5, 1e-6);
            assert_eq!(values[1], 0.0);
            assert_close(libm::erfc(f64::from(values[2])), 1.5, 1e-6);
            assert!(values[3].is_nan());
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn erfcinv_preserves_rank_zero_and_rank_one_single_tensors() {
    for (shape, input) in [(Vec::new(), vec![1.0_f32]), (vec![2], vec![0.5_f32, 1.5])] {
        let tensor = Tensor::from_f32(input.clone(), shape.clone()).expect("low-rank single");
        let Value::Tensor(output) = erfcinv_builtin(Value::Tensor(tensor)).expect("erfcinv") else {
            panic!("expected native single tensor");
        };
        assert_eq!(output.shape(), shape.as_slice());
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        for (actual, input) in output.materialize_f64().iter().zip(input) {
            assert_close(libm::erfc(*actual), f64::from(input), 1e-6);
        }
    }
}

#[test]
fn rejects_logical_integer_and_char_inputs() {
    assert!(erfcinv_builtin(Value::Bool(true)).is_err());
    assert!(erfcinv_builtin(Value::Int(IntValue::U8(2))).is_err());
    let chars = CharArray::from_unicode_scalars(vec!['\0', '\u{1}'], 1, 2).unwrap();
    assert!(erfcinv_builtin(Value::CharArray(chars)).is_err());
}

#[test]
fn rejects_complex_string_and_sparse_inputs() {
    assert!(erfcinv_builtin(Value::Complex(1.0, 1.0)).is_err());
    let complex = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
    assert!(erfcinv_builtin(Value::ComplexTensor(complex)).is_err());
    assert!(erfcinv_builtin(Value::String("1".to_string().into())).is_err());
    let sparse = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
    assert!(erfcinv_builtin(Value::SparseTensor(sparse)).is_err());
}

#[test]
fn rejects_logical_and_integer_class_arrays() {
    let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
    assert!(erfcinv_builtin(Value::LogicalArray(logical)).is_err());
    let ints = Tensor::new_with_dtype(vec![0.0, 1.0], vec![1, 2], NumericDType::U8).unwrap();
    assert!(erfcinv_builtin(Value::Tensor(ints)).is_err());
}

#[test]
fn gpu_roundtrip_matches_cpu() {
    test_support::with_test_provider(|provider| {
        let host = Tensor::new(vec![0.25, 0.75, 1.25, 1.75], vec![2, 2]).unwrap();
        let cpu = match erfcinv_builtin(Value::Tensor(host.clone())).unwrap() {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected cpu tensor, got {other:?}"),
        };
        let view = HostTensorView {
            data: host.as_f64_slice().expect("double host"),
            shape: host.shape(),
        };
        let handle = provider
            .upload(&view)
            .expect("upload")
            .with_provenance(GpuHandleProvenance::Explicit)
            .expect("explicit view");
        let resident = erfcinv_builtin(Value::GpuTensor(handle)).unwrap();
        let Value::GpuTensor(output) = &resident else {
            panic!("expected gpu tensor, got {resident:?}");
        };
        assert_eq!(
            runmat_accelerate_api::handle_provenance(output),
            Some(GpuHandleProvenance::Explicit)
        );
        let gpu = test_support::gather(resident).expect("gather");
        assert_eq!(gpu.shape(), cpu.shape());
        for (actual, expected) in gpu
            .as_f64_slice()
            .expect("double gpu result")
            .iter()
            .zip(cpu.as_f64_slice().expect("double cpu result"))
        {
            assert_close(*actual, *expected, 1e-12);
        }
    });
}

#[test]
fn gpu_rejects_integer_and_logical_storage_before_provider_dispatch() {
    test_support::with_test_provider(|provider| {
        let integer =
            Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap();
        let integer_handle =
            gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
        let integer_error = erfcinv_builtin(Value::GpuTensor(integer_handle)).unwrap_err();
        assert_eq!(
            integer_error.identifier(),
            ERFCINV_ERROR_INVALID_INPUT.identifier
        );

        let logical_handle = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1],
                shape: &[1, 2],
            })
            .expect("logical upload");
        let weak = logical_handle.downgrade();
        provider.reset_telemetry();
        let logical_error = erfcinv_builtin(Value::GpuTensor(logical_handle.clone()))
            .expect_err("logical GPU input");
        assert_eq!(
            logical_error.identifier(),
            ERFCINV_ERROR_INVALID_INPUT.identifier
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(runmat_accelerate_api::handle_is_logical(&logical_handle));
        logical_handle
            .allocation_lease()
            .validate_live()
            .expect("caller input survives rejection");
        drop(logical_error);
        logical_handle.release().expect("release logical caller");
        assert!(weak.acquire().is_err());
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn wgpu_provider_keeps_erfcinv_resident() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let tensor = Tensor::new(vec![0.25, 0.5, 1.0, 1.5, 1.75], vec![1, 5]).unwrap();
    let cpu = erfcinv_tensor(tensor.clone()).expect("cpu erfcinv");
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let view = HostTensorView {
        data: tensor.as_f64_slice().expect("double tensor"),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let gpu_value = block_on(super::erfcinv_gpu(handle)).expect("gpu erfcinv");
    assert!(
        matches!(gpu_value, Value::GpuTensor(_)),
        "erfcinv should keep WGPU provider results resident"
    );
    let gathered = test_support::gather(gpu_value).expect("gather");
    assert_eq!(gathered.shape(), cpu.shape());
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-8,
        runmat_accelerate_api::ProviderPrecision::F32 => 2e-4,
    };
    for (actual, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(cpu.materialize_f64().iter())
    {
        assert_close(*actual, *expected, tol);
    }
}
