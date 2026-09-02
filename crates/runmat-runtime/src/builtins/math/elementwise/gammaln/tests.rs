use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{ComplexTensor, IntValue, IntegerStorage, LogicalArray, SparseTensor};

fn call(value: Value) -> BuiltinResult<Value> {
    block_on(gammaln_builtin(value))
}

#[test]
fn gammaln_rejects_excess_outputs() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = call(Value::Num(1.0)).expect_err("second output must reject");
    assert_eq!(
        error.identifier(),
        GAMMALN_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}

fn approx_eq(got: f64, expected: f64, tol: f64) {
    assert!(
        (got - expected).abs() <= tol,
        "got {got}, expected {expected}, tol {tol}"
    );
}

fn values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_scalar_values() {
    match call(Value::Num(1.0)).expect("gammaln") {
        Value::Num(v) => approx_eq(v, 0.0, 1e-14),
        other => panic!("expected scalar result, got {other:?}"),
    }
    match call(Value::Num(5.0)).expect("gammaln") {
        Value::Num(v) => approx_eq(v, 24.0_f64.ln(), 1e-13),
        other => panic!("expected scalar result, got {other:?}"),
    }
    match call(Value::Num(0.5)).expect("gammaln") {
        Value::Num(v) => approx_eq(v, std::f64::consts::PI.sqrt().ln(), 1e-14),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_avoids_overflow_for_large_values() {
    match call(Value::Num(171.0)).expect("gammaln") {
        Value::Num(v) => {
            assert!(v.is_finite());
            approx_eq(v, 706.573_062_245_787_5, 1e-10);
        }
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_tiny_positive_values_use_log_asymptote() {
    let tiny = f64::MIN_POSITIVE / 2.0;
    match call(Value::Num(tiny)).expect("gammaln") {
        Value::Num(v) => approx_eq(v, -tiny.ln(), 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_tensor_shape_and_single_dtype() {
    let tensor =
        Tensor::new_with_dtype(vec![0.5, 1.0, 2.0, 5.0], vec![2, 2], NumericDType::F32).unwrap();
    let result = call(Value::Tensor(tensor)).expect("gammaln");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.numeric_dtype(), NumericDType::F32);
            let values = values_f64(&t);
            approx_eq(values[0], std::f32::consts::PI.sqrt().ln() as f64, 1e-7);
            approx_eq(values[1], 0.0, 1e-7);
            approx_eq(values[2], 0.0, 1e-7);
            approx_eq(values[3], 24.0_f32.ln() as f64, 1e-6);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_integer_bool_logical_and_char_promote() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    match call(Value::Int(IntValue::I32(5))).expect("gammaln") {
        Value::Num(v) => approx_eq(v, 24.0_f64.ln(), 1e-13),
        other => panic!("expected scalar result, got {other:?}"),
    }
    match call(Value::Bool(true)).expect("gammaln") {
        Value::Num(v) => approx_eq(v, 0.0, 1e-14),
        other => panic!("expected scalar result, got {other:?}"),
    }

    let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
    match call(Value::LogicalArray(logical)).expect("gammaln") {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let values = values_f64(&t);
            approx_eq(values[0], 0.0, 1e-14);
            assert_eq!(values[1], f64::INFINITY);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }

    let chars = CharArray::new(vec!['\0', '\u{1}'], 1, 2).unwrap();
    match call(Value::CharArray(chars)).expect("gammaln") {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let values = values_f64(&t);
            assert_eq!(values[0], f64::INFINITY);
            approx_eq(values[1], 0.0, 1e-14);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_reads_typed_integer_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let scalar = Tensor::new_integer(IntegerStorage::U16(vec![5]), vec![1, 1]).expect("int tensor");
    match call(Value::Tensor(scalar)).expect("gammaln") {
        Value::Num(v) => approx_eq(v, 24.0_f64.ln(), 1e-13),
        other => panic!("expected scalar result, got {other:?}"),
    }

    let tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![5, 1]), vec![1, 2]).expect("int tensor");
    match call(Value::Tensor(tensor)).expect("gammaln") {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert_eq!(t.numeric_dtype(), NumericDType::F64);
            let values = values_f64(&t);
            approx_eq(values[0], 24.0_f64.ln(), 1e-13);
            approx_eq(values[1], 0.0, 1e-14);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn gammaln_integer_and_logical_extensions_are_independently_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = call(Value::Int(IntValue::I16(5))).unwrap_err();
    assert_eq!(
        integer.identifier(),
        Some("RunMat:compatibility:GammalnIntegerInputExtension")
    );
    let logical = call(Value::Bool(true)).unwrap_err();
    assert_eq!(
        logical.identifier(),
        Some("RunMat:compatibility:GammalnLogicalInputExtension")
    );
    let character = call(Value::CharArray(
        CharArray::new(vec!['A'], 1, 1).expect("character"),
    ))
    .unwrap_err();
    assert_eq!(
        character.identifier(),
        Some("RunMat:compatibility:GammalnCharacterInputExtension")
    );
}

#[test]
fn gammaln_integer_extension_covers_all_classes_and_exact_binary64_boundary() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = [
        IntegerStorage::I8(vec![1, 5]),
        IntegerStorage::I16(vec![1, 5]),
        IntegerStorage::I32(vec![1, 5]),
        IntegerStorage::I64(vec![1, 5]),
        IntegerStorage::U8(vec![1, 5]),
        IntegerStorage::U16(vec![1, 5]),
        IntegerStorage::U32(vec![1, 5]),
        IntegerStorage::U64(vec![1, 5]),
    ];
    for storage in storages {
        let tensor = Tensor::new_integer(storage, vec![1, 2]).unwrap();
        let Value::Tensor(output) = call(Value::Tensor(tensor)).expect("integer gammaln") else {
            panic!("expected tensor output")
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F64);
    }

    let exact = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1]).unwrap();
    call(Value::Tensor(exact)).expect("exact wide power of two");
    let inexact =
        Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();
    let error = call(Value::Tensor(inexact)).unwrap_err();
    assert!(error.message().contains("exactly representable as double"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_nan_zero_and_infinity() {
    match call(Value::Num(0.0)).expect("gammaln") {
        Value::Num(v) => assert_eq!(v, f64::INFINITY),
        other => panic!("expected scalar result, got {other:?}"),
    }
    match call(Value::Num(f64::INFINITY)).expect("gammaln") {
        Value::Num(v) => assert_eq!(v, f64::INFINITY),
        other => panic!("expected scalar result, got {other:?}"),
    }
    match call(Value::Num(f64::NAN)).expect("gammaln") {
        Value::Num(v) => assert!(v.is_nan()),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_rejects_negative_complex_string_and_sparse_inputs() {
    let err = call(Value::Num(-0.5)).expect_err("negative should error");
    assert_eq!(err.identifier(), GAMMALN_ERROR_DOMAIN.identifier);

    let err = call(Value::Complex(1.0, 1.0)).expect_err("complex should error");
    assert_eq!(err.identifier(), GAMMALN_ERROR_INVALID_INPUT.identifier);

    let complex = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
    let err = call(Value::ComplexTensor(complex)).expect_err("complex should error");
    assert_eq!(err.identifier(), GAMMALN_ERROR_INVALID_INPUT.identifier);

    let err = call(Value::from("1")).expect_err("string should error");
    assert_eq!(err.identifier(), GAMMALN_ERROR_INVALID_INPUT.identifier);

    let sparse = SparseTensor::zeros(2, 2);
    let err = call(Value::SparseTensor(sparse)).expect_err("sparse should error");
    assert_eq!(err.identifier(), GAMMALN_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.5, 1.0, 2.0, 5.0, 171.0], vec![1, 5]).unwrap();
        let view = HostTensorView {
            data: tensor.as_f64_slice().expect("double input"),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = call(Value::GpuTensor(handle)).expect("gammaln");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 5]);
        for (got, input) in gathered
            .materialize_f64()
            .iter()
            .zip(tensor.as_f64_slice().expect("double input"))
        {
            approx_eq(*got, gammaln_nonnegative_scalar(*input), 1e-10);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gammaln_gpu_negative_errors() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, -0.5], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: tensor.as_f64_slice().expect("double input"),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let err = call(Value::GpuTensor(handle)).expect_err("negative gpu should error");
        assert_eq!(err.identifier(), GAMMALN_ERROR_DOMAIN.identifier);
    });
}

#[test]
fn gammaln_resident_integer_gate_precedes_provider_and_restores_double_output() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![1, 5]), vec![1, 2])
            .expect("integer input");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
            .expect("integer upload");
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = call(Value::GpuTensor(handle.clone())).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:GammalnIntegerInputExtension")
            );
            assert!(runmat_accelerate_api::provider_for_handle(&handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider)));
        }
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let Value::GpuTensor(output) = call(Value::GpuTensor(handle)).expect("gammaln") else {
            panic!("expected resident output")
        };
        assert!(runmat_accelerate_api::handle_integer_type(&output).is_none());
        assert!(runmat_accelerate_api::provider_for_handle(&output)
            .is_some_and(|owner| std::ptr::eq(owner, provider)));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn gammaln_wgpu_matches_cpu_elementwise() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let tensor = Tensor::new(vec![0.25, 0.5, 1.0, 2.0, 5.0, 32.0, 171.0], vec![1, 7]).unwrap();
    let cpu = match gammaln_tensor(tensor.clone()).expect("cpu gammaln") {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor result, got {other:?}"),
    };
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let view = HostTensorView {
        data: tensor.as_f64_slice().expect("double input"),
        shape: &tensor.shape,
    };
    let handle = provider.upload(&view).expect("upload");
    let gpu_value = block_on(gammaln_gpu(handle)).expect("gpu gammaln");
    let gathered = test_support::gather(gpu_value).expect("gather");
    assert_eq!(gathered.shape, cpu.shape);
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
        runmat_accelerate_api::ProviderPrecision::F32 => 2e-4,
    };
    for (got, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(cpu.as_f64_slice().expect("double cpu result"))
    {
        approx_eq(*got, *expected, tol);
    }
}
