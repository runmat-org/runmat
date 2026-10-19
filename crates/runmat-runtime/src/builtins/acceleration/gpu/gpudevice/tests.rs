use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, NumericStorage, Tensor};

fn call(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    block_on(gpu_device_builtin(args))
}

#[test]
fn gpu_device_without_provider_has_its_own_public_error() {
    let _guard = test_support::accel_test_lock();
    let error = call(Vec::new()).expect_err("no provider should be an error");
    assert_eq!(error.identifier(), GPU_DEVICE_ERROR_NO_PROVIDER.identifier);
    assert_eq!(error.to_string(), GPU_DEVICE_ERROR_NO_PROVIDER.message);
}

#[test]
fn gpu_device_provider_info_facade_is_gated() {
    test_support::with_test_provider(|_| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(gpu_device_builtin(Vec::new()))
            .expect_err("provider info is a RunMat extension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:GpuDeviceProviderInfoExtension")
        );
    });
}

#[test]
fn logical_extension_gate_keeps_runtime_precedence_over_arity() {
    let _guard = test_support::accel_test_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(gpu_device_builtin(vec![Value::Bool(true), Value::Num(1.0)]))
        .expect_err("logical selector extension must be checked first");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:GpuDeviceLogicalSelectorExtension")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_returns_struct() {
    test_support::with_test_provider(|_| {
        let value = call(Vec::new()).expect("gpuDevice");
        match value {
            Value::Struct(s) => {
                assert!(s.fields.contains_key("device_id"));
                assert!(s.fields.contains_key("index"));
                assert!(s.fields.contains_key("name"));
                assert!(s.fields.contains_key("vendor"));
                assert!(s.fields.contains_key("precision"));
                assert!(s.fields.contains_key("supports_double"));
            }
            other => panic!("expected struct, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_accepts_current_index() {
    test_support::with_test_provider(|_| {
        let tensor_scalar =
            runmat_value::Tensor::new(vec![1.0], vec![1, 1]).expect("scalar tensor");
        let logical_scalar = runmat_value::LogicalArray::new(vec![1u8], vec![1]).unwrap();
        let cases = vec![
            Value::Int(IntValue::I32(1)),
            Value::Num(1.0),
            Value::Bool(true),
            Value::Tensor(tensor_scalar),
            Value::LogicalArray(logical_scalar),
        ];
        for case in cases {
            let value = call(vec![case]).expect("gpuDevice");
            assert!(matches!(value, Value::Struct(_)));
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_index_parser_reads_typed_integer_storage_exactly() {
    test_support::with_test_provider(|_| {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).expect("integer tensor");

        let value = call(vec![Value::Tensor(tensor)]).expect("gpuDevice");
        assert!(matches!(value, Value::Struct(_)));

        let negative =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("integer tensor");
        let err = call(vec![Value::Tensor(negative)]).unwrap_err().to_string();
        assert_eq!(err, GPU_DEVICE_ERROR_INVALID_INDEX.message);
    });
}

#[test]
fn gpu_device_rejects_out_of_range_unsigned_scalar_without_signed_wrapping() {
    test_support::with_test_provider(|_| {
        let error = call(vec![Value::Int(IntValue::U64(u64::MAX))])
            .expect_err("out of range unsigned index");
        assert_eq!(
            error.identifier(),
            GPU_DEVICE_ERROR_INVALID_INDEX.identifier
        );
    });
}

#[test]
fn gpu_device_index_parser_reads_native_single_storage() {
    test_support::with_test_provider(|_| {
        let tensor = Tensor::from_numeric_storage(NumericStorage::F32(vec![1.0]), vec![1, 1])
            .expect("single tensor");
        let value = call(vec![Value::Tensor(tensor)]).expect("gpuDevice");
        assert!(matches!(value, Value::Struct(_)));
    });
}

#[test]
fn gpu_device_float_selector_uses_shared_numeric_boundary() {
    test_support::with_test_provider(|_| {
        assert!(matches!(
            call(vec![Value::Num(1.0 - 1e-10)]),
            Ok(Value::Struct(_))
        ));
        let error = call(vec![Value::Num(1.5)]).expect_err("nonintegral index");
        assert_eq!(
            error.identifier(),
            GPU_DEVICE_ERROR_INVALID_INDEX.identifier
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_index_parser_rejects_out_of_range_double_before_casting() {
    test_support::with_test_provider(|_| {
        let err = call(vec![Value::Num((u32::MAX as f64) + 1.0)]).unwrap_err();
        assert_eq!(err.to_string(), GPU_DEVICE_ERROR_INVALID_INDEX.message);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_out_of_range_index_errors() {
    test_support::with_test_provider(|_| {
        let err = call(vec![Value::Num(2.0)]).unwrap_err().to_string();
        assert!(
            err.contains("gpuDevice: GPU device with index 2 not available"),
            "unexpected error: {err}"
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_unsupported_argument_errors() {
    test_support::with_test_provider(|_| {
        let err = call(vec![Value::from("status")]).unwrap_err();
        assert_eq!(
            err.to_string(),
            GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT.message
        );
        assert_eq!(
            err.identifier(),
            GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT.identifier
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_reset_argument_reports_not_supported() {
    test_support::with_test_provider(|_| {
        let err = call(vec![Value::from(" RESET ")]).unwrap_err();
        assert_eq!(
            err.to_string(),
            GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED.message
        );
        assert_eq!(
            err.identifier(),
            GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED.identifier
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_reset_char_array_argument_reports_not_supported() {
    test_support::with_test_provider(|_| {
        let chars = runmat_value::CharArray::new("reset".chars().collect(), 1, 5).unwrap();
        let err = call(vec![Value::CharArray(chars)]).unwrap_err();
        assert_eq!(
            err.to_string(),
            GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED.message
        );
        assert_eq!(
            err.identifier(),
            GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED.identifier
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_empty_array_argument_reports_not_supported() {
    test_support::with_test_provider(|_| {
        let empty = runmat_value::Tensor::zeros(vec![0, 0]);
        let err = call(vec![Value::Tensor(empty)]).unwrap_err();
        assert_eq!(
            err.to_string(),
            GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED.message
        );
        assert_eq!(
            err.identifier(),
            GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED.identifier
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_device_invalid_index_rejected() {
    test_support::with_test_provider(|_| {
        let cases = vec![
            Value::Num(0.0),
            Value::Int(IntValue::I32(0)),
            Value::Num(-1.0),
            Value::Num(1.5),
            Value::Bool(false),
            Value::LogicalArray(runmat_value::LogicalArray::new(vec![0u8], vec![1]).unwrap()),
            Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        ];
        for case in cases {
            let err = call(vec![case]).unwrap_err().to_string();
            assert_eq!(err, GPU_DEVICE_ERROR_INVALID_INDEX.message);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
#[ignore = "requires an available WGPU adapter; run as a required hardware witness"]
fn gpu_device_wgpu_reports_metadata() {
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;

    let _provider =
        wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default())
            .expect("required WGPU adapter");
    let value = call(Vec::new()).expect("gpuDevice");
    match value {
        Value::Struct(info) => {
            let name = info
                .fields
                .get("name")
                .and_then(|v| match v {
                    Value::String(s) => Some(s),
                    _ => None,
                })
                .expect("name field");
            assert!(
                !name.is_empty(),
                "expected non-empty adapter name from wgpu provider"
            );

            let backend = info.fields.get("backend").and_then(|v| match v {
                Value::String(s) => Some(s),
                _ => None,
            });
            assert!(backend.is_some(), "expected backend field to be present");

            if let Some(Value::Int(memory)) = info.fields.get("memory_bytes") {
                assert!(
                    memory.to_i64() > 0,
                    "expected positive memory_bytes, got {:?}",
                    memory
                );
            }

            if let Some(Value::Bool(supports_double)) = info.fields.get("supports_double") {
                if *supports_double {
                    assert_eq!(
                        info.fields.get("precision"),
                        Some(&Value::String("double".to_string()))
                    );
                }
            }
        }
        other => panic!("expected struct, got {other:?}"),
    }
}
