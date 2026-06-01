//! MATLAB-compatible `gpuDevice` builtin with provider-aware semantics.

use runmat_accelerate_api::{ApiDeviceInfo, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, StructValue, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::acceleration::gpu::type_resolvers::gpudevice_type;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "gpuDevice";

const GPU_DEVICE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Struct describing active GPU provider/device metadata.",
}];

const GPU_DEVICE_INPUTS_EMPTY: [BuiltinParamDescriptor; 0] = [];

const GPU_DEVICE_INPUTS_INDEX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arg",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Device index selector or reset-like argument.",
}];

const GPU_DEVICE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "info = gpuDevice()",
        inputs: &GPU_DEVICE_INPUTS_EMPTY,
        outputs: &GPU_DEVICE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "info = gpuDevice(arg)",
        inputs: &GPU_DEVICE_INPUTS_INDEX,
        outputs: &GPU_DEVICE_OUTPUT,
    },
];

pub(crate) const GPU_DEVICE_ERROR_NO_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.NO_PROVIDER",
    identifier: Some("RunMat:gpuDevice:NoProvider"),
    when: "No acceleration provider is registered.",
    message: "gpuDevice: no acceleration provider registered",
};

const GPU_DEVICE_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.TOO_MANY_INPUTS",
    identifier: Some("RunMat:gpuDevice:TooManyInputs"),
    when: "More than one input argument was supplied.",
    message: "gpuDevice: too many input arguments",
};

const GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.UNSUPPORTED_ARGUMENT",
    identifier: Some("RunMat:gpuDevice:UnsupportedArgument"),
    when: "Argument does not map to a valid device selector/reset operation.",
    message: "gpuDevice: unsupported input argument",
};

const GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.RESET_NOT_SUPPORTED",
    identifier: Some("RunMat:gpuDevice:ResetNotSupported"),
    when: "Reset was requested but active provider does not support reset.",
    message: "gpuDevice: reset is not supported by the active provider",
};

const GPU_DEVICE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.INVALID_INDEX",
    identifier: Some("RunMat:gpuDevice:InvalidIndex"),
    when: "Device index is not a positive integer scalar.",
    message: "gpuDevice: device index must be a positive integer",
};

const GPU_DEVICE_ERROR_DEVICE_NOT_AVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.DEVICE_NOT_AVAILABLE",
    identifier: Some("RunMat:gpuDevice:DeviceNotAvailable"),
    when: "Requested device index does not match the currently active provider index.",
    message: "gpuDevice: requested device index is not available",
};

const GPU_DEVICE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    GPU_DEVICE_ERROR_NO_PROVIDER,
    GPU_DEVICE_ERROR_TOO_MANY_INPUTS,
    GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT,
    GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED,
    GPU_DEVICE_ERROR_INVALID_INDEX,
    GPU_DEVICE_ERROR_DEVICE_NOT_AVAILABLE,
];

pub const GPU_DEVICE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GPU_DEVICE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GPU_DEVICE_ERRORS,
};

fn gpu_device_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    gpu_device_error_with_message(error.message, error)
}

fn gpu_device_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::gpudevice")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gpuDevice",
    op_kind: GpuOpKind::Custom("device-info"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Pure metadata query; does not enqueue GPU kernels. Returns an error when no provider is registered.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::acceleration::gpu::gpudevice"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gpuDevice",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; the builtin returns a host-resident struct.",
};

/// Query the active provider and return a metadata struct describing the GPU device.
#[runtime_builtin(
    name = "gpuDevice",
    category = "acceleration/gpu",
    summary = "Query active GPU device metadata.",
    keywords = "gpu,gpuDevice,device,info,accelerate",
    examples = "info = gpuDevice();",
    type_resolver(gpudevice_type),
    descriptor(crate::builtins::acceleration::gpu::gpudevice::GPU_DEVICE_DESCRIPTOR),
    builtin_path = "crate::builtins::acceleration::gpu::gpudevice"
)]
async fn gpu_device_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.as_slice() {
        [] => active_device_struct().map(Value::Struct),
        [arg] => handle_single_argument(arg),
        _ => Err(gpu_device_error(&GPU_DEVICE_ERROR_TOO_MANY_INPUTS).into()),
    }
}

/// Internal helper that queries the provider and returns a populated struct.
pub(crate) fn active_device_struct() -> BuiltinResult<StructValue> {
    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| gpu_device_error(&GPU_DEVICE_ERROR_NO_PROVIDER))?;
    let info = provider.device_info_struct();
    let precision = provider.precision();
    Ok(build_struct(&info, precision))
}

fn build_struct(info: &ApiDeviceInfo, precision: ProviderPrecision) -> StructValue {
    let mut st = StructValue::new();
    st.insert("device_id", Value::Int(IntValue::U32(info.device_id)));
    st.insert(
        "index",
        Value::Int(IntValue::U32(info.device_id.saturating_add(1))),
    );
    st.insert("name", Value::String(info.name.clone()));
    st.insert("vendor", Value::String(info.vendor.clone()));
    if let Some(backend) = info.backend.as_ref() {
        st.insert("backend", Value::String(backend.clone()));
    }
    if let Some(bytes) = info.memory_bytes {
        st.insert("memory_bytes", Value::Int(IntValue::U64(bytes)));
    }
    st.insert(
        "precision",
        Value::String(
            match precision {
                ProviderPrecision::F64 => "double",
                ProviderPrecision::F32 => "single",
            }
            .to_string(),
        ),
    );
    st.insert(
        "supports_double",
        Value::Bool(matches!(precision, ProviderPrecision::F64)),
    );
    st
}

fn is_keyword(value: &Value, keyword: &str) -> bool {
    match value {
        Value::String(s) => s.trim().eq_ignore_ascii_case(keyword),
        Value::CharArray(ca) if ca.rows == 1 => {
            let collected: String = ca.data.iter().collect();
            collected.trim().eq_ignore_ascii_case(keyword)
        }
        _ => false,
    }
}

fn handle_single_argument(arg: &Value) -> BuiltinResult<Value> {
    if is_reset_arg(arg) {
        return Err(gpu_device_error(&GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED).into());
    }

    match parse_device_index(arg)? {
        Some(index) => {
            let info = active_device_struct()?;
            let current_index = struct_device_index(&info).unwrap_or(1);
            if index == current_index {
                Ok(Value::Struct(info))
            } else {
                Err(gpu_device_error_with_message(
                    format!("gpuDevice: GPU device with index {} not available", index),
                    &GPU_DEVICE_ERROR_DEVICE_NOT_AVAILABLE,
                )
                .into())
            }
        }
        None => Err(gpu_device_error(&GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT).into()),
    }
}

fn struct_device_index(info: &StructValue) -> Option<u32> {
    info.fields.get("index").and_then(|value| match value {
        Value::Int(intv) => {
            let idx = intv.to_i64();
            if idx >= 0 && idx <= u32::MAX as i64 {
                Some(idx as u32)
            } else {
                None
            }
        }
        Value::Num(n) if n.is_finite() && *n >= 0.0 => {
            let rounded = n.round();
            if (rounded - n).abs() <= 1e-9 {
                Some(rounded as u32)
            } else {
                None
            }
        }
        _ => None,
    })
}

fn is_reset_arg(value: &Value) -> bool {
    if is_keyword(value, "reset") {
        return true;
    }
    match value {
        Value::Tensor(t) => t.data.is_empty(),
        Value::LogicalArray(la) => la.data.is_empty(),
        _ => false,
    }
}

fn parse_device_index(value: &Value) -> BuiltinResult<Option<u32>> {
    match value {
        Value::Int(i) => int_to_index(i.to_i64()),
        Value::Num(n) => num_to_index(*n),
        Value::Bool(b) => {
            if *b {
                Ok(Some(1))
            } else {
                Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into())
            }
        }
        Value::Tensor(t) => match t.data.len() {
            0 => Ok(None),
            1 => num_to_index(t.data[0]),
            _ => Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into()),
        },
        Value::LogicalArray(la) => match la.data.len() {
            0 => Ok(None),
            1 => {
                if la.data[0] != 0 {
                    Ok(Some(1))
                } else {
                    Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into())
                }
            }
            _ => Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into()),
        },
        _ => Ok(None),
    }
}

fn int_to_index(raw: i64) -> BuiltinResult<Option<u32>> {
    if raw <= 0 {
        return Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into());
    }
    if raw > u32::MAX as i64 {
        return Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into());
    }
    Ok(Some(raw as u32))
}

fn num_to_index(raw: f64) -> BuiltinResult<Option<u32>> {
    if !raw.is_finite() {
        return Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into());
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-9 {
        return Err(gpu_device_error(&GPU_DEVICE_ERROR_INVALID_INDEX).into());
    }
    let idx = rounded as i64;
    int_to_index(idx)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};

    fn call(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(gpu_device_builtin(args))
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
                runmat_builtins::Tensor::new(vec![1.0], vec![1, 1]).expect("scalar tensor");
            let logical_scalar = runmat_builtins::LogicalArray::new(vec![1u8], vec![1]).unwrap();
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
            let chars = runmat_builtins::CharArray::new("reset".chars().collect(), 1, 5).unwrap();
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
            let empty = runmat_builtins::Tensor::zeros(vec![0, 0]);
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
                Value::LogicalArray(
                    runmat_builtins::LogicalArray::new(vec![0u8], vec![1]).unwrap(),
                ),
                Value::Tensor(runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            ];
            for case in cases {
                let err = call(vec![case]).unwrap_err().to_string();
                assert_eq!(err, GPU_DEVICE_ERROR_INVALID_INDEX.message);
            }
        });
    }

    #[test]
    fn gpudevice_type_is_struct() {
        assert!(matches!(
            gpudevice_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Struct { .. }
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn gpu_device_wgpu_reports_metadata() {
        use runmat_accelerate::backend::wgpu::provider as wgpu_provider;

        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
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
}
