//! MATLAB-compatible `uint8` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, IntValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "uint8";
const UINT8_MAX_F64: f64 = u8::MAX as f64;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::uint8")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "uint8",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "No provider-native uint8 hook yet; gpuArray inputs gather to host for saturating conversion and are then re-uploaded when possible.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::uint8")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "uint8",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Runs outside fusion today because integer storage remains host-represented in f64 buffers.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn conversion_error(type_name: &str) -> RuntimeError {
    builtin_error(format!(
        "uint8: conversion to uint8 from {type_name} is not possible"
    ))
}

#[runtime_builtin(
    name = "uint8",
    category = "math/elementwise",
    summary = "Convert scalars, arrays, and gpuArray values to uint8 using MATLAB saturating rounding.",
    keywords = "uint8,cast,integer,conversion,gpuArray",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::uint8"
)]
async fn uint8_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(builtin_error("uint8: too many input arguments"));
    }
    match value {
        Value::Num(n) => Ok(Value::Int(IntValue::U8(cast_scalar_to_uint8(n)))),
        Value::Int(i) => Ok(Value::Int(IntValue::U8(cast_scalar_to_uint8(i.to_f64())))),
        Value::Bool(flag) => Ok(Value::Int(IntValue::U8(if flag { 1 } else { 0 }))),
        Value::Tensor(tensor) => Ok(uint8_value_from_tensor(uint8_tensor_to_host(tensor))),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)
                .map_err(|e| builtin_error(format!("uint8: {e}")))?;
            Ok(uint8_value_from_tensor(uint8_tensor_to_host(tensor)))
        }
        Value::CharArray(chars) => uint8_from_char_array(chars),
        Value::GpuTensor(handle) => uint8_from_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(conversion_error("complex")),
        Value::String(_) | Value::StringArray(_) => Err(conversion_error("string")),
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) => Err(conversion_error("struct")),
        Value::Object(obj) => Err(conversion_error(&obj.class_name)),
        Value::HandleObject(handle) => Err(conversion_error(&handle.class_name)),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_) | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_) => Err(conversion_error("MException")),
        Value::OutputList(_) => Err(conversion_error("OutputList")),
    }
}

fn uint8_from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = chars
        .data
        .iter()
        .map(|&ch| cast_scalar_to_uint8(ch as u32 as f64) as f64)
        .collect();
    let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| builtin_error(format!("uint8: {e}")))?;
    Ok(uint8_value_from_tensor(tensor))
}

async fn uint8_from_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let converted = uint8_tensor_to_host(gpu_helpers::gather_tensor_async(&handle).await?);

    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        let _ = provider.free(&handle);
        let view = HostTensorView {
            data: &converted.data,
            shape: &converted.shape,
        };
        match provider.upload(&view) {
            Ok(new_handle) => return Ok(Value::GpuTensor(new_handle)),
            Err(err) => trace!("uint8: provider upload failed after gather ({err})"),
        }
    }

    Ok(uint8_value_from_tensor(converted))
}

fn uint8_tensor_to_host(mut tensor: Tensor) -> Tensor {
    for value in &mut tensor.data {
        *value = cast_scalar_to_uint8(*value) as f64;
    }
    tensor
}

fn uint8_value_from_tensor(tensor: Tensor) -> Value {
    if tensor.data.len() == 1 {
        Value::Int(IntValue::U8(cast_scalar_to_uint8(tensor.data[0])))
    } else {
        Value::Tensor(tensor)
    }
}

fn cast_scalar_to_uint8(value: f64) -> u8 {
    if value.is_nan() {
        return 0;
    }
    if value.is_infinite() {
        return if value.is_sign_negative() { 0 } else { u8::MAX };
    }
    value.round().clamp(0.0, UINT8_MAX_F64) as u8
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};

    fn uint8_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::uint8_builtin(value, rest))
    }

    #[test]
    fn uint8_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_scalar_saturates_and_rounds() {
        assert_eq!(
            uint8_builtin(Value::Num(3.5), Vec::new()).expect("uint8"),
            Value::Int(IntValue::U8(4))
        );
        assert_eq!(
            uint8_builtin(Value::Num(-1.0), Vec::new()).expect("uint8"),
            Value::Int(IntValue::U8(0))
        );
        assert_eq!(
            uint8_builtin(Value::Num(f64::INFINITY), Vec::new()).expect("uint8"),
            Value::Int(IntValue::U8(u8::MAX))
        );
        assert_eq!(
            uint8_builtin(Value::Num(f64::NAN), Vec::new()).expect("uint8"),
            Value::Int(IntValue::U8(0))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![-2.0, 2.49, 2.5, 300.0], vec![2, 2]).unwrap();
        let result = uint8_builtin(Value::Tensor(tensor), Vec::new()).expect("uint8");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0.0, 2.0, 3.0, u8::MAX as f64]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_char_array_produces_codes() {
        let chars = CharArray::new_row("Az");
        let result = uint8_builtin(Value::CharArray(chars), Vec::new()).expect("uint8");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![65.0, 122.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_errors_on_string_input() {
        let err = uint8_builtin(Value::String("hello".to_string()), Vec::new())
            .expect_err("expected error");
        assert!(err.message().contains("uint8"));
        assert!(err.message().contains("string"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn uint8_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-3.0, 4.4, 300.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = uint8_builtin(Value::GpuTensor(handle), Vec::new()).expect("uint8");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![3, 1]);
                    assert_eq!(gathered.data, vec![0.0, 4.0, u8::MAX as f64]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }
}
