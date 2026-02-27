//! MATLAB-compatible `int32` builtin with GPU-aware semantics for RunMat.

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

const BUILTIN_NAME: &str = "int32";
const INT32_MIN_F64: f64 = i32::MIN as f64;
const INT32_MAX_F64: f64 = i32::MAX as f64;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::int32")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "int32",
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
    notes: "No provider-native int32 hook yet; gpuArray inputs gather to host for saturating conversion and are then re-uploaded when possible.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::int32")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "int32",
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
        "int32: conversion to int32 from {type_name} is not possible"
    ))
}

#[runtime_builtin(
    name = "int32",
    category = "math/elementwise",
    summary = "Convert scalars, arrays, and gpuArray values to int32 using MATLAB saturating rounding.",
    keywords = "int32,cast,integer,conversion,gpuArray",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::int32"
)]
async fn int32_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(builtin_error("int32: too many input arguments"));
    }
    match value {
        Value::Num(n) => Ok(Value::Int(IntValue::I32(cast_scalar_to_int32(n)))),
        Value::Int(i) => Ok(Value::Int(IntValue::I32(cast_scalar_to_int32(i.to_f64())))),
        Value::Bool(flag) => Ok(Value::Int(IntValue::I32(if flag { 1 } else { 0 }))),
        Value::Tensor(tensor) => Ok(int32_value_from_tensor(int32_tensor_to_host(tensor))),
        Value::LogicalArray(array) => {
            let tensor = tensor::logical_to_tensor(&array)
                .map_err(|e| builtin_error(format!("int32: {e}")))?;
            Ok(int32_value_from_tensor(int32_tensor_to_host(tensor)))
        }
        Value::CharArray(chars) => int32_from_char_array(chars),
        Value::GpuTensor(handle) => int32_from_gpu(handle).await,
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

fn int32_from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = chars
        .data
        .iter()
        .map(|&ch| cast_scalar_to_int32(ch as u32 as f64) as f64)
        .collect();
    let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| builtin_error(format!("int32: {e}")))?;
    Ok(int32_value_from_tensor(tensor))
}

async fn int32_from_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let converted = int32_tensor_to_host(gpu_helpers::gather_tensor_async(&handle).await?);

    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        let _ = provider.free(&handle);
        let view = HostTensorView {
            data: &converted.data,
            shape: &converted.shape,
        };
        match provider.upload(&view) {
            Ok(new_handle) => return Ok(Value::GpuTensor(new_handle)),
            Err(err) => trace!("int32: provider upload failed after gather ({err})"),
        }
    }

    Ok(int32_value_from_tensor(converted))
}

fn int32_tensor_to_host(mut tensor: Tensor) -> Tensor {
    for value in &mut tensor.data {
        *value = cast_scalar_to_int32(*value) as f64;
    }
    tensor
}

fn int32_value_from_tensor(tensor: Tensor) -> Value {
    if tensor.data.len() == 1 {
        Value::Int(IntValue::I32(cast_scalar_to_int32(tensor.data[0])))
    } else {
        Value::Tensor(tensor)
    }
}

fn cast_scalar_to_int32(value: f64) -> i32 {
    if value.is_nan() {
        return 0;
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            i32::MIN
        } else {
            i32::MAX
        };
    }
    value.round().clamp(INT32_MIN_F64, INT32_MAX_F64) as i32
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};

    fn int32_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::int32_builtin(value, rest))
    }

    #[test]
    fn int32_type_preserves_tensor_shape() {
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
    fn int32_scalar_saturates_and_rounds() {
        assert_eq!(
            int32_builtin(Value::Num(3.5), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(4))
        );
        assert_eq!(
            int32_builtin(Value::Num(-3.5), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(-4))
        );
        assert_eq!(
            int32_builtin(Value::Num(f64::INFINITY), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(i32::MAX))
        );
        assert_eq!(
            int32_builtin(Value::Num(f64::NEG_INFINITY), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(i32::MIN))
        );
        assert_eq!(
            int32_builtin(Value::Num(f64::NAN), Vec::new()).expect("int32"),
            Value::Int(IntValue::I32(0))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![-2.0, 2.49, 2.5, 1.0e20], vec![2, 2]).unwrap();
        let result = int32_builtin(Value::Tensor(tensor), Vec::new()).expect("int32");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![-2.0, 2.0, 3.0, i32::MAX as f64]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_char_array_produces_codes() {
        let chars = CharArray::new_row("Az");
        let result = int32_builtin(Value::CharArray(chars), Vec::new()).expect("int32");
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
    fn int32_errors_on_string_input() {
        let err = int32_builtin(Value::String("hello".to_string()), Vec::new())
            .expect_err("expected error");
        assert!(err.message().contains("int32"));
        assert!(err.message().contains("string"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn int32_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-3.0, 4.4, 5.6], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = int32_builtin(Value::GpuTensor(handle), Vec::new()).expect("int32");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![3, 1]);
                    assert_eq!(gathered.data, vec![-3.0, 4.0, 6.0]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }
}
