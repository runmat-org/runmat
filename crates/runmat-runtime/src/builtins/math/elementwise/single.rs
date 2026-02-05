//! MATLAB-compatible `single` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    random_args::keyword_of,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
        FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
        ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};

use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::single")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "single",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_single",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Casts tensors to float32. Providers may implement `unary_single`; otherwise the runtime gathers, converts, and re-uploads to keep gpuArray results resident.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::single")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "single",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            match ctx.scalar_ty {
                ScalarType::F32 => Ok(input.to_string()),
                ScalarType::F64 => Ok(format!("f64(f32({input}))")),
                _ => Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            }
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits a cast via `f32` then widens back to the WGSL scalar type when necessary.",
};

const BUILTIN_NAME: &str = "single";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn conversion_error(type_name: &str) -> RuntimeError {
    builtin_error(format!(
        "single: conversion to single from {type_name} is not possible"
    ))
}

#[runtime_builtin(
    name = "single",
    category = "math/elementwise",
    summary = "Convert scalars, arrays, and gpuArray values to single precision.",
    keywords = "single,float32,cast,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::single"
)]
async fn single_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    let converted = match value {
        Value::Num(n) => Ok(Value::Num(cast_f64_to_single(n))),
        Value::Int(i) => Ok(Value::Num(cast_f64_to_single(i.to_f64()))),
        Value::Bool(flag) => Ok(Value::Num(if flag { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) => single_from_tensor(tensor),
        Value::Complex(re, im) => Ok(Value::Complex(
            cast_f64_to_single(re),
            cast_f64_to_single(im),
        )),
        Value::ComplexTensor(tensor) => single_from_complex_tensor(tensor),
        Value::LogicalArray(array) => single_from_logical_array(array),
        Value::CharArray(chars) => single_from_char_array(chars),
        Value::GpuTensor(handle) => single_from_gpu(handle).await,
        Value::String(_) | Value::StringArray(_) => Err(conversion_error("string")),
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) => Err(conversion_error("struct")),
        Value::Object(obj) => Err(conversion_error(&obj.class_name)),
        Value::HandleObject(handle) => Err(conversion_error(&handle.class_name)),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_) | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_) => Err(conversion_error("MException")),
    }?;
    apply_output_template(converted, &template).await
}

fn single_from_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    single_tensor_to_host(tensor).map(Value::Tensor)
}

fn single_from_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    single_complex_tensor_to_host(tensor).map(Value::ComplexTensor)
}

fn single_from_logical_array(array: LogicalArray) -> BuiltinResult<Value> {
    let tensor =
        tensor::logical_to_tensor(&array).map_err(|e| builtin_error(format!("single: {e}")))?;
    single_tensor_to_host(tensor).map(Value::Tensor)
}

fn single_from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let tensor = char_array_to_tensor(&chars)?;
    single_tensor_to_host(tensor).map(Value::Tensor)
}

async fn single_from_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match provider.unary_single(&handle).await {
            Ok(result) => {
                let _ = provider.free(&handle);
                return Ok(Value::GpuTensor(result));
            }
            Err(err) => {
                trace!("single: provider unary_single hook unavailable, falling back ({err})");
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let converted = single_tensor_to_host(tensor)?;
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        let _ = provider.free(&handle);
        let view = HostTensorView {
            data: &converted.data,
            shape: &converted.shape,
        };
        match provider.upload(&view) {
            Ok(new_handle) => {
                return Ok(Value::GpuTensor(new_handle));
            }
            Err(err) => {
                trace!("single: provider upload failed after gather ({err})");
            }
        }
    }
    Ok(tensor::tensor_into_value(converted))
}

fn single_tensor_to_host(mut tensor: Tensor) -> BuiltinResult<Tensor> {
    cast_slice_to_single(&mut tensor.data);
    tensor.dtype = NumericDType::F32;
    Ok(tensor)
}

fn single_complex_tensor_to_host(mut tensor: ComplexTensor) -> BuiltinResult<ComplexTensor> {
    cast_complex_slice_to_single(&mut tensor.data);
    Ok(tensor)
}

fn cast_slice_to_single(data: &mut [f64]) {
    for value in data.iter_mut() {
        *value = (*value as f32) as f64;
    }
}

fn cast_complex_slice_to_single(data: &mut [(f64, f64)]) {
    for (re, im) in data.iter_mut() {
        *re = (*re as f32) as f64;
        *im = (*im as f32) as f64;
    }
}

fn cast_f64_to_single(value: f64) -> f64 {
    (value as f32) as f64
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let ascii: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(ascii, vec![chars.rows, chars.cols])
        .map_err(|e| builtin_error(format!("single: {e}")))
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err(builtin_error("single: expected prototype after 'like'"))
            } else {
                Err(builtin_error("single: unrecognised argument for single"))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(builtin_error(
                    "single: unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(builtin_error("single: too many input arguments")),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(_) => convert_to_gpu(value),
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value).await,
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(builtin_error(
                "single: complex prototypes for 'like' are not supported yet",
            )),
            _ => Err(builtin_error(
                "single: unsupported prototype for 'like'; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

fn convert_to_gpu(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        builtin_error(
            "single: GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider
                .upload(&view)
                .map_err(|e| builtin_error(format!("single: {e}")))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("single: {e}")))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(builtin_error(
            "single: GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(builtin_error(format!(
            "single: unsupported result type for GPU output via 'like' ({other:?})"
        ))),
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy)
                .await
                .map_err(|e| builtin_error(format!("single: {e}")))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};

    fn single_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::single_builtin(value, rest))
    }

    #[test]
    fn single_type_preserves_tensor_shape() {
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

    #[test]
    fn single_type_scalar_tensor_returns_num() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_scalar_rounds_to_f32() {
        let value = Value::Num(std::f64::consts::PI);
        let result = single_builtin(value, Vec::new()).expect("single");
        match result {
            Value::Num(n) => assert_eq!(n, (std::f64::consts::PI as f32) as f64),
            other => panic!("expected scalar Num, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![1.25, 2.5, 3.75, 4.5], vec![2, 2]).unwrap();
        let result = single_builtin(Value::Tensor(tensor), Vec::new()).expect("single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected: Vec<f64> = [1.25f32, 2.5, 3.75, 4.5]
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                assert_eq!(t.data, expected);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_complex_tensor_rounds_both_components() {
        let tensor = ComplexTensor::new(
            vec![(1.234567, -9.876543), (0.3333333, 0.6666667)],
            vec![1, 2],
        )
        .unwrap();
        let result = single_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("single");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<(f64, f64)> = vec![
                    ((1.234567f32) as f64, (-9.876543f32) as f64),
                    ((0.3333333f32) as f64, (0.6666667f32) as f64),
                ];
                assert_eq!(t.data, expected);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_char_array_produces_codes() {
        let chars = CharArray::new_row("AZ");
        let result = single_builtin(Value::CharArray(chars), Vec::new()).expect("single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![65.0, 90.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_errors_on_string_input() {
        let err = single_builtin(Value::String("hello".to_string()), Vec::new())
            .expect_err("expected error");
        assert!(err.message().contains("single"));
        assert!(err.message().contains("string"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = single_builtin(Value::GpuTensor(handle), Vec::new()).expect("single");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> = [0.1f32, 0.2, 0.3, 0.4]
                        .into_iter()
                        .map(|v| v as f64)
                        .collect();
                    assert_eq!(gathered.shape, vec![2, 2]);
                    assert_eq!(gathered.data, expected);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_like_host_prototype() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = single_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("like"), Value::Num(0.0)],
        )
        .expect("single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, tensor.shape);
                let expected: Vec<f64> = [1.0f32, 2.0, 3.0, 4.0]
                    .into_iter()
                    .map(|v| v as f64)
                    .collect();
                assert_eq!(t.data, expected);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto_handle = provider.upload(&proto_view).expect("upload");
            let result = single_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto_handle)],
            )
            .expect("single");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> = [0.5f32, 1.5, 2.5, 3.5]
                        .into_iter()
                        .map(|v| v as f64)
                        .collect();
                    assert_eq!(gathered.shape, vec![2, 2]);
                    assert_eq!(gathered.data, expected);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_like_case_insensitive() {
        let tensor = Tensor::new(vec![1.5, 2.25], vec![2, 1]).unwrap();
        let result = single_builtin(
            Value::Tensor(tensor.clone()),
            vec![
                Value::CharArray(CharArray::new_row("LIKE")),
                Value::Num(0.0),
            ],
        )
        .expect("single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, tensor.shape);
                let expected: Vec<f64> = [1.5f32, 2.25f32].into_iter().map(|v| v as f64).collect();
                assert_eq!(t.data, expected);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_large_ones_preserve_values() {
        // Create a large ones tensor and ensure single() preserves ones exactly.
        let m = 200_000usize;
        let tensor = Tensor::ones(vec![m, 1]);
        let result = single_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, tensor.shape);
                // Sum should equal m exactly.
                let sum: f64 = t.data.iter().copied().sum();
                assert!(
                    (sum - (m as f64)).abs() < 1e-9,
                    "sum expected {m}, got {sum}"
                );
                // All entries must be exactly 1.0
                assert!(t.data.iter().all(|&v| v == 1.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn single_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.2, 1.8, -3.4, 7.25], vec![2, 2]).unwrap();
        let cpu = single_tensor_to_host(tensor.clone()).expect("cpu conversion");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(single_from_gpu(handle)).expect("gpu single");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn single_wgpu_large_ones_all_ones() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let m = 200_000usize;
        let tensor = Tensor::ones(vec![m, 1]);
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = block_on(single_from_gpu(handle)).expect("gpu single");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, tensor.shape);
        let sum: f64 = gathered.data.iter().copied().sum();
        assert!(
            (sum - (m as f64)).abs() < 1e-9,
            "sum expected {} got {}",
            m,
            sum
        );
        assert!(gathered.data.iter().all(|&v| v == 1.0));
    }
}
