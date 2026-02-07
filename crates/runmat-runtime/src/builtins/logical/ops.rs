//! MATLAB-compatible `logical` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{self, AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    CharArray, ComplexTensor, LogicalArray, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    shape::{canonical_scalar_shape, normalize_scalar_shape},
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};
use crate::builtins::logical::type_resolvers::logical_like;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::ops")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "logical",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_ne",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Preferred path issues elem_ne(X, 0) on the device; missing hooks trigger a gather → host cast → re-upload sequence flagged as logical.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::ops")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "logical",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion support will arrive alongside a dedicated WGSL template; today the builtin executes outside fusion plans.",
};

const BUILTIN_NAME: &str = "logical";

fn logical_type(args: &[Type], _context: &ResolveContext) -> Type {
    args.first().map(logical_like).unwrap_or(Type::logical())
}

fn logical_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "logical",
    category = "logical",
    summary = "Convert scalars, arrays, and gpuArray values to logical outputs.",
    keywords = "logical,boolean,gpuArray,mask,conversion",
    accel = "unary",
    type_resolver(logical_type),
    builtin_path = "crate::builtins::logical::ops"
)]
async fn logical_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(logical_error("logical: too many input arguments"));
    }
    convert_value_to_logical(value).await
}

async fn convert_value_to_logical(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => Ok(value),
        Value::Num(n) => Ok(Value::Bool(n != 0.0)),
        Value::Int(i) => Ok(Value::Bool(!i.is_zero())),
        Value::Complex(re, im) => Ok(Value::Bool(!complex_is_zero(re, im))),
        Value::Tensor(tensor) => logical_from_tensor(tensor),
        Value::ComplexTensor(tensor) => logical_from_complex_tensor(tensor),
        Value::CharArray(chars) => logical_from_char_array(chars),
        Value::StringArray(strings) => logical_from_string_array(strings),
        Value::GpuTensor(handle) => logical_from_gpu(handle).await,
        Value::String(_) => Err(conversion_error("string")),
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

fn logical_from_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    let buffer = LogicalBuffer::from_real_tensor(&tensor);
    logical_buffer_to_host(buffer)
}

fn logical_from_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let buffer = LogicalBuffer::from_complex_tensor(&tensor);
    logical_buffer_to_host(buffer)
}

fn logical_from_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let buffer = LogicalBuffer::from_char_array(&chars);
    logical_buffer_to_host(buffer)
}

fn logical_from_string_array(strings: StringArray) -> BuiltinResult<Value> {
    let bits: Vec<u8> = strings
        .data
        .iter()
        .map(|s| if s.is_empty() { 0 } else { 1 })
        .collect();
    let shape = canonical_shape(&strings.shape, bits.len());
    logical_buffer_to_host(LogicalBuffer { bits, shape })
}

async fn logical_from_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_is_logical(&handle) {
        return Ok(Value::GpuTensor(handle));
    }

    let provider = runmat_accelerate_api::provider();

    if let Some(p) = provider {
        match p.logical_islogical(&handle) {
            Ok(true) => {
                runmat_accelerate_api::set_handle_logical(&handle, true);
                return Ok(Value::GpuTensor(handle));
            }
            Ok(false) => {}
            Err(err) => {
                trace!("logical: provider logical_islogical hook unavailable, falling back ({err})")
            }
        }
        if let Some(result) = try_gpu_cast(p, &handle).await {
            return Ok(gpu_helpers::logical_gpu_value(result));
        } else {
            trace!(
                "logical: provider elem_ne/zeros_like unavailable for buffer {} – gathering",
                handle.buffer_id
            );
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|err| logical_error(format!("{BUILTIN_NAME}: {err}")))?;
    let buffer = LogicalBuffer::from_real_tensor(&tensor);
    logical_buffer_to_gpu(buffer, provider)
}

fn logical_buffer_to_host(buffer: LogicalBuffer) -> BuiltinResult<Value> {
    let LogicalBuffer { bits, shape } = buffer;
    if tensor::element_count(&shape) == 1 && bits.len() == 1 {
        Ok(Value::Bool(bits[0] != 0))
    } else {
        LogicalArray::new(bits, shape)
            .map(Value::LogicalArray)
            .map_err(|e| logical_error(format!("logical: {e}")))
    }
}

fn logical_buffer_to_gpu(
    buffer: LogicalBuffer,
    provider: Option<&'static dyn AccelProvider>,
) -> BuiltinResult<Value> {
    if let Some(p) = provider {
        let floats: Vec<f64> = buffer
            .bits
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect();
        let view = HostTensorView {
            data: &floats,
            shape: &buffer.shape,
        };
        match p.upload(&view) {
            Ok(handle) => Ok(gpu_helpers::logical_gpu_value(handle)),
            Err(err) => {
                trace!("logical: upload failed during fallback path ({err})");
                logical_buffer_to_host(buffer)
            }
        }
    } else {
        logical_buffer_to_host(buffer)
    }
}

async fn try_gpu_cast(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
) -> Option<GpuTensorHandle> {
    let zeros = provider.zeros_like(input).ok()?;
    let result = provider.elem_ne(input, &zeros).await.ok();
    let _ = provider.free(&zeros);
    result
}

fn complex_is_zero(re: f64, im: f64) -> bool {
    re == 0.0 && im == 0.0
}

fn conversion_error(type_name: &str) -> RuntimeError {
    logical_error(format!(
        "logical: conversion to logical from {} is not possible",
        type_name
    ))
}

#[derive(Clone)]
struct LogicalBuffer {
    bits: Vec<u8>,
    shape: Vec<usize>,
}

impl LogicalBuffer {
    fn from_real_tensor(tensor: &Tensor) -> Self {
        let bits: Vec<u8> = tensor
            .data
            .iter()
            .map(|&v| if v != 0.0 { 1 } else { 0 })
            .collect();
        let shape = canonical_shape(&tensor.shape, bits.len());
        Self { bits, shape }
    }

    fn from_complex_tensor(tensor: &ComplexTensor) -> Self {
        let bits: Vec<u8> = tensor
            .data
            .iter()
            .map(|&(re, im)| if !complex_is_zero(re, im) { 1 } else { 0 })
            .collect();
        let shape = canonical_shape(&tensor.shape, bits.len());
        Self { bits, shape }
    }

    fn from_char_array(chars: &CharArray) -> Self {
        let bits: Vec<u8> = chars
            .data
            .iter()
            .map(|&ch| if (ch as u32) != 0 { 1 } else { 0 })
            .collect();
        let original_shape = vec![chars.rows, chars.cols];
        let shape = canonical_shape(&original_shape, bits.len());
        Self { bits, shape }
    }
}

fn canonical_shape(shape: &[usize], len: usize) -> Vec<usize> {
    if tensor::element_count(shape) == len {
        return normalize_scalar_shape(shape);
    }
    if len == 0 {
        if shape.len() > 1 {
            return shape.to_vec();
        }
        return vec![0];
    }
    if len == 1 {
        canonical_scalar_shape()
    } else {
        vec![len, 1]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CellArray, IntValue, MException, ObjectInstance, StructValue};

    fn logical_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::logical_builtin(value, rest))
    }

    fn assert_error_message(err: crate::RuntimeError, expected: &str) {
        assert_eq!(err.message(), expected);
    }

    fn assert_error_contains(err: crate::RuntimeError, expected: &str) {
        assert!(
            err.message().contains(expected),
            "unexpected error: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_scalar_num() {
        let result = logical_builtin(Value::Num(5.0), Vec::new()).expect("logical");
        assert_eq!(result, Value::Bool(true));

        let zero_result = logical_builtin(Value::Num(0.0), Vec::new()).expect("logical");
        assert_eq!(zero_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_nan_is_true() {
        let tensor = Tensor::new(vec![0.0, f64::NAN, -0.0], vec![1, 3]).unwrap();
        let result = logical_builtin(Value::Tensor(tensor), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => assert_eq!(array.data, vec![0, 1, 0]),
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_tensor_matrix() {
        let tensor = Tensor::new(vec![0.0, 2.0, -3.0, 0.0], vec![2, 2]).unwrap();
        let result = logical_builtin(Value::Tensor(tensor), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(array.data, vec![0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_complex_conversion() {
        let complex =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)], vec![3, 1]).unwrap();
        let result = logical_builtin(Value::ComplexTensor(complex), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.data, vec![0, 1, 1]);
            }
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_char_array_conversion() {
        let chars = CharArray::new(vec!['A', '\0', 'C'], 1, 3).unwrap();
        let result = logical_builtin(Value::CharArray(chars), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => assert_eq!(array.data, vec![1, 0, 1]),
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_string_error() {
        let err = logical_builtin(Value::String("runmat".to_string()), Vec::new()).unwrap_err();
        assert_error_message(
            err,
            "logical: conversion to logical from string is not possible",
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_struct_error() {
        let mut st = StructValue::new();
        st.insert("field", Value::Num(1.0));
        let err = logical_builtin(Value::Struct(st), Vec::new()).unwrap_err();
        assert_error_contains(err, "struct");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_cell_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell creation");
        let err = logical_builtin(Value::Cell(cell), Vec::new()).unwrap_err();
        assert_error_message(
            err,
            "logical: conversion to logical from cell is not possible",
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_function_handle_error() {
        let err = logical_builtin(Value::FunctionHandle("foo".into()), Vec::new()).unwrap_err();
        assert_error_message(
            err,
            "logical: conversion to logical from function_handle is not possible",
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_object_error() {
        let obj = ObjectInstance::new("DemoClass".to_string());
        let err = logical_builtin(Value::Object(obj), Vec::new()).unwrap_err();
        assert_error_contains(err, "DemoClass");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_mexception_error() {
        let mex = MException::new("id:logical".into(), "message".into());
        let err = logical_builtin(Value::MException(mex), Vec::new()).unwrap_err();
        assert_error_message(
            err,
            "logical: conversion to logical from MException is not possible",
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, -2.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                logical_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("logical");
            let gathered = test_support::gather(result.clone()).expect("gather");
            assert_eq!(gathered.data, vec![0.0, 1.0, 1.0]);
            if let Value::GpuTensor(out) = result {
                assert!(runmat_accelerate_api::handle_is_logical(&out));
            } else {
                panic!("expected gpu tensor output");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_gpu_passthrough_for_logical_handle() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_logical(&handle, true);
            let result =
                logical_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("logical");
            match result {
                Value::GpuTensor(out) => assert_eq!(out, handle),
                other => panic!("expected gpu tensor, got {:?}", other),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_bool_and_logical_inputs_passthrough() {
        let res_bool = logical_builtin(Value::Bool(true), Vec::new()).expect("logical");
        assert_eq!(res_bool, Value::Bool(true));

        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let res_array =
            logical_builtin(Value::LogicalArray(logical.clone()), Vec::new()).expect("logical");
        assert_eq!(res_array, Value::LogicalArray(logical));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = logical_builtin(Value::Tensor(tensor), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => {
                assert!(array.data.is_empty());
                assert_eq!(array.shape, vec![0, 3]);
            }
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_integer_scalar() {
        let res = logical_builtin(Value::Int(IntValue::I32(0)), Vec::new()).expect("logical");
        assert_eq!(res, Value::Bool(false));

        let res_nonzero =
            logical_builtin(Value::Int(IntValue::I32(-5)), Vec::new()).expect("logical");
        assert_eq!(res_nonzero, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn logical_wgpu_matches_cpu_conversion() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(vec![0.0, 2.0, -3.0, f64::NAN], vec![2, 2]).unwrap();
        let cpu = logical_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = logical_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        let out_handle = match gpu_value {
            Value::GpuTensor(ref h) => {
                assert!(runmat_accelerate_api::handle_is_logical(h));
                h.clone()
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        };

        let gathered = test_support::gather(Value::GpuTensor(out_handle)).expect("gather");

        let (expected, expected_shape): (Vec<f64>, Vec<usize>) = match cpu {
            Value::LogicalArray(arr) => (
                arr.data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect(),
                arr.shape.clone(),
            ),
            Value::Bool(flag) => (vec![if flag { 1.0 } else { 0.0 }], vec![1, 1]),
            other => panic!("unexpected cpu result {other:?}"),
        };

        assert_eq!(gathered.shape, expected_shape);
        assert_eq!(gathered.data, expected);
    }
}
