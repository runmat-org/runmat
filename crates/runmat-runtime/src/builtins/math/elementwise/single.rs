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
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "single",
        builtin_path = "crate::builtins::math::elementwise::single"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "single"
category: "math/elementwise"
keywords: ["single", "float32", "cast", "convert to single", "gpuArray single"]
summary: "Convert numeric values, character arrays, and gpuArray handles to IEEE single precision."
references:
  - https://www.mathworks.com/help/matlab/ref/single.html
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers a device-side cast when providers expose a dedicated unary hook; otherwise gathers, converts on the host, and re-uploads when a provider is present."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::single::tests"
  gpu: "builtins::math::elementwise::single::tests::single_gpu_roundtrip"
  wgpu: "builtins::math::elementwise::single::tests::single_wgpu_matches_cpu"
---

# What does the `single` function do in MATLAB / RunMat?
`single(X)` converts scalars, arrays, complex values, characters, logical inputs, and gpuArray handles
into single-precision (`float32`) values while preserving MATLAB's column-major shapes.

## How does the `single` function behave in MATLAB / RunMat?
- Numeric scalars and arrays (double or integer) are rounded to IEEE single precision. Scalars remain
  scalars; arrays keep their exact shape and column-major layout.
- Logical inputs become floating-point `0`/`1`, matching MATLAB’s implicit promotion rules for logical
  arithmetic.
- Complex values convert both the real and imaginary components to single precision using MATLAB’s
  analytic extension rules.
- Character arrays return their Unicode code points (equivalent to `double(char)`), quantised to
  single precision. Strings, structs, cells, objects, and other unsupported classes raise MATLAB-style
  errors (`"single: conversion to single from <type> is not possible"`).
- Empty arrays stay empty, singleton expansion is unaffected, and metadata such as orientation is
  preserved exactly.

## `single` Function GPU Execution Behaviour
- When the input is a gpuArray and the provider implements `unary_single`, RunMat keeps the data on
  device and returns a new gpuArray handle that already contains single-precision values. This path is
  also used by fused elementwise kernels emitted by Turbine.
- If the provider lacks the hook, RunMat gathers the data to host memory, converts it to single
  precision, and re-uploads it so downstream GPU code still receives a device-resident value. The
  original handle is freed once the replacement upload succeeds.
- When no provider is active the fallback stops after the host conversion, returning a CPU tensor that
  still respects single-precision rounding. Scalars follow the same logic (`single(gpuArray(pi))`
  returns a device scalar when possible, or a host scalar if the provider is absent).

## Examples of using the `single` function in MATLAB / RunMat

### Convert a matrix to single precision
```matlab
A = [1 2 3; 4 5 6];
B = single(A);
```
Expected output:
```matlab
B =
  2×3 single matrix
     1     2     3
     4     5     6
```

### Convert a scalar double to single precision
```matlab
pi_single = single(pi);
```
Expected output:
```matlab
pi_single =
  single
     3.1416
```

### Convert complex numbers to single precision
```matlab
z = [1+2i, 3-4i];
single_z = single(z);
```
Expected output:
```matlab
single_z =
  1×2 single complex row vector
   1.0000 + 2.0000i   3.0000 - 4.0000i
```

### Convert a character array to single precision codes
```matlab
codes = single('ABC');
```
Expected output:
```matlab
codes =
  1×3 single row vector
    65    66    67
```

### Keep gpuArray inputs on the GPU
```matlab
G = gpuArray(reshape(0:5, 3, 2));
H = single(G);
gather(H)
```
Expected output:
```matlab
ans =
  3×2 single matrix
     0     3
     1     4
     2     5
```

### Convert logical masks to single precision values
```matlab
mask = logical([0 1 0 1]);
weights = single(mask);
```
Expected output:
```matlab
weights =
  1×4 single row vector
     0     1     0     1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You rarely need to call `gpuArray` solely for type conversion. RunMat’s planner already keeps hot
paths on the GPU. When `single` runs on gpuArray inputs it attempts a device-side cast first, and
falls back to gather/convert/re-upload when hooks are missing. The fallback is documented so users
understand the temporary host hop.

## FAQ

### Why does `single` change the numeric value slightly?
Single precision has ~7 decimal digits of accuracy. Values that require more precision are rounded
to the nearest representable float32 number, matching MATLAB exactly.

### Does `single` accept integer types?
Yes. Integer and logical inputs are promoted to floating point with MATLAB’s semantics. Saturation is
not required because the target type is floating-point.

### Can I convert strings with `single`?
No. RunMat mirrors MATLAB’s behaviour: strings raise
`"single: conversion to single from string is not possible"`. Convert strings to character arrays
first with `char`.

### What about structs, cells, or user objects?
They raise the same conversion error with their class name. Extract the numeric data you need before
calling `single`.

### Does `single` support complex numbers?
Yes. Both the real and imaginary components are converted elementwise, just like MATLAB’s analytic
extension.

### How does `single` behave on empty arrays?
Empty arrays stay empty. Shapes and orientations are preserved (`0×n` stays `0×n`).

### Will gpuArray inputs stay on the GPU?
Yes. RunMat attempts a device-side cast, and if the provider lacks one it converts on the host and
re-uploads the result so downstream GPU code still sees a gpuArray.

### How does this affect `class` or `isa`?
RunMat currently reports host tensors as `"double"` because their backing storage is f64. This mirrors
other parts of the runtime (for example, `gather` of a single-precision gpuArray). GPU results still
behave correctly when dispatched through Accelerate.

### Do NaN or Inf values survive the cast?
Yes. IEEE semantics are preserved; NaNs, ±Inf, and signed zeros remain after conversion.

### Where can I learn more?
See MathWorks’ documentation linked above or inspect this builtin’s source file for the exact
implementation.

## See Also
[gpuArray](./gpuarray), [gather](./gather), [logical](./ops), [single (MathWorks)](https://www.mathworks.com/help/matlab/ref/single.html)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/math/elementwise/single.rs`
- Issues & feature requests: [https://github.com/runmat-org/runmat/issues/new/choose](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

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

#[runtime_builtin(
    name = "single",
    category = "math/elementwise",
    summary = "Convert scalars, arrays, and gpuArray values to single precision.",
    keywords = "single,float32,cast,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::single"
)]
fn single_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
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
        Value::GpuTensor(handle) => single_from_gpu(handle),
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
    apply_output_template(converted, &template)
}

fn single_from_tensor(tensor: Tensor) -> Result<Value, String> {
    single_tensor_to_host(tensor).map(Value::Tensor)
}

fn single_from_complex_tensor(tensor: ComplexTensor) -> Result<Value, String> {
    single_complex_tensor_to_host(tensor).map(Value::ComplexTensor)
}

fn single_from_logical_array(array: LogicalArray) -> Result<Value, String> {
    let tensor = tensor::logical_to_tensor(&array).map_err(|e| format!("single: {e}"))?;
    single_tensor_to_host(tensor).map(Value::Tensor)
}

fn single_from_char_array(chars: CharArray) -> Result<Value, String> {
    let tensor = char_array_to_tensor(&chars)?;
    single_tensor_to_host(tensor).map(Value::Tensor)
}

fn single_from_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match provider.unary_single(&handle) {
            Ok(result) => {
                let _ = provider.free(&handle);
                return Ok(Value::GpuTensor(result));
            }
            Err(err) => {
                trace!("single: provider unary_single hook unavailable, falling back ({err})");
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
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

fn single_tensor_to_host(mut tensor: Tensor) -> Result<Tensor, String> {
    cast_slice_to_single(&mut tensor.data);
    tensor.dtype = NumericDType::F32;
    Ok(tensor)
}

fn single_complex_tensor_to_host(mut tensor: ComplexTensor) -> Result<ComplexTensor, String> {
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

fn char_array_to_tensor(chars: &CharArray) -> Result<Tensor, String> {
    let ascii: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(ascii, vec![chars.rows, chars.cols]).map_err(|e| format!("single: {e}"))
}

fn conversion_error(type_name: &str) -> String {
    format!(
        "single: conversion to single from {} is not possible",
        type_name
    )
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> Result<OutputTemplate, String> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err("single: expected prototype after 'like'".to_string())
            } else {
                Err("single: unrecognised argument for single".to_string())
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err("single: unsupported option; only 'like' is accepted".to_string())
            }
        }
        _ => Err("single: too many input arguments".to_string()),
    }
}

fn apply_output_template(value: Value, template: &OutputTemplate) -> Result<Value, String> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(_) => convert_to_gpu(value),
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value),
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                Err("single: complex prototypes for 'like' are not supported yet".to_string())
            }
            _ => Err(
                "single: unsupported prototype for 'like'; provide a numeric or gpuArray prototype"
                    .to_string(),
            ),
        },
    }
}

fn convert_to_gpu(value: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        "single: GPU output requested via 'like' but no acceleration provider is active".to_string()
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|e| format!("single: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("single: {e}"))?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("single: GPU prototypes for 'like' only support real numeric outputs".to_string())
        }
        other => Err(format!(
            "single: unsupported result type for GPU output via 'like' ({other:?})"
        )),
    }
}

fn convert_to_host_like(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value(&proxy).map_err(|e| format!("single: {e}"))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;

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
        assert!(err.contains("single"));
        assert!(err.contains("string"));
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
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
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
        let gpu_value = single_from_gpu(handle).expect("gpu single");
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
        let gpu_value = single_from_gpu(handle).expect("gpu single");
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
