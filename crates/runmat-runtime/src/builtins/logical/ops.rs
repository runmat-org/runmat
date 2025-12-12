//! MATLAB-compatible `logical` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::{self, AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{
    gpu_helpers,
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
    },
    tensor,
};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "logical",
        builtin_path = "crate::builtins::logical::ops"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "logical"
category: "logical"
keywords: ["logical", "boolean conversion", "truth mask", "gpuArray", "mask array"]
summary: "Convert scalars, arrays, and gpuArray values to MATLAB-compatible logical values."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers a device-side elem\\_ne(X, 0) cast when the provider supports elem_ne and zeros_like; otherwise gathers to the host, converts, and re-uploads the logical result."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::ops::tests"
  integration: "builtins::logical::ops::tests::logical_gpu_roundtrip"
---

# What does the `logical` function do in MATLAB / RunMat?
`logical(X)` converts numeric, logical, character, and gpuArray inputs into MATLAB logical values (booleans). Any non-zero (or `NaN`/`Inf`) element maps to `true`, while zero maps to `false`. Logical inputs are returned unchanged.

## How does the `logical` function behave in MATLAB / RunMat?
- `logical` accepts scalars, dense arrays, N-D tensors, and gpuArrays. Shapes are preserved bit-for-bit.
- Non-zero numeric values, `NaN`, and `Inf` map to `true`; `0` and `-0` map to `false`.
- Complex inputs are considered `true` when either the real or imaginary component is non-zero.
- Character arrays are converted elementwise by interpreting code points (so `'A'` becomes `true`, `'\0'` becomes `false`).
- Strings, structs, cells, objects, and other non-numeric types raise MATLAB-compatible errors (`"Conversion to logical from <type> is not possible"`).
- Scalar results become logical scalars (`true`/`false`); higher-rank arrays produce dense logical arrays.

## `logical` Function GPU Execution Behaviour
- When a GPU provider implements `elem_ne` and `zeros_like`, RunMat performs the conversion in-place on the device by evaluating `elem_ne(X, 0)`, then marks the resulting handle as logical so predicates like `islogical` work without downloads.
- If the provider cannot service the request (missing hooks, unsupported dtype, or allocation failure), the value is transparently gathered to the host, converted, and—when a provider is still available—re-uploaded as a logical gpuArray. The fallback is documented so users understand potential host/device transitions.
- Handles that are already flagged as logical (`gpuArray.logical`) are returned without modification.
- Scalars remain scalars: converting a `gpuArray` scalar preserves the residency and returns a logical gpuArray scalar.

## Examples of using the `logical` function in MATLAB / RunMat

### Creating a logical mask from numeric data
```matlab
values = [0 2 -3 0];
mask = logical(values);
```
Expected output:
```matlab
mask =
  1×4 logical array
     0     1     1     0
```

### Building a logical mask from a matrix
```matlab
M = [-4 0 8; 0 1 0];
mask = logical(M);
```
Expected output:
```matlab
mask =
  2×3 logical array
     1     0     1
     0     1     0
```

### Treating NaN and Inf values as true
```matlab
flags = logical([NaN Inf 0]);
```
Expected output:
```matlab
flags =
  1×3 logical array
     1     1     0
```

### Converting complex numbers to logical scalars
```matlab
z = logical(3 + 4i);
w = logical(0 + 0i);
```
Expected output:
```matlab
z =
     1
w =
     0
```

### Converting character arrays to logical values
```matlab
chars = ['A' 0 'C'];
mask = logical(chars);
```
Expected output:
```matlab
mask =
  1×3 logical array
     1     0     1
```

### Keeping gpuArray inputs on the device
```matlab
G = gpuArray([0 1 2]);
maskGPU = logical(G);
hostMask = gather(maskGPU);
```
Expected output:
```matlab
hostMask =
  1×3 logical array
     0     1     1
```

### Preserving empty shapes through logical conversion
```matlab
emptyVec = zeros(0, 3);
logicalEmpty = logical(emptyVec);
```
Expected output:
```matlab
logicalEmpty =
  0×3 logical array
     []
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You rarely need to call `gpuArray` manually. When the acceleration provider is active, RunMat keeps logical conversions on the GPU by issuing `elem_ne(X, 0)` kernels (backed by `zeros_like` allocations) and flagging the handle as logical metadata. Explicit `gpuArray` calls are available for MATLAB compatibility or when you want to pin residency before interacting with external libraries. When the provider lacks the necessary hook, RunMat documents the fallback: it gathers the data, converts it on the host, and—if a provider is still available—re-uploads the logical mask so downstream GPU code continues to work without residency surprises.

## FAQ

### Which input types does `logical` support?
Numeric, logical, complex, character, and gpuArray values are accepted. Strings, structs, cells, objects, and function handles are rejected with MATLAB-compatible error messages.

### How are NaN or Inf values treated?
They evaluate to `true`. MATLAB defines logical conversion as “non-zero”, and `NaN` / `Inf` both satisfy that rule.

### How does `logical` handle complex numbers?
The result is `true` when either the real or imaginary component is non-zero (or `NaN`/`Inf`). Only `0 + 0i` converts to `false`.

### Does the builtin change array shapes?
No. Shapes are preserved exactly, including empty dimensions and higher-rank tensors.

### What happens to existing logical arrays?
They are returned verbatim. Logical gpuArrays remain on the device without triggering new allocations.

### Can I convert strings with `logical`?
No. MATLAB rejects string inputs, and RunMat mirrors that behaviour: `"logical: conversion to logical from string is not possible"`.

### What about structs, cells, or objects?
They raise the same conversion error as MATLAB. Use functions like `~cellfun(@isempty, ...)` to derive masks instead.

### Does the GPU path allocate new buffers?
Only when the provider cannot operate in-place. The preferred path performs `elem_ne` against a zero tensor and reuses the resulting buffer. Fallback paths allocate a new gpuArray after gathering to the host.

### Where can I learn more?
See the references below and the RunMat source for implementation details.

## See Also
[`islogical`](./islogical), [`gpuArray`](./gpuarray), [`gather`](./gather), [`find`](./find)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/logical/ops.rs`
- Issues & feature requests: [https://github.com/runmat-org/runmat/issues/new/choose](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

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

#[runtime_builtin(
    name = "logical",
    category = "logical",
    summary = "Convert scalars, arrays, and gpuArray values to logical outputs.",
    keywords = "logical,boolean,gpuArray,mask,conversion",
    accel = "unary",
    builtin_path = "crate::builtins::logical::ops"
)]
fn logical_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    if !rest.is_empty() {
        return Err("logical: too many input arguments".to_string());
    }
    convert_value_to_logical(value)
}

fn convert_value_to_logical(value: Value) -> Result<Value, String> {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => Ok(value),
        Value::Num(n) => Ok(Value::Bool(n != 0.0)),
        Value::Int(i) => Ok(Value::Bool(!i.is_zero())),
        Value::Complex(re, im) => Ok(Value::Bool(!complex_is_zero(re, im))),
        Value::Tensor(tensor) => logical_from_tensor(tensor),
        Value::ComplexTensor(tensor) => logical_from_complex_tensor(tensor),
        Value::CharArray(chars) => logical_from_char_array(chars),
        Value::StringArray(strings) => logical_from_string_array(strings),
        Value::GpuTensor(handle) => logical_from_gpu(handle),
        Value::String(_) => Err(conversion_error("string")),
        Value::Cell(_) => Err(conversion_error("cell")),
        Value::Struct(_) => Err(conversion_error("struct")),
        Value::Object(obj) => Err(conversion_error(&obj.class_name)),
        Value::HandleObject(handle) => Err(conversion_error(&handle.class_name)),
        Value::Listener(_) => Err(conversion_error("event.listener")),
        Value::FunctionHandle(_) | Value::Closure(_) => Err(conversion_error("function_handle")),
        Value::ClassRef(_) => Err(conversion_error("meta.class")),
        Value::MException(_) => Err(conversion_error("MException")),
    }
}

fn logical_from_tensor(tensor: Tensor) -> Result<Value, String> {
    let buffer = LogicalBuffer::from_real_tensor(&tensor);
    logical_buffer_to_host(buffer)
}

fn logical_from_complex_tensor(tensor: ComplexTensor) -> Result<Value, String> {
    let buffer = LogicalBuffer::from_complex_tensor(&tensor);
    logical_buffer_to_host(buffer)
}

fn logical_from_char_array(chars: CharArray) -> Result<Value, String> {
    let buffer = LogicalBuffer::from_char_array(&chars);
    logical_buffer_to_host(buffer)
}

fn logical_from_string_array(strings: StringArray) -> Result<Value, String> {
    let bits: Vec<u8> = strings
        .data
        .iter()
        .map(|s| if s.is_empty() { 0 } else { 1 })
        .collect();
    let shape = canonical_shape(&strings.shape, bits.len());
    logical_buffer_to_host(LogicalBuffer { bits, shape })
}

fn logical_from_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
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
        if let Some(result) = try_gpu_cast(p, &handle) {
            return Ok(gpu_helpers::logical_gpu_value(result));
        } else {
            trace!(
                "logical: provider elem_ne/zeros_like unavailable for buffer {} – gathering",
                handle.buffer_id
            );
        }
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let buffer = LogicalBuffer::from_real_tensor(&tensor);
    logical_buffer_to_gpu(buffer, provider)
}

fn logical_buffer_to_host(buffer: LogicalBuffer) -> Result<Value, String> {
    let LogicalBuffer { bits, shape } = buffer;
    if tensor::element_count(&shape) == 1 && bits.len() == 1 {
        Ok(Value::Bool(bits[0] != 0))
    } else {
        LogicalArray::new(bits, shape)
            .map(Value::LogicalArray)
            .map_err(|e| format!("logical: {e}"))
    }
}

fn logical_buffer_to_gpu(
    buffer: LogicalBuffer,
    provider: Option<&'static dyn AccelProvider>,
) -> Result<Value, String> {
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

fn try_gpu_cast(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
) -> Option<GpuTensorHandle> {
    let zeros = provider.zeros_like(input).ok()?;
    let result = provider.elem_ne(input, &zeros).ok();
    let _ = provider.free(&zeros);
    result
}

fn complex_is_zero(re: f64, im: f64) -> bool {
    re == 0.0 && im == 0.0
}

fn conversion_error(type_name: &str) -> String {
    format!(
        "logical: conversion to logical from {} is not possible",
        type_name
    )
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
    if !shape.is_empty() && tensor::element_count(shape) == len {
        return shape.to_vec();
    }
    if len == 0 {
        if shape.len() > 1 {
            return shape.to_vec();
        }
        return vec![0];
    }
    if len == 1 {
        vec![1, 1]
    } else {
        vec![len, 1]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CellArray, IntValue, MException, ObjectInstance, StructValue};

    #[test]
    fn logical_scalar_num() {
        let result = logical_builtin(Value::Num(5.0), Vec::new()).expect("logical");
        assert_eq!(result, Value::Bool(true));

        let zero_result = logical_builtin(Value::Num(0.0), Vec::new()).expect("logical");
        assert_eq!(zero_result, Value::Bool(false));
    }

    #[test]
    fn logical_nan_is_true() {
        let tensor = Tensor::new(vec![0.0, f64::NAN, -0.0], vec![1, 3]).unwrap();
        let result = logical_builtin(Value::Tensor(tensor), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => assert_eq!(array.data, vec![0, 1, 0]),
            other => panic!("expected logical array, got {:?}", other),
        }
    }

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

    #[test]
    fn logical_char_array_conversion() {
        let chars = CharArray::new(vec!['A', '\0', 'C'], 1, 3).unwrap();
        let result = logical_builtin(Value::CharArray(chars), Vec::new()).expect("logical");
        match result {
            Value::LogicalArray(array) => assert_eq!(array.data, vec![1, 0, 1]),
            other => panic!("expected logical array, got {:?}", other),
        }
    }

    #[test]
    fn logical_string_error() {
        let err = logical_builtin(Value::String("runmat".to_string()), Vec::new()).unwrap_err();
        assert_eq!(
            err,
            "logical: conversion to logical from string is not possible"
        );
    }

    #[test]
    fn logical_struct_error() {
        let mut st = StructValue::new();
        st.insert("field", Value::Num(1.0));
        let err = logical_builtin(Value::Struct(st), Vec::new()).unwrap_err();
        assert!(err.contains("struct"), "unexpected error message: {err}");
    }

    #[test]
    fn logical_cell_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell creation");
        let err = logical_builtin(Value::Cell(cell), Vec::new()).unwrap_err();
        assert_eq!(
            err,
            "logical: conversion to logical from cell is not possible"
        );
    }

    #[test]
    fn logical_function_handle_error() {
        let err = logical_builtin(Value::FunctionHandle("foo".into()), Vec::new()).unwrap_err();
        assert_eq!(
            err,
            "logical: conversion to logical from function_handle is not possible"
        );
    }

    #[test]
    fn logical_object_error() {
        let obj = ObjectInstance::new("DemoClass".to_string());
        let err = logical_builtin(Value::Object(obj), Vec::new()).unwrap_err();
        assert!(
            err.contains("DemoClass"),
            "expected class name in error, got {err}"
        );
    }

    #[test]
    fn logical_mexception_error() {
        let mex = MException::new("id:logical".into(), "message".into());
        let err = logical_builtin(Value::MException(mex), Vec::new()).unwrap_err();
        assert_eq!(
            err,
            "logical: conversion to logical from MException is not possible"
        );
    }

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

    #[test]
    fn logical_bool_and_logical_inputs_passthrough() {
        let res_bool = logical_builtin(Value::Bool(true), Vec::new()).expect("logical");
        assert_eq!(res_bool, Value::Bool(true));

        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let res_array =
            logical_builtin(Value::LogicalArray(logical.clone()), Vec::new()).expect("logical");
        assert_eq!(res_array, Value::LogicalArray(logical));
    }

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

    #[test]
    fn logical_integer_scalar() {
        let res = logical_builtin(Value::Int(IntValue::I32(0)), Vec::new()).expect("logical");
        assert_eq!(res, Value::Bool(false));

        let res_nonzero =
            logical_builtin(Value::Int(IntValue::I32(-5)), Vec::new()).expect("logical");
        assert_eq!(res_nonzero, Value::Bool(true));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

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
