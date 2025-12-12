//! MATLAB-compatible `isempty` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_numel;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "isempty",
        builtin_path = "crate::builtins::array::introspection::isempty"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "isempty"
category: "array/introspection"
keywords: ["isempty", "empty array", "metadata query", "gpu", "logical"]
summary: "Return true when an array has zero elements, matching MATLAB semantics."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Reads tensor metadata from handles; falls back to gathering only when provider metadata is absent."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::introspection::isempty::tests"
  integration: "builtins::array::introspection::isempty::tests::isempty_gpu_tensor_respects_shape"
---

# What does the `isempty` function do in MATLAB / RunMat?
`isempty(A)` returns logical `true` when the MATLAB value `A` contains zero elements. It mirrors MATLAB's
behaviour across numeric arrays, logical arrays, character arrays, string arrays, cell arrays, structs,
GPU tensors, and handle-like values.

## How does the `isempty` function behave in MATLAB / RunMat?
- Arrays are empty when any dimension is zero (`prod(size(A)) == 0`).
- Character arrays report empty when either dimension is zero. String scalars are never empty (`""` is a
  `1×1` string array whose content may be empty, but it still counts as one element).
- Cell arrays, struct arrays, and tables (future work) follow their MATLAB dimensions; `cell(0, n)` is
  empty, while `cell(1, 1)` is not.
- Scalars, logical values, numeric scalars, function handles, objects, and handle objects always return
  `false` because they occupy one element.
- GPU tensors rely on device-provided shape metadata to avoid unnecessary transfers. If metadata is
  missing, RunMat gathers once to confirm the shape.

## `isempty` Function GPU Execution Behaviour
`isempty` does not launch GPU kernels. For GPU-resident tensors (`gpuArray` values), RunMat inspects the
shape stored in the `GpuTensorHandle`. When the active provider omits that metadata, the runtime downloads
the tensor once to maintain MATLAB-compatible results. The builtin always returns a host logical scalar and
does not allocate device memory, so it is safe in fused GPU expressions.

## Examples of using the `isempty` function in MATLAB / RunMat

### Checking if a matrix has any elements

```matlab
A = zeros(0, 3);
tf = isempty(A);
```

Expected output:

```matlab
tf = logical(1)
```

### Detecting an empty cell array

```matlab
C = cell(0, 4);
tf = isempty(C);
```

Expected output:

```matlab
tf = logical(1)
```

### Using `isempty` on a GPU-resident tensor

```matlab
G = gpuArray(zeros(5, 0));
tf = isempty(G);
```

Expected output:

```matlab
tf = logical(1)
```

### Distinguishing char arrays from string scalars

```matlab
chars = '';
tf_chars = isempty(chars);
str = "";
tf_str = isempty(str);
```

Expected output:

```matlab
tf_chars = logical(1)
tf_str = logical(0)
```

### Confirming that scalars are never empty

```matlab
value = 42;
tf = isempty(value);
```

Expected output:

```matlab
tf = logical(0)
```

### Inspecting empty string arrays

```matlab
S = strings(0, 2);
tf = isempty(S);
```

Expected output:

```matlab
tf = logical(1)
```

## FAQ

### Does `isempty` gather GPU data?
Only when the provider fails to populate shape metadata on the GPU handle. Otherwise, it answers using the
metadata and avoids transfers.

### Why is `isempty("")` false but `isempty('')` true?
String scalars are `1×1` arrays that hold text content, so they are not empty even when the text has length
zero. Character arrays store individual characters; the empty char literal `''` has zero columns, so it is
empty.

### How does `isempty` treat structs and objects?
Structs follow their array dimensions. A scalar struct (`1×1`) is not empty, while a `0×0` struct array is
empty. Value objects and handle objects are always treated as scalar values, so `isempty` returns `false`.

### Can `isempty` be used inside GPU-fused expressions?
Yes. The builtin returns a host logical scalar and does not allocate GPU buffers, so fusion plans remain
valid.

### Does `isempty` look inside cell arrays?
No. It only checks the container dimensions. To look at the contents, inspect individual cells.

### What does `isempty` return for logical or numeric scalars?
They behave like any other scalar and return `false`.

## See Also
[numel](./numel), [size](./size), [length](./length), [gpuArray](./gpuarray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::isempty")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isempty",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Queries tensor metadata; gathers only when the provider fails to expose shapes.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::isempty"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isempty",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that returns a host logical scalar for fusion planning.",
};

#[runtime_builtin(
    name = "isempty",
    category = "array/introspection",
    summary = "Return true when an array has zero elements, matching MATLAB semantics.",
    keywords = "isempty,empty array,metadata query,gpu,logical",
    accel = "metadata",
    builtin_path = "crate::builtins::array::introspection::isempty"
)]
fn isempty_builtin(value: Value) -> Result<Value, String> {
    let is_empty = value_is_empty(&value);
    Ok(Value::Bool(is_empty))
}

fn value_is_empty(value: &Value) -> bool {
    value_numel(value) == 0
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_builtins::{CellArray, CharArray, Tensor};

    #[test]
    fn isempty_empty_tensor_returns_true() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = isempty_builtin(Value::Tensor(tensor)).expect("isempty");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn isempty_scalar_returns_false() {
        let result = isempty_builtin(Value::Num(5.0)).expect("isempty");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn isempty_char_array_behaves_like_matlab() {
        let empty_chars = CharArray::new_row("");
        let non_empty_chars = CharArray::new_row("RunMat");
        let empty = isempty_builtin(Value::CharArray(empty_chars)).expect("isempty");
        let non_empty = isempty_builtin(Value::CharArray(non_empty_chars)).expect("isempty");
        assert_eq!(empty, Value::Bool(true));
        assert_eq!(non_empty, Value::Bool(false));
    }

    #[test]
    fn isempty_cell_array_uses_dimensions() {
        let empty_cell = CellArray::new(Vec::new(), 0, 2).unwrap();
        let populated_cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let empty = isempty_builtin(Value::Cell(empty_cell)).expect("isempty");
        let populated = isempty_builtin(Value::Cell(populated_cell)).expect("isempty");
        assert_eq!(empty, Value::Bool(true));
        assert_eq!(populated, Value::Bool(false));
    }

    #[test]
    fn isempty_string_scalar_is_false_even_if_empty_text() {
        let result = isempty_builtin(Value::String(String::new())).expect("isempty");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn isempty_string_array_zero_rows_is_true() {
        let array = runmat_builtins::StringArray::new(Vec::new(), vec![0, 2]).unwrap();
        let result = isempty_builtin(Value::StringArray(array)).expect("isempty");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn isempty_gpu_tensor_respects_shape() {
        test_support::with_test_provider(|provider| {
            let empty_tensor = Tensor::new(Vec::new(), vec![0, 4]).unwrap();
            let non_empty_tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();

            let empty_view = runmat_accelerate_api::HostTensorView {
                data: &empty_tensor.data,
                shape: &empty_tensor.shape,
            };
            let non_empty_view = runmat_accelerate_api::HostTensorView {
                data: &non_empty_tensor.data,
                shape: &non_empty_tensor.shape,
            };

            let empty_handle = provider.upload(&empty_view).expect("upload empty");
            let non_empty_handle = provider.upload(&non_empty_view).expect("upload non-empty");

            let empty_result =
                isempty_builtin(Value::GpuTensor(empty_handle)).expect("isempty empty");
            let non_empty_result =
                isempty_builtin(Value::GpuTensor(non_empty_handle)).expect("isempty non-empty");

            assert_eq!(empty_result, Value::Bool(true));
            assert_eq!(non_empty_result, Value::Bool(false));
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn isempty_wgpu_provider_uses_handle_shape() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let empty_tensor = Tensor::new(Vec::new(), vec![0, 4]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &empty_tensor.data,
            shape: &empty_tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        assert_eq!(
            handle.shape,
            vec![0, 4],
            "provider should surface tensor shape"
        );

        let result = isempty_builtin(Value::GpuTensor(handle)).expect("isempty");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
