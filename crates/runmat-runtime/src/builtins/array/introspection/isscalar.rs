//! MATLAB-compatible `isscalar` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::{value_dimensions, value_numel};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "isscalar",
        builtin_path = "crate::builtins::array::introspection::isscalar"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "isscalar"
category: "array/introspection"
keywords: ["isscalar", "scalar", "metadata query", "gpu", "logical"]
summary: "Return true when a value has exactly one element and unit dimensions."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Inspects tensor metadata on GPU handles; gathers only when metadata is missing."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::introspection::isscalar::tests"
  integration: "builtins::array::introspection::isscalar::tests::isscalar_gpu_tensor_checks_dimensions"
---

# What does the `isscalar` function do in MATLAB / RunMat?
`isscalar(A)` returns logical `true` when an input has exactly one element and every visible
dimension equals one. The builtin mirrors MATLAB behaviour across numeric values, logical arrays,
strings, character arrays, cells, structs, GPU tensors, and handle-like values.

## How does the `isscalar` function behave in MATLAB / RunMat?
- `isscalar` is true if and only if `numel(A) == 1` **and** every entry of `size(A)` equals `1`.
- Numeric, logical, and complex scalars (`42`, `true`, `3+4i`) satisfy both conditions.
- Row, column, or higher-dimensional vectors (e.g. `[1 2 3]`, `zeros(2,1)`) are not scalar because at
  least one dimension exceeds one.
- String scalars (`"hello"`, `""`) are scalar because they are `1×1` string arrays. Character arrays
  must be `1×1` to count as scalar—`'h'` is scalar, while `'runmat'` (`1×6`) is not.
- Cell arrays, structs, objects, and handle objects are scalar when their MATLAB dimensions are `1×1`,
  regardless of the contents they wrap.
- Empty arrays (`[]`, `cell(0,1)`, strings of size `0×1`) are **not** scalar because they contain zero
  elements even if some dimensions equal one.
- GPU tensors rely on the shape metadata stored in their `GpuTensorHandle`. If a provider omits that
  metadata, RunMat gathers once to confirm the dimensions before answering.

## `isscalar` Function GPU Execution Behaviour
`isscalar` never launches GPU kernels. For `gpuArray` inputs the builtin first inspects the shape stored
in the `GpuTensorHandle`. When the active acceleration provider omits that metadata, RunMat performs a
single gather to compute `numel` and dimensions on the host. The builtin always returns a host logical
scalar, so fusion planning treats it as a metadata query instead of a device-side kernel.

## Examples of using the `isscalar` function in MATLAB / RunMat

### Checking if a numeric value is scalar

```matlab
tf = isscalar(42);
```

Expected output:

```matlab
tf = logical(1)
```

### Detecting that a row vector is not scalar

```matlab
tf = isscalar([1 2 3]);
```

Expected output:

```matlab
tf = logical(0)
```

### Verifying scalar status of string vs char arrays

```matlab
tf_string = isscalar("hello");
tf_char = isscalar('h');
tf_char_row = isscalar('runmat');
```

Expected output:

```matlab
tf_string = logical(1)
tf_char = logical(1)
tf_char_row = logical(0)
```

### Identifying that an empty array is not scalar

```matlab
tf_empty = isscalar([]);
```

Expected output:

```matlab
tf_empty = logical(0)
```

### Confirming that single-element cell arrays are scalar

```matlab
C = {pi};
tf_cell = isscalar(C);
```

Expected output:

```matlab
tf_cell = logical(1)
```

### Inspecting a GPU-resident tensor

```matlab
G = gpuArray(ones(1,1));
tf_gpu = isscalar(G);
```

Expected output:

```matlab
tf_gpu = logical(1)
```

## FAQ

### Does `isscalar([])` return true?
No. Empty arrays contain zero elements, so `isscalar([])` returns `false`.

### Are string scalars considered scalar even when the text is empty?
Yes. A string scalar is a `1×1` array regardless of its text, so `isscalar("")` returns `true`.

### How does `isscalar` treat GPU arrays?
It checks the metadata exposed by the `GpuTensorHandle`. If the provider omits shape metadata, RunMat
downloads the tensor once to obtain it and then answers on the host.

### What about objects or structs?
Objects and structs are scalar when their array dimensions are `1×1`. The contents do not affect the result.

### Is a 1-by-1-by-1 array scalar?
Yes. Any array whose `numel` equals 1 and whose dimensions are all ones is considered scalar.

### Do character arrays behave differently from strings?
Yes. Character arrays reflect their matrix dimensions. `'abc'` is `1×3`, so `isscalar('abc')` returns `false`.

### Will `isscalar` trigger GPU computation?
No. It is a metadata query and never launches GPU kernels. At most it might gather data when metadata is unavailable.

### Can I rely on `isscalar` inside fused expressions?
Yes. The builtin returns a host logical scalar and the fusion planner treats it as a metadata operation.

## See Also
[isempty](./isempty), [numel](./numel), [size](./size), [gpuArray](./gpuarray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::introspection::isscalar"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isscalar",
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
    notes: "Inspects tensor metadata; downloads handles only when providers omit shapes.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::isscalar"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isscalar",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that returns a host logical scalar for fusion planning.",
};

#[runtime_builtin(
    name = "isscalar",
    category = "array/introspection",
    summary = "Return true when a value has exactly one element and unit dimensions.",
    keywords = "isscalar,scalar,metadata query,gpu,logical",
    accel = "metadata",
    builtin_path = "crate::builtins::array::introspection::isscalar"
)]
fn isscalar_builtin(value: Value) -> Result<Value, String> {
    Ok(Value::Bool(value_is_scalar(&value)))
}

fn value_is_scalar(value: &Value) -> bool {
    if value_numel(value) != 1 {
        return false;
    }
    value_dimensions(value).into_iter().all(|dim| dim == 1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_builtins::{CellArray, CharArray, StructValue, Tensor};

    #[test]
    fn isscalar_numeric_scalar_returns_true() {
        let result = isscalar_builtin(Value::Num(5.0)).expect("isscalar");
        assert_eq!(result, Value::Bool(true));

        let bool_result = isscalar_builtin(Value::Bool(true)).expect("isscalar bool");
        assert_eq!(bool_result, Value::Bool(true));

        let complex_result = isscalar_builtin(Value::Complex(2.0, -3.0)).expect("isscalar complex");
        assert_eq!(complex_result, Value::Bool(true));
    }

    #[test]
    fn isscalar_vector_returns_false() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = isscalar_builtin(Value::Tensor(tensor)).expect("isscalar");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn isscalar_char_array_obeys_dimensions() {
        let single = CharArray::new(vec!['a'], 1, 1).unwrap();
        let row = CharArray::new_row("RunMat");
        let single_result = isscalar_builtin(Value::CharArray(single)).expect("isscalar single");
        let row_result = isscalar_builtin(Value::CharArray(row)).expect("isscalar row");
        assert_eq!(single_result, Value::Bool(true));
        assert_eq!(row_result, Value::Bool(false));
    }

    #[test]
    fn isscalar_string_scalar_true_but_empty_array_false() {
        let scalar = runmat_builtins::StringArray::new(vec!["RunMat".into()], vec![1, 1]).unwrap();
        let empty = runmat_builtins::StringArray::new(Vec::new(), vec![0, 1]).unwrap();
        let scalar_result =
            isscalar_builtin(Value::StringArray(scalar)).expect("isscalar string scalar");
        let empty_result =
            isscalar_builtin(Value::StringArray(empty)).expect("isscalar string empty");
        let string_value_result =
            isscalar_builtin(Value::String("scalar".into())).expect("isscalar string value");
        assert_eq!(scalar_result, Value::Bool(true));
        assert_eq!(empty_result, Value::Bool(false));
        assert_eq!(string_value_result, Value::Bool(true));
    }

    #[test]
    fn isscalar_cell_and_struct_follow_dimensions() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let not_scalar_cell = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let mut struct_scalar = StructValue::new();
        struct_scalar.fields.insert("value".into(), Value::Num(1.0));
        let scalar_cell = isscalar_builtin(Value::Cell(cell)).expect("isscalar cell");
        let nonscalar_cell =
            isscalar_builtin(Value::Cell(not_scalar_cell)).expect("isscalar non-scalar cell");
        let struct_result =
            isscalar_builtin(Value::Struct(struct_scalar)).expect("isscalar struct");
        assert_eq!(scalar_cell, Value::Bool(true));
        assert_eq!(nonscalar_cell, Value::Bool(false));
        assert_eq!(struct_result, Value::Bool(true));
    }

    #[test]
    fn isscalar_gpu_tensor_checks_dimensions() {
        test_support::with_test_provider(|provider| {
            let scalar_tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let vector_tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let scalar_view = runmat_accelerate_api::HostTensorView {
                data: &scalar_tensor.data,
                shape: &scalar_tensor.shape,
            };
            let vector_view = runmat_accelerate_api::HostTensorView {
                data: &vector_tensor.data,
                shape: &vector_tensor.shape,
            };
            let scalar_handle = provider.upload(&scalar_view).expect("upload scalar");
            let vector_handle = provider.upload(&vector_view).expect("upload vector");
            let scalar_result =
                isscalar_builtin(Value::GpuTensor(scalar_handle)).expect("isscalar gpu scalar");
            let vector_result =
                isscalar_builtin(Value::GpuTensor(vector_handle)).expect("isscalar gpu vector");
            assert_eq!(scalar_result, Value::Bool(true));
            assert_eq!(vector_result, Value::Bool(false));
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn isscalar_wgpu_provider_respects_metadata() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        assert_eq!(
            handle.shape,
            vec![1, 1],
            "provider should supply tensor shape"
        );
        let result = isscalar_builtin(Value::GpuTensor(handle)).expect("isscalar");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
