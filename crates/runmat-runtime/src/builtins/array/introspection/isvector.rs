//! MATLAB-compatible `isvector` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_dimensions;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "isvector",
        builtin_path = "crate::builtins::array::introspection::isvector"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "isvector"
category: "array/introspection"
keywords: ["isvector", "vector detection", "metadata query", "gpu", "logical"]
summary: "Return true when an array is 1-by-N or N-by-1 (including scalars)."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Reads tensor metadata directly from GPU handles; gathers only when shape metadata is unavailable."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::introspection::isvector::tests"
  integration: "builtins::array::introspection::isvector::tests::isvector_gpu_tensor_uses_handle_shape"
---

# What does the `isvector` function do in MATLAB / RunMat?
`isvector(A)` returns logical `true` when the input is 1-by-`N` or `N`-by-1 (including scalars) and has at most two dimensions.
The builtin mirrors MATLAB semantics across numeric arrays, logical arrays, characters,
strings, cells, structs, objects, and GPU-resident tensors.

## How does the `isvector` function behave in MATLAB / RunMat?
- `isvector` is `true` for 1-by-`N` and `N`-by-1 arrays (with `N ≥ 0`), including scalars (`1×1`).
- Arrays with more than two dimensions always return `false`, even when all trailing dimensions are singleton (for example, `1×1×1`).
- Empty arrays follow MATLAB rules: size `0×1` or `1×0` is a vector, but size `0×3` is not.
- Character arrays, string arrays, cell arrays, structs, objects, and handle objects follow the same dimension check.
- GPU tensors use metadata stored in their `GpuTensorHandle`. If metadata is missing, RunMat gathers once to inspect dimensions.
- Sparse matrices currently follow dense semantics; when sparse support lands, `isvector` will continue to rely on reported dimensions.

## `isvector` Function GPU Execution Behaviour
`isvector` never launches GPU kernels. For `gpuArray` inputs the builtin first inspects the shape
metadata on the `GpuTensorHandle`. If a provider omits that metadata, RunMat downloads the tensor
once to obtain it and then computes the result on the host. The builtin always returns a host logical
scalar, so fusion planning treats it as a metadata query instead of a device-side kernel.

## GPU residency in RunMat (Do I need `gpuArray`?)

`isvector` performs its checks on whatever residency the input already uses. It will happily accept a
GPU tensor without the caller moving it to the CPU first, because the builtin only needs the tensor's
metadata. When a provider fully populates tensor shapes (as the WGPU provider does), the runtime
never downloads the data. If metadata is missing, the runtime gathers once to recover the shape and
then continues on the host. Users can still call `gpuArray` for compatibility with MathWorks MATLAB,
but it is not required for this builtin.

## Examples of using the `isvector` function in MATLAB / RunMat

### Checking if a row vector input is a vector

```matlab
tf = isvector([1 2 3]);
```

Expected output:

```matlab
tf = logical(1)
```

### Detecting that a matrix is not a vector

```matlab
tf = isvector([1 2 3; 4 5 6]);
```

Expected output:

```matlab
tf = logical(0)
```

### Verifying that a scalar counts as a vector

```matlab
tf = isvector(42);
```

Expected output:

```matlab
tf = logical(1)
```

### Handling empty dimensions exactly like MATLAB

```matlab
tf_zero_col = isvector(zeros(1,0));
tf_zero_row = isvector(zeros(0,1));
tf_zero_by_three = isvector(zeros(0,3));
```

Expected output:

```matlab
tf_zero_col = logical(1)
tf_zero_row = logical(1)
tf_zero_by_three = logical(0)
```

### Working with character and string arrays

```matlab
tf_char = isvector('RunMat');
tf_string_row = isvector(["a","b","c"]);
```

Expected output:

```matlab
tf_char = logical(1)
tf_string_row = logical(1)
```

### Confirming GPU tensor metadata without gathering

```matlab
G = gpuArray((1:5)');
tf_gpu = isvector(G);
```

Expected output:

```matlab
tf_gpu = logical(1)
```

### Rejecting higher-dimensional arrays

```matlab
tf = isvector(ones(1,1,4));
```

Expected output:

```matlab
tf = logical(0)
```

### Trailing singleton dimensions are still rejected

```matlab
tf = isvector(ones(1,1,1));
```

Expected output:

```matlab
tf = logical(0)
```

## FAQ

### Does `isvector` treat scalars as vectors?
Yes. Scalars are 1-by-1 arrays, so they are vectors.

### How does `isvector` handle empty dimensions?
It follows MATLAB rules: size 0-by-1 or 1-by-0 is a vector, but other empty shapes such as 0-by-3 return `false`.

### Are higher-dimensional arrays ever considered vectors?
No. Any array with more than two dimensions returns `false`, even if the trailing dimensions equal one (for example, `1×1×N`).

### Do character vectors and string scalars count as vectors?
Yes. Character vectors are 1-by-N arrays, and string scalars are 1-by-1 string arrays, so both return `true`.

### What happens with GPU arrays?
RunMat reads the shape metadata from the `GpuTensorHandle`. If the provider does not populate that metadata, the runtime gathers the tensor once to inspect its dimensions.

### Are cell arrays and structs supported?
Yes. `isvector` uses the MATLAB-visible dimensions, so any 1-by-N or N-by-1 cell/struct array returns `true`.

### Will `isvector` launch GPU kernels?
No. The builtin is purely a metadata query and never dispatches GPU work.

### Can I rely on `isvector` inside fused expressions?
Yes. The builtin returns a host logical scalar, so fusion planning treats it as a metadata operation.

### Does sparse support change the semantics?
Sparse inputs will follow the same dimension check once sparse tensors are introduced; no behavioural change is expected.

### How does `isvector` interact with `isscalar` and `ismatrix`?
`isscalar(A)` implies `isvector(A)` but not vice versa. Conversely, `ismatrix(A)` can be `true` even when `isvector(A)` is `false`.

## See Also
[isscalar](./isscalar), [isempty](./isempty), [length](./length), [numel](./numel), [size](./size), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)
"#;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::introspection::isvector"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isvector",
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
    notes: "Reads tensor metadata; falls back to gathering when providers omit shape information.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::isvector"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isvector",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that always returns a host logical scalar for fusion planning.",
};

#[runtime_builtin(
    name = "isvector",
    category = "array/introspection",
    summary = "Return true when an array is 1-by-N or N-by-1 (including scalars).",
    keywords = "isvector,vector detection,metadata query,gpu,logical",
    accel = "metadata",
    builtin_path = "crate::builtins::array::introspection::isvector"
)]
fn isvector_builtin(value: Value) -> Result<Value, String> {
    Ok(Value::Bool(value_is_vector(&value)))
}

fn value_is_vector(value: &Value) -> bool {
    let dims = value_dimensions(value);
    if dims.len() > 2 {
        return false;
    }
    let mut non_singleton_dims = 0usize;

    for &dim in dims.iter() {
        if dim != 1 {
            non_singleton_dims += 1;
            if non_singleton_dims > 1 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_builtins::{CellArray, CharArray, Tensor};

    #[test]
    fn isvector_detects_row_and_column_vectors() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let col = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let row_result = isvector_builtin(Value::Tensor(row)).expect("isvector row");
        let col_result = isvector_builtin(Value::Tensor(col)).expect("isvector col");
        assert_eq!(row_result, Value::Bool(true));
        assert_eq!(col_result, Value::Bool(true));
    }

    #[test]
    fn isvector_rejects_matrices_and_higher_dimensions() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let cube = Tensor::new(vec![0.0; 4], vec![1, 1, 4]).unwrap();
        let matrix_result = isvector_builtin(Value::Tensor(matrix)).expect("isvector matrix");
        let cube_result = isvector_builtin(Value::Tensor(cube)).expect("isvector cube");
        assert_eq!(matrix_result, Value::Bool(false));
        assert_eq!(cube_result, Value::Bool(false));
    }

    #[test]
    fn isvector_counts_scalars_and_empty_one_dimensional_arrays() {
        let scalar_result = isvector_builtin(Value::Num(5.0)).expect("isvector scalar");
        let empty_row = Tensor::new(Vec::new(), vec![1, 0]).unwrap();
        let empty_col = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let empty_wide = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let row_result = isvector_builtin(Value::Tensor(empty_row)).expect("isvector 1x0");
        let col_result = isvector_builtin(Value::Tensor(empty_col)).expect("isvector 0x1");
        let wide_result = isvector_builtin(Value::Tensor(empty_wide)).expect("isvector 0x3");
        assert_eq!(scalar_result, Value::Bool(true));
        assert_eq!(row_result, Value::Bool(true));
        assert_eq!(col_result, Value::Bool(true));
        assert_eq!(wide_result, Value::Bool(false));
    }

    #[test]
    fn isvector_char_and_cell_arrays_follow_dimensions() {
        let char_row = CharArray::new_row("RunMat");
        let char_matrix = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let cell_vector = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let cell_matrix = CellArray::new(
            vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::Num(4.0),
            ],
            2,
            2,
        )
        .unwrap();
        let char_row_result = isvector_builtin(Value::CharArray(char_row)).expect("isvector char");
        let char_matrix_result =
            isvector_builtin(Value::CharArray(char_matrix)).expect("isvector char matrix");
        let cell_vector_result = isvector_builtin(Value::Cell(cell_vector)).expect("isvector cell");
        let cell_matrix_result =
            isvector_builtin(Value::Cell(cell_matrix)).expect("isvector cell matrix");
        assert_eq!(char_row_result, Value::Bool(true));
        assert_eq!(char_matrix_result, Value::Bool(false));
        assert_eq!(cell_vector_result, Value::Bool(true));
        assert_eq!(cell_matrix_result, Value::Bool(false));
    }

    #[test]
    fn isvector_trailing_singleton_dimensions_are_rejected() {
        let scalar_with_extra = Tensor::new(vec![5.0], vec![1, 1, 1]).unwrap();
        let result =
            isvector_builtin(Value::Tensor(scalar_with_extra)).expect("isvector trailing ones");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn isvector_gpu_tensor_uses_handle_shape() {
        test_support::with_test_provider(|provider| {
            let vector = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let vector_view = runmat_accelerate_api::HostTensorView {
                data: &vector.data,
                shape: &vector.shape,
            };
            let matrix_view = runmat_accelerate_api::HostTensorView {
                data: &matrix.data,
                shape: &matrix.shape,
            };
            let vector_handle = provider.upload(&vector_view).expect("upload vector");
            let matrix_handle = provider.upload(&matrix_view).expect("upload matrix");
            let vector_result =
                isvector_builtin(Value::GpuTensor(vector_handle)).expect("isvector gpu vector");
            let matrix_result =
                isvector_builtin(Value::GpuTensor(matrix_handle)).expect("isvector gpu matrix");
            assert_eq!(vector_result, Value::Bool(true));
            assert_eq!(matrix_result, Value::Bool(false));
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn isvector_wgpu_provider_populates_shape() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        assert_eq!(
            handle.shape,
            vec![3, 1],
            "provider should supply tensor shape metadata"
        );
        let result = isvector_builtin(Value::GpuTensor(handle)).expect("isvector");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
