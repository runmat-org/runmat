//! MATLAB-compatible `ismatrix` builtin with GPU-aware semantics for RunMat.

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
        name = "ismatrix",
        builtin_path = "crate::builtins::array::introspection::ismatrix"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ismatrix"
category: "array/introspection"
keywords: ["ismatrix", "matrix detection", "metadata query", "logical", "gpu"]
summary: "Return true when an array has at most two dimensions (m-by-n, including vectors and scalars)."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Inspects tensor metadata directly on GPU handles; falls back to gathering when providers omit shape information."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::introspection::ismatrix::tests"
  integration: "builtins::array::introspection::ismatrix::tests::ismatrix_gpu_tensor_uses_handle_shape"
---

# What does the `ismatrix` function do in MATLAB / RunMat?
`ismatrix(A)` returns logical `true` when the input has two dimensions (`m`-by-`n`) and logical `false`
otherwise. Scalars and vectors are considered matrices because they occupy a 1-by-1 or 1-by-`n` grid.
The builtin mirrors MATLAB semantics across numeric, logical, character, string, cell, struct, and GPU-resident data.

## How does the `ismatrix` function behave in MATLAB / RunMat?
- Returns `true` for scalars, row vectors, column vectors, and ordinary 2-D matrices.
- Returns `false` for arrays whose rank exceeds two, even if trailing dimensions are singleton (for example, `1×1×1`).
- Empty arrays such as `0×0`, `1×0`, and `0×1` are matrices; higher-rank empties such as `0×0×3` are not.
- Character arrays, string arrays, and cell arrays honor their reported MATLAB dimensions.
- GPU tensors rely on their `GpuTensorHandle` metadata; if a provider omits the shape, RunMat gathers once to inspect it.
- Objects, structs, numeric scalars, and logical scalars behave like their underlying dimensions (`1×1`).

## `ismatrix` Function GPU Execution Behaviour
`ismatrix` never launches GPU kernels. For `gpuArray` inputs, the builtin first inspects the shape metadata stored
inside the `GpuTensorHandle`. Providers such as the WGPU backend fully populate that metadata so no data transfer occurs.
If metadata is absent, RunMat gathers the tensor once to recover its dimensions and then evaluates the predicate on the host.
The result is always a host logical scalar, so fusion treats `ismatrix` as a metadata query rather than a device-side operation.

## GPU residency in RunMat (Do I need `gpuArray`?)

You do not need to move data between host and device to call `ismatrix`. The builtin respects existing residency and only
downloads data when the provider fails to report shapes. Users who prefer MATLAB-compatible workflows can still call `gpuArray`
explicitly, but it is not required for this metadata check.

## Examples of using the `ismatrix` function in MATLAB / RunMat

### Confirming that a 2-D matrix returns true

```matlab
tf = ismatrix([1 2 3; 4 5 6]);
```

Expected output:

```matlab
tf = logical(1)
```

### Showing that vectors and scalars count as matrices

```matlab
scalar_tf = ismatrix(42);
row_tf = ismatrix(1:5);
col_tf = ismatrix((1:5)');
```

Expected output:

```matlab
scalar_tf = logical(1)
row_tf = logical(1)
col_tf = logical(1)
```

### Detecting that higher-dimensional arrays are not matrices

```matlab
tf = ismatrix(ones(2,2,3));
```

Expected output:

```matlab
tf = logical(0)
```

### Working with empty arrays

```matlab
tf_empty = ismatrix([]);
tf_row0 = ismatrix(zeros(1,0));
tf_col0 = ismatrix(zeros(0,1));
tf_3d_empty = ismatrix(zeros(0,0,3));
```

Expected output:

```matlab
tf_empty = logical(1)
tf_row0 = logical(1)
tf_col0 = logical(1)
tf_3d_empty = logical(0)
```

### Character and string arrays follow their visible dimensions

```matlab
tf_char = ismatrix('RunMat');
tf_string = ismatrix(["a","b","c"]);
```

Expected output:

```matlab
tf_char = logical(1)
tf_string = logical(1)
```

### Cell and struct arrays respect their grid layout

```matlab
C = {1, 2; 3, 4};
S = repmat(struct("field", 1), 1, 3, 1);
tf_cell = ismatrix(C);
tf_struct = ismatrix(S);
```

Expected output:

```matlab
tf_cell = logical(1)
tf_struct = logical(0)
```

### Inspecting GPU arrays without explicit gathers

```matlab
G = gpuArray(rand(4,4));
tf_gpu = ismatrix(G);
```

Expected output:

```matlab
tf_gpu = logical(1)
```

## FAQ

### Does `ismatrix` treat scalars as matrices?
Yes. Scalars occupy a 1-by-1 grid, which is a valid two-dimensional matrix in MATLAB semantics.

### Are vectors considered matrices?
Yes. Row and column vectors are 1-by-`N` and `N`-by-1 matrices, so `ismatrix` returns `true`.

### Do trailing singleton dimensions change the result?
Yes. Any array whose rank exceeds two returns `false`, even if the extra dimensions are size one (for example, `1×1×1`).

### How are empty arrays handled?
Empty arrays with at most two dimensions (such as `0×0`, `1×0`, or `0×1`) return `true`. Higher-rank empties return `false`.

### What about character, string, or cell arrays?
They follow their MATLAB-visible dimensions. A character row vector or 1-by-`N` string array returns `true`, whereas a 1-by-1-by-`N` cell array returns `false`.

### Will `ismatrix` download GPU data?
Only when the active provider fails to populate shape metadata on the `GpuTensorHandle`. Providers that supply the shape avoid any data transfer.

### Does `ismatrix` participate in fusion or GPU kernels?
No. The builtin is a metadata query, always returns a host logical scalar, and never dispatches GPU work.

### How does `ismatrix` relate to `isvector` and `isscalar`?
`isscalar(A)` and `isvector(A)` both imply `ismatrix(A)` because they describe special cases of 2-D arrays. The reverse implication does not necessarily hold.

### Are objects and structs supported?
Yes. Objects and structs are treated as 1-by-1 unless they represent arrays of instances, in which case the reported dimensions determine the answer.

### Does sparse support change the behaviour?
Future sparse tensors will report their dimensions through the same metadata, so the `ismatrix` predicate will remain unchanged.

## See Also
[isscalar](./isscalar), [isvector](./isvector), [ndims](./ndims), [size](./size), [gpuArray](./gpuarray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::introspection::ismatrix"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ismatrix",
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
    notes: "Consumes tensor shape metadata; falls back to gathering only when providers omit shape information.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::ismatrix"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ismatrix",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that always yields a host logical scalar; fusion treats it as a control predicate.",
};

#[runtime_builtin(
    name = "ismatrix",
    category = "array/introspection",
    summary = "Return true when an array has at most two dimensions (m-by-n, including vectors and scalars).",
    keywords = "ismatrix,matrix detection,metadata query,logical,gpu",
    accel = "metadata",
    builtin_path = "crate::builtins::array::introspection::ismatrix"
)]
fn ismatrix_builtin(value: Value) -> Result<Value, String> {
    Ok(Value::Bool(value_is_matrix(&value)))
}

fn value_is_matrix(value: &Value) -> bool {
    value_dimensions(value).len() <= 2
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{
        CellArray, CharArray, LogicalArray, ObjectInstance, StringArray, StructValue, Tensor,
    };

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_accepts_scalars_vectors_and_matrices() {
        let scalar = ismatrix_builtin(Value::Num(5.0)).expect("ismatrix scalar");
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let col = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let row_result = ismatrix_builtin(Value::Tensor(row)).expect("ismatrix row");
        let col_result = ismatrix_builtin(Value::Tensor(col)).expect("ismatrix col");
        let matrix_result = ismatrix_builtin(Value::Tensor(matrix)).expect("ismatrix matrix");
        assert_eq!(scalar, Value::Bool(true));
        assert_eq!(row_result, Value::Bool(true));
        assert_eq!(col_result, Value::Bool(true));
        assert_eq!(matrix_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_rejects_higher_rank_arrays() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let result = ismatrix_builtin(Value::Tensor(tensor)).expect("ismatrix");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_handles_empty_dimensions_like_matlab() {
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let row_empty = Tensor::new(Vec::new(), vec![1, 0]).unwrap();
        let col_empty = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let empty_3d = Tensor::new(Vec::new(), vec![0, 0, 3]).unwrap();
        let empty_result = ismatrix_builtin(Value::Tensor(empty)).expect("ismatrix []");
        let row_result = ismatrix_builtin(Value::Tensor(row_empty)).expect("ismatrix 1x0");
        let col_result = ismatrix_builtin(Value::Tensor(col_empty)).expect("ismatrix 0x1");
        let empty_3d_result = ismatrix_builtin(Value::Tensor(empty_3d)).expect("ismatrix 0x0x3");
        assert_eq!(empty_result, Value::Bool(true));
        assert_eq!(row_result, Value::Bool(true));
        assert_eq!(col_result, Value::Bool(true));
        assert_eq!(empty_3d_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_handles_scalar_like_runtime_values() {
        let bool_result = ismatrix_builtin(Value::Bool(true)).expect("ismatrix bool");
        let string_result =
            ismatrix_builtin(Value::String("runmat".into())).expect("ismatrix string");
        let func_result =
            ismatrix_builtin(Value::FunctionHandle("sin".into())).expect("ismatrix function");
        let object = Value::Object(ObjectInstance::new("TestClass".into()));
        let object_result = ismatrix_builtin(object).expect("ismatrix object");
        assert_eq!(bool_result, Value::Bool(true));
        assert_eq!(string_result, Value::Bool(true));
        assert_eq!(func_result, Value::Bool(true));
        assert_eq!(object_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_logical_arrays_respect_shape_rank() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).expect("logical array");
        let logical_result =
            ismatrix_builtin(Value::LogicalArray(logical)).expect("ismatrix logical");
        assert_eq!(logical_result, Value::Bool(true));

        let logical3d =
            LogicalArray::new(vec![0, 1, 0, 1], vec![1, 1, 4]).expect("logical 3d array");
        let logical3d_result =
            ismatrix_builtin(Value::LogicalArray(logical3d)).expect("ismatrix logical 3d");
        assert_eq!(logical3d_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_char_string_cell_and_struct_metadata() {
        let char_array = CharArray::new_row("RunMat");
        let string_array = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![1, 3])
            .expect("string array");
        let cell_array =
            CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).expect("cell array");
        let mut struct_value = StructValue::new();
        struct_value.fields.insert("field".into(), Value::Num(1.0));
        let struct_vector = CellArray::new(
            vec![
                Value::Struct(struct_value.clone()),
                Value::Struct(struct_value.clone()),
            ],
            1,
            2,
        )
        .expect("struct array handles");
        let char_result = ismatrix_builtin(Value::CharArray(char_array)).expect("ismatrix char");
        let string_result =
            ismatrix_builtin(Value::StringArray(string_array)).expect("ismatrix string");
        let cell_result = ismatrix_builtin(Value::Cell(cell_array)).expect("ismatrix cell");
        let struct_result =
            ismatrix_builtin(Value::Cell(struct_vector)).expect("ismatrix struct array");
        assert_eq!(char_result, Value::Bool(true));
        assert_eq!(string_result, Value::Bool(true));
        assert_eq!(cell_result, Value::Bool(true));
        assert_eq!(struct_result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_rejects_struct_arrays_with_extra_dimensions() {
        let mut struct_value = StructValue::new();
        struct_value.fields.insert("field".into(), Value::Num(1.0));
        let handles = vec![
            Value::Struct(struct_value.clone()),
            Value::Struct(struct_value.clone()),
        ];
        let array = CellArray::new(handles, 1, 2).expect("cell array");
        let nested = Tensor::new(vec![0.0; 2], vec![1, 1, 2]).unwrap();
        let array_result = ismatrix_builtin(Value::Cell(array)).expect("ismatrix cell");
        let nested_result = ismatrix_builtin(Value::Tensor(nested)).expect("ismatrix nested");
        assert_eq!(array_result, Value::Bool(true));
        assert_eq!(nested_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_gpu_tensor_uses_handle_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
            let tensor3d = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
            let tensor_view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let tensor3d_view = runmat_accelerate_api::HostTensorView {
                data: &tensor3d.data,
                shape: &tensor3d.shape,
            };
            let handle = provider.upload(&tensor_view).expect("upload matrix");
            let handle3d = provider.upload(&tensor3d_view).expect("upload 3d");
            let result = ismatrix_builtin(Value::GpuTensor(handle)).expect("ismatrix gpu");
            let result3d = ismatrix_builtin(Value::GpuTensor(handle3d)).expect("ismatrix 3d gpu");
            assert_eq!(result, Value::Bool(true));
            assert_eq!(result3d, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_gpu_tensor_vector_shape_is_matrix() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload vector");
            assert_eq!(handle.shape, vec![3]);
            let result = ismatrix_builtin(Value::GpuTensor(handle)).expect("ismatrix gpu vector");
            assert_eq!(result, Value::Bool(true));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ismatrix_gpu_handle_without_shape_falls_back() {
        test_support::with_test_provider(|_| {
            let handle = runmat_accelerate_api::GpuTensorHandle {
                shape: Vec::new(),
                device_id: 0,
                buffer_id: u64::MAX,
            };
            let result = ismatrix_builtin(Value::GpuTensor(handle)).expect("ismatrix gpu fallback");
            assert_eq!(result, Value::Bool(true));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn value_is_matrix_matches_dimensions_helper() {
        let tensor = Tensor::new(vec![0.0; 12], vec![3, 4]).unwrap();
        assert!(value_is_matrix(&Value::Tensor(tensor)));
        let higher = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        assert!(!value_is_matrix(&Value::Tensor(higher)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ismatrix_wgpu_provider_populates_shape() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        assert_eq!(handle.shape, vec![2, 2]);
        let result = ismatrix_builtin(Value::GpuTensor(handle)).expect("ismatrix");
        assert_eq!(result, Value::Bool(true));
    }
}
