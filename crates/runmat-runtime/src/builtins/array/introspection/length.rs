//! MATLAB-compatible `length` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_dimensions;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::containers::map::map_length;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "length",
        builtin_path = "crate::builtins::array::introspection::length"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "length"
category: "array/introspection"
keywords: ["length", "largest dimension", "vector length", "gpu metadata", "array size"]
summary: "Return the length of the largest dimension of scalars, vectors, matrices, and N-D arrays."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Reads tensor metadata from handles; falls back to host when provider metadata is incomplete."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::introspection::length::tests"
  integration: "builtins::array::introspection::length::tests::length_gpu_tensor_reads_shape"
---

# What does the `length` function do in MATLAB / RunMat?
`length(A)` returns the length of the largest dimension of `A`. Scalars report `1`, column vectors
report their row count, row vectors return their column count, and rectangular matrices yield the
larger of their row or column dimensions. For higher-rank arrays, the function returns the maximum
extent across every dimension.

## How does the `length` function behave in MATLAB / RunMat?
- The result is always a double-precision scalar.
- Empty arrays report `0` when every dimension is zero; otherwise the maximum non-zero dimension is
  reported (e.g., `0×5` arrays have length `5`).
- Character arrays and string arrays follow their array dimensions, not the number of code points.
- Cell arrays use their MATLAB array shape (`rows × cols`) with the maximum dimension returned.
- GPU-resident arrays are inspected without gathering when the provider populates shape metadata on
  the tensor handle; otherwise, RunMat gathers once to recover the correct dimensions.
- Arguments that are not arrays (scalars, logicals, strings, handle objects) are treated as `1×1`.

## `length` Function GPU Execution Behaviour
`length` is a metadata query. When the input is a GPU tensor, RunMat looks at the shape embedded in
the GPU handle (`GpuTensorHandle.shape`). If the active provider leaves that metadata empty, the
runtime invokes `provider.download(handle)` a single time to recover the shape, ensuring
MATLAB-compatible behaviour. No GPU kernels are launched and no device buffers are allocated by this
builtin.

## Examples of using the `length` function in MATLAB / RunMat

### Determine the length of a row vector

```matlab
row = [1 2 3 4];
n = length(row);
```

Expected output:

```matlab
n = 4;
```

### Find the longer side of a rectangular matrix

```matlab
A = randn(5, 12);
len = length(A);
```

Expected output:

```matlab
len = 12;
```

### Handle empty arrays that still have a non-zero dimension

```matlab
E = zeros(0, 7);
len = length(E);
```

Expected output:

```matlab
len = 7;
```

### Measure the length of a character array

```matlab
name = 'RunMat';
len = length(name);
```

Expected output:

```matlab
len = 6;
```

### Inspect the length of a gpuArray without gathering

```matlab
G = gpuArray(ones(256, 4));
len = length(G);
```

Expected output:

```matlab
len = 256;
```

## FAQ

### How is `length` different from `size`?
`length(A)` returns the maximum dimension length as a scalar, whereas `size(A)` returns every
dimension in a row vector. Use `length` when you only care about the longest dimension.

### What does `length` return for scalars?
Scalars are treated as `1×1`, so `length(scalar)` returns `1`.

### How does `length` behave for empty arrays?
If all dimensions are zero, `length` returns `0`. If any dimension is non-zero, the maximum
dimension is returned. For example, `zeros(0, 5)` has length `5`, but `zeros(0, 0)` has length `0`.

### Does `length` gather GPU data?
No. The runtime relies on shape metadata stored in the GPU tensor handle. Only when that metadata is
missing does RunMat gather the tensor to maintain correctness.

### Can I use `length` on cell arrays and structs?
Yes. `length` examines the MATLAB array shape of the container, so cell arrays and struct arrays
return the maximum dimension of their array layout.

### Does `length` count characters or bytes?
For character arrays, the length reflects the array dimensions (rows and columns), not encoded byte
length or Unicode scalar counts.

### What about string arrays?
String arrays are treated like any other array. String scalars are `1×1`, so `length("abc")`
returns `1`. Use `strlength` if you need the number of characters in each element.

### Is `length` safe to use inside fused GPU expressions?
Yes. `length` never allocates or keeps data on the GPU. It returns a host scalar immediately, so it
won't break fusion plans.

## See Also
[size](./size), [numel (MathWorks)](https://www.mathworks.com/help/matlab/ref/numel.html), [strlength (MathWorks)](https://www.mathworks.com/help/matlab/ref/strlength.html)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::length")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "length",
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
    notes: "Reads tensor metadata from handles; falls back to gathering only when provider metadata is absent.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::length"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "length",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query; fusion planner treats this as a host scalar lookup.",
};

#[runtime_builtin(
    name = "length",
    category = "array/introspection",
    summary = "Return the length of the largest dimension of scalars, vectors, matrices, and N-D arrays.",
    keywords = "length,largest dimension,vector length,gpu metadata,array size",
    accel = "metadata",
    builtin_path = "crate::builtins::array::introspection::length"
)]
fn length_builtin(value: Value) -> Result<Value, String> {
    if let Some(count) = map_length(&value) {
        return Ok(Value::Num(count as f64));
    }
    let len = max_dimension(&value) as f64;
    Ok(Value::Num(len))
}

fn max_dimension(value: &Value) -> usize {
    let dims = value_dimensions(value);
    dims.into_iter().max().unwrap_or(0)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{
        CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
    };

    #[test]
    fn length_scalar_is_one() {
        let result = length_builtin(Value::Num(5.0)).expect("length");
        assert_eq!(result, Value::Num(1.0));
    }

    #[test]
    fn length_column_vector_uses_rows() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(3.0));
    }

    #[test]
    fn length_matrix_returns_larger_dimension() {
        let tensor = Tensor::new(vec![0.0; 10], vec![2, 5]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(5.0));
    }

    #[test]
    fn length_high_rank_tensor_reports_global_max() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(4.0));
    }

    #[test]
    fn length_partial_empty_tensor_returns_max_dimension() {
        let tensor = Tensor::new(vec![], vec![0, 0, 5]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(5.0));
    }

    #[test]
    fn length_empty_matrix_with_nonzero_dimension() {
        let tensor = Tensor::new(vec![], vec![0, 7]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(7.0));
    }

    #[test]
    fn length_fully_empty_matrix_returns_zero() {
        let tensor = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = length_builtin(Value::Tensor(tensor)).expect("length");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn length_character_array_uses_shape() {
        let chars = CharArray::new_row("RunMat");
        let result = length_builtin(Value::CharArray(chars)).expect("length");
        assert_eq!(result, Value::Num(6.0));
    }

    #[test]
    fn length_complex_tensor_uses_shape() {
        let complex = ComplexTensor::new(vec![(0.0, 0.0); 12], vec![3, 4]).unwrap();
        let result = length_builtin(Value::ComplexTensor(complex)).expect("length");
        assert_eq!(result, Value::Num(4.0));
    }

    #[test]
    fn length_cell_array_respects_dimensions() {
        let cells = CellArray::new(
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
        let result = length_builtin(Value::Cell(cells)).expect("length");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn length_string_array_defaults_to_shape() {
        let sa = StringArray::new(vec!["a".into(), "bb".into()], vec![2, 1]).unwrap();
        let result = length_builtin(Value::StringArray(sa)).expect("length");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn length_logical_array_uses_shape() {
        let la = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = length_builtin(Value::LogicalArray(la)).expect("length");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn length_gpu_tensor_reads_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..12).map(|x| x as f64).collect(), vec![3, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = length_builtin(Value::GpuTensor(handle)).expect("length");
            assert_eq!(result, Value::Num(4.0));
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
    #[test]
    #[cfg(feature = "wgpu")]
    fn length_wgpu_tensor_uses_handle_shape() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new((0..24).map(|v| v as f64).collect(), vec![6, 4]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = length_builtin(Value::GpuTensor(handle)).expect("length");
        assert_eq!(result, Value::Num(6.0));
    }
}
