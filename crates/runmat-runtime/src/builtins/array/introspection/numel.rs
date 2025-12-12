//! MATLAB-compatible `numel` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::{value_dimensions, value_numel};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "numel",
        builtin_path = "crate::builtins::array::introspection::numel"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "numel"
category: "array/introspection"
keywords: ["numel", "number of elements", "array length", "gpu metadata", "dimensions"]
summary: "Count the number of elements in scalars, vectors, matrices, and N-D arrays."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Reads tensor metadata from handles; falls back to host inspection when provider metadata is incomplete."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::introspection::numel::tests"
  integration: "builtins::array::introspection::numel::tests::numel_gpu_tensor_uses_shape"
---

# What does the `numel` function do in MATLAB / RunMat?
`numel(A)` returns the number of elements contained in `A`, matching MATLAB semantics for dense
arrays, logical arrays, cell arrays, structs, strings, and character arrays. The result is always a
double-precision scalar.

## How does the `numel` function behave in MATLAB / RunMat?
- `numel(A)` counts every stored element; for matrices this is `prod(size(A))`.
- `numel(A, dim1, dim2, ...)` multiplies the extents of the requested dimensions. Each dimension
  argument must be a positive integer. Dimensions beyond the array rank contribute a factor of `1`.
- Passing a numeric vector of dimensions (for example `numel(A, [1 3])`) is equivalent to listing
  them separately.
- Scalars return `1`, character arrays report `rows × cols`, and cell arrays count the number of
  cells.
- Empty arrays return `0` when any dimension is zero.
- GPU-resident arrays use metadata stored on the device handle; when unavailable, RunMat gathers the
  data to maintain correctness.

## `numel` Function GPU Execution Behaviour
`numel` does not launch GPU kernels or register provider hooks. When the input is a GPU tensor, the
runtime consults the shape metadata stored in the handle to compute the count. If the active
provider omits this metadata, RunMat downloads the tensor once to recover the correct answer. The
builtin always returns a host double scalar and never allocates device memory, so it can safely be
used inside fused GPU expressions without breaking residency.

## Examples of using the `numel` function in MATLAB / RunMat

### Counting elements in a matrix

```matlab
A = [1 2 3; 4 5 6];
n = numel(A);
```

Expected output:

```matlab
n = 6;
```

### Checking the number of elements in a cell array

```matlab
C = {1, 2, 3; 4, 5, 6};
cells = numel(C);
```

Expected output:

```matlab
cells = 6;
```

### Getting the number of elements along selected dimensions

```matlab
T = rand(4, 3, 2);
plane = numel(T, 1, 2);   % product of the first two dimensions
```

Expected output:

```matlab
plane = 12;
```

### Using `numel` with gpuArray data

```matlab
G = gpuArray(ones(256, 4));
count = numel(G);
```

Expected output:

```matlab
count = 1024;
```

### Confirming the number of characters in a char array

```matlab
name = ['R' 'u' 'n' 'M' 'a' 't'];
chars = numel(name);
```

Expected output:

```matlab
chars = 6;
```

## FAQ

### How is `numel` different from `length`?
`numel(A)` counts every element in `A`, whereas `length(A)` returns the size of the largest single
dimension.

### What happens when I specify dimensions?
`numel(A, dim1, dim2, ...)` multiplies the lengths of the listed dimensions. Missing dimensions (for
example, requesting the fourth dimension of a matrix) contribute a factor of `1`, matching MATLAB.

### Does `numel` gather GPU data?
Usually no. The runtime reads shape metadata from the GPU tensor handle. It gathers only when the
provider fails to populate that metadata.

### What does `numel` return for empty arrays?
If any dimension is zero, `numel` returns `0`. This behaviour matches MATLAB for empty matrices,
cell arrays, and logical arrays.

### Can I use `numel` on strings and character arrays?
Yes. Character arrays return the total number of characters (`rows × cols`). String arrays are
treated as ordinary arrays with MATLAB-compatible dimensions.

### Is `numel` safe inside fused GPU expressions?
Yes. The builtin returns a host double scalar and does not allocate device buffers, so fusion plans
remain intact.

### Does `numel` accept non-integer dimensions?
No. Dimension arguments must be positive integers. Fractional, negative, or non-numeric values raise
an error that mirrors MATLAB.

## See Also
[size](./size), [length](./length), [MathWorks numel reference](https://www.mathworks.com/help/matlab/ref/numel.html), [MathWorks size reference](https://www.mathworks.com/help/matlab/ref/size.html)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::introspection::numel")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "numel",
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
    notes:
        "Counts elements using tensor metadata; gathers once only if provider metadata is missing.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::introspection::numel"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "numel",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query; fusion planner treats this builtin as a host scalar.",
};

#[runtime_builtin(
    name = "numel",
    category = "array/introspection",
    summary = "Count the number of elements in scalars, vectors, matrices, and N-D arrays.",
    keywords = "numel,number of elements,array length,gpu metadata,dimensions",
    accel = "metadata",
    builtin_path = "crate::builtins::array::introspection::numel"
)]
fn numel_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return Ok(Value::Num(value_numel(&value) as f64));
    }

    let dims = parse_dimension_args(&rest)?;
    let shape = value_dimensions(&value);

    let mut product = 1usize;
    for dim in dims {
        let extent = dimension_extent(&shape, dim);
        product = product.saturating_mul(extent);
    }

    Ok(Value::Num(product as f64))
}

fn parse_dimension_args(args: &[Value]) -> Result<Vec<usize>, String> {
    let mut dims = Vec::new();
    for arg in args {
        match arg {
            Value::Int(_) | Value::Num(_) => {
                dims.push(tensor::parse_dimension(arg, "numel")?);
            }
            Value::Tensor(t) => {
                ensure_dim_vector(t)?;
                if t.data.is_empty() {
                    return Err(
                        "numel: dimension vector must contain at least one element".to_string()
                    );
                }
                let parsed = t
                    .data
                    .iter()
                    .map(|&raw| parse_dim_scalar(raw))
                    .collect::<Result<Vec<_>, _>>()?;
                dims.extend(parsed);
            }
            _ => {
                return Err(
                    "numel: dimension arguments must be numeric scalars or vectors".to_string(),
                );
            }
        }
    }
    if dims.is_empty() {
        return Err("numel: dimension list must contain at least one element".to_string());
    }
    Ok(dims)
}

fn ensure_dim_vector(t: &Tensor) -> Result<(), String> {
    let non_unit = t.shape.iter().filter(|&&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err("numel: dimension vector must be a vector of positive integers".to_string())
    }
}

fn parse_dim_scalar(raw: f64) -> Result<usize, String> {
    if !raw.is_finite() {
        return Err("numel: dimension must be finite".to_string());
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > f64::EPSILON {
        return Err("numel: dimension must be an integer".to_string());
    }
    if rounded < 1.0 {
        return Err("numel: dimension must be >= 1".to_string());
    }
    Ok(rounded as usize)
}

fn dimension_extent(dimensions: &[usize], dim: usize) -> usize {
    dimensions.get(dim.saturating_sub(1)).copied().unwrap_or(1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CellArray, CharArray, Tensor};

    #[test]
    fn numel_scalar_is_one() {
        let result = numel_builtin(Value::Num(42.0), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(1.0));
    }

    #[test]
    fn numel_matrix_counts_elements() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = numel_builtin(Value::Tensor(tensor), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(4.0));
    }

    #[test]
    fn numel_cell_array_counts_cells() {
        let cells = vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let cell_array = CellArray::new(cells, 2, 2).unwrap();
        let result = numel_builtin(Value::Cell(cell_array), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(4.0));
    }

    #[test]
    fn numel_char_array_counts_characters() {
        let chars = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
        let result = numel_builtin(Value::CharArray(chars), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(6.0));
    }

    #[test]
    fn numel_selected_dimensions_multiplies_extents() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let args = vec![Value::from(1.0), Value::from(2.0)];
        let result = numel_builtin(Value::Tensor(tensor), args).expect("numel");
        assert_eq!(result, Value::Num(6.0));
    }

    #[test]
    fn numel_dimension_vector_argument_supported() {
        let tensor = Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result =
            numel_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("numel");
        assert_eq!(result, Value::Num(8.0));
    }

    #[test]
    fn numel_gpu_tensor_uses_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0; 12], vec![3, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = numel_builtin(Value::GpuTensor(handle), Vec::new()).expect("numel");
            assert_eq!(result, Value::Num(12.0));
        });
    }

    #[test]
    fn numel_dimension_must_be_positive_integer() {
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let err = numel_builtin(Value::Tensor(tensor), vec![Value::from(0.0)])
            .expect_err("expected dimension error");
        assert!(
            err.contains("dimension must be >= 1"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn numel_dimension_vector_requires_vector_shape() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = numel_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)])
            .expect_err("expected vector shape error");
        assert!(
            err.contains("dimension vector must be a vector"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn numel_dimension_arguments_must_be_numeric() {
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let err = numel_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")])
            .expect_err("expected numeric argument error");
        assert!(
            err.contains("dimension arguments must be numeric"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn numel_wgpu_counts_elements() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0; 18], vec![3, 3, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let result = numel_builtin(Value::GpuTensor(handle), Vec::new()).expect("numel");
        assert_eq!(result, Value::Num(18.0));
    }
}
