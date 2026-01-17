//! MATLAB-compatible `sort` builtin with multi-output and GPU-aware semantics.

use std::cmp::Ordering;

use runmat_accelerate_api::{
    GpuTensorHandle, SortComparison as ProviderSortComparison, SortOrder as ProviderSortOrder,
};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::build_runtime_error;
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "sort",
        builtin_path = "crate::builtins::array::sorting_sets::sort"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "sort"
category: "array/sorting_sets"
keywords: ["sort", "ascending", "descending", "indices", "comparisonmethod", "gpu"]
summary: "Sort scalars, vectors, matrices, or N-D tensors along a dimension, with optional index outputs."
references:
  - https://www.mathworks.com/help/matlab/ref/sort.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses the provider `sort_dim` hook when available; otherwise tensors are gathered and sorted on the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::sorting_sets::sort::tests"
  integration: "builtins::array::sorting_sets::sort::tests::sort_gpu_round_trip"
---

# What does the `sort` function do in MATLAB / RunMat?
`sort` orders the elements of its input along a chosen dimension. By default, it sorts ascending along the first non-singleton dimension and can also return the permutation indices that achieve the ordering.

## How does the `sort` function behave in MATLAB / RunMat?
- Scalars remain unchanged; vectors are reordered into ascending or descending order.
- For matrices and higher-dimensional tensors, the sort operates along a specified dimension (default: first non-singleton). All other dimensions remain untouched.
- `[B, I] = sort(A, ...)` returns both the sorted values `B` and the permutation indices `I`, using MATLAB's one-based indexing.
- Direction arguments accept `'ascend'` (default) or `'descend'`.
- Name-value pairs support `'ComparisonMethod'` with values `'auto'`, `'real'`, or `'abs'`. The `'abs'` option sorts by absolute value while breaking ties using the signed value.
- For complex inputs, the default `'ComparisonMethod'='auto'` behaves like `'abs'` (magnitude ordering) and `'real'` compares the real component first with the imaginary component used for tie-breaking.
- NaN values are treated as missing: they appear at the end for ascending sorts and at the beginning for descending sorts (matching MATLAB's `'MissingPlacement','auto'` behaviour for doubles).
- Dimensions greater than `ndims(A)` are treated as singleton dimensions (size 1) and therefore leave `A` unchanged while returning index values of `1`.

## GPU execution in RunMat
- `sort` is registered as a sink builtin. When tensors reside on a GPU without a specialised sort kernel, RunMat gathers them to host memory, performs the sort, and returns host-resident outputs.
- Providers may implement a future `sort_dim` hook to keep data on the GPU. Until then, all providers fall back to the host path automatically.
- The returned index tensor is always host-resident double precision.

## Examples of using `sort` in MATLAB / RunMat

### Sorting a column vector in ascending order
```matlab
A = [3; 1; 2];
B = sort(A);
```
Expected output:
```matlab
B =
     1
     2
     3
```

### Sorting rows by specifying the dimension
```matlab
A = [1 4 2; 3 2 5];
B = sort(A, 2);
```
Expected output:
```matlab
B =
     1     2     4
     2     3     5
```

### Sorting values in descending order
```matlab
A = [10 4 7 9];
B = sort(A, 'descend');
```
Expected output:
```matlab
B =
    10     9     7     4
```

### Retrieving permutation indices alongside the sorted values
```matlab
A = [4 1 9 2];
[B, I] = sort(A);
```
Expected output:
```matlab
B =
     1     2     4     9
I =
     2     4     1     3
```

### Sorting by absolute value using `ComparisonMethod`
```matlab
A = [-8 -1 3 -2];
B = sort(A, 'ComparisonMethod', 'abs');
```
Expected output:
```matlab
B =
    -1    -2     3    -8
```

### Sorting tensors containing NaN values
```matlab
A = [NaN 4 1 2];
[B, I] = sort(A);
```
Expected output:
```matlab
B =
     1     2     4   NaN
I =
     3     4     2     1
```

### Sorting GPU tensors with automatic host fallback
```matlab
G = gpuArray(randn(5, 1));
[B, I] = sort(G, 'descend');
```
The runtime gathers `G`, performs the sort on the host, and returns host-resident results. The ordering matches MATLAB's semantics exactly.

## FAQ

### Can `sort` return both values and indices?
Yes. Use `[B, I] = sort(A, ...)` to receive the permutation indices alongside the sorted values.

### How are NaN values handled?
NaN values are considered missing. They appear last for ascending sorts and first for descending sorts, matching MATLAB's `'MissingPlacement','auto'` default for doubles.

### What happens when I sort along a dimension that does not exist?
`sort(A, dim)` treats dimensions beyond `ndims(A)` as singleton dimensions (size 1). The data remains unchanged and the index output is filled with ones.

### Does `sort` support name-value arguments?
Yes. `'ComparisonMethod'` accepts `'auto'`, `'real'`, or `'abs'`. Other name-value pairs such as `'MissingPlacement'` are currently not supported and raise an error.

### Are GPU tensors sorted in-place?
Not yet. `sort` is a sink builtin that gathers GPU tensors to host memory when no GPU sort kernel is available. Providers can implement specialised hooks in the future.

### Does `sort` preserve the shape of the input?
Yes. The output is the same size as the input tensor. Only values along the selected dimension are reordered.

### What numeric type do the index outputs use?
Permutation indices are returned as double-precision tensors (or scalars) using MATLAB's one-based indexing.

### Is the sorting stable?
Yes. Equal elements (including ties when sorting by absolute value) preserve their original order.

### How does `ComparisonMethod` behave with real inputs?
`'auto'` and `'real'` behave identically. `'abs'` sorts by absolute value and uses the signed value to break ties so that results match MATLAB.

### How does `ComparisonMethod` behave with complex inputs?
`'auto'` (the default) and `'abs'` order values by magnitude. `'real'` compares the real component first and falls back to the imaginary component to break ties while preserving stability.

### Do logical or scalar inputs work?
Yes. Logical inputs are promoted to double precision automatically. Scalars are returned unchanged with an index output of `1` when requested.

## See also
[sortrows](./sortrows), [unique](./unique), [max](./max), [min](./min), [permute](./permute)

## Source & Feedback
- Source code: [`crates/runmat-runtime/src/builtins/array/sorting_sets/sort.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/sort.rs)
- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::sorting_sets::sort")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sort",
    op_kind: GpuOpKind::Custom("sort"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("sort_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Providers may add a dedicated sort kernel in the future; today tensors are gathered to host memory before sorting.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::sorting_sets::sort")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sort",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Sorting breaks fusion chains and acts as a residency sink; upstream tensors are gathered to host memory.",
};

fn sort_error(message: impl Into<String>) -> crate::RuntimeControlFlow {
    build_runtime_error(message)
        .with_builtin("sort")
        .build()
        .into()
}

#[runtime_builtin(
    name = "sort",
    category = "array/sorting_sets",
    summary = "Sort scalars, vectors, matrices, or N-D tensors along a dimension, with optional index outputs.",
    keywords = "sort,ascending,descending,indices,comparisonmethod,gpu",
    accel = "sink",
    sink = true,
    builtin_path = "crate::builtins::array::sorting_sets::sort"
)]
fn sort_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    Ok(evaluate(value, &rest)?.into_sorted_value())
}

/// Evaluate the `sort` builtin once and expose both outputs.
pub fn evaluate(value: Value, rest: &[Value]) -> crate::BuiltinResult<SortEvaluation> {
    let args = SortArgs::parse(rest)?;
    match value {
        Value::GpuTensor(handle) => sort_gpu(handle, &args),
        other => sort_host(other, &args),
    }
}

fn sort_gpu(handle: GpuTensorHandle, args: &SortArgs) -> crate::BuiltinResult<SortEvaluation> {
    let shape = handle.shape.clone();
    let dim = args.dimension.unwrap_or_else(|| default_dimension(&shape));
    if dim == 0 {
        return Err(sort_error("sort: dimension must be >= 1"));
    }
    let dim_len = dimension_length(&shape, dim);
    if dim_len > 1 {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let order = args.direction.to_provider();
            let comparison = args.comparison.to_provider();
            let zero_based = dim - 1;
            if let Ok(result) = provider.sort_dim(&handle, zero_based, order, comparison) {
                let sorted_tensor = Tensor::new(result.values.data, result.values.shape)
                    .map_err(|e| sort_error(format!("sort: {e}")))?;
                let sorted_value = tensor::tensor_into_value(sorted_tensor);
                let indices_tensor = Tensor::new(result.indices.data, result.indices.shape)
                    .map_err(|e| sort_error(format!("sort: {e}")))?;
                return Ok(SortEvaluation {
                    sorted: sorted_value,
                    indices: indices_tensor,
                });
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    sort_real_tensor(tensor, args)
}

fn sort_host(value: Value, args: &SortArgs) -> crate::BuiltinResult<SortEvaluation> {
    match value {
        Value::ComplexTensor(ct) => sort_complex_tensor(ct, args),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| sort_error(format!("sort: {e}")))?;
            sort_complex_tensor(tensor, args)
        }
        other => {
            let tensor = tensor::value_into_tensor_for("sort", other)
                .map_err(|e| sort_error(e))?;
            sort_real_tensor(tensor, args)
        }
    }
}

fn sort_real_tensor(tensor: Tensor, args: &SortArgs) -> crate::BuiltinResult<SortEvaluation> {
    let dim = args
        .dimension
        .unwrap_or_else(|| default_dimension(&tensor.shape));
    if dim == 0 {
        return Err(sort_error("sort: dimension must be >= 1"));
    }

    let dim_len = dimension_length(&tensor.shape, dim);
    if tensor.data.is_empty() || dim_len <= 1 {
        let indices = vec![1.0; tensor.data.len()];
        let index_tensor = Tensor::new(indices, tensor.shape.clone())
            .map_err(|e| sort_error(format!("sort: {e}")))?;
        let sorted_value = tensor::tensor_into_value(tensor);
        return Ok(SortEvaluation {
            sorted: sorted_value,
            indices: index_tensor,
        });
    }

    let stride_before = stride_before(&tensor.shape, dim);
    let stride_after = stride_after(&tensor.shape, dim);
    let mut sorted = tensor.data.clone();
    let mut indices = vec![0.0f64; tensor.data.len()];
    let mut buffer: Vec<(usize, f64)> = Vec::with_capacity(dim_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            buffer.clear();
            for k in 0..dim_len {
                let idx = before + k * stride_before + after * stride_before * dim_len;
                let value = tensor.data[idx];
                buffer.push((k, value));
            }
            buffer.sort_by(|a, b| compare_real_values(a.1, b.1, args));
            for (pos, (original_index, value)) in buffer.iter().enumerate() {
                let target = before + pos * stride_before + after * stride_before * dim_len;
                sorted[target] = *value;
                indices[target] = (*original_index + 1) as f64;
            }
        }
    }

    let sorted_tensor = Tensor::new(sorted, tensor.shape.clone())
        .map_err(|e| sort_error(format!("sort: {e}")))?;
    let index_tensor = Tensor::new(indices, tensor.shape.clone())
        .map_err(|e| sort_error(format!("sort: {e}")))?;

    Ok(SortEvaluation {
        sorted: tensor::tensor_into_value(sorted_tensor),
        indices: index_tensor,
    })
}

fn sort_complex_tensor(tensor: ComplexTensor, args: &SortArgs) -> crate::BuiltinResult<SortEvaluation> {
    let dim = args
        .dimension
        .unwrap_or_else(|| default_dimension(&tensor.shape));
    if dim == 0 {
        return Err(sort_error("sort: dimension must be >= 1"));
    }

    let dim_len = dimension_length(&tensor.shape, dim);
    if tensor.data.is_empty() || dim_len <= 1 {
        let indices = vec![1.0; tensor.data.len()];
        let index_tensor = Tensor::new(indices, tensor.shape.clone())
            .map_err(|e| sort_error(format!("sort: {e}")))?;
        return Ok(SortEvaluation {
            sorted: complex_tensor_into_value(tensor),
            indices: index_tensor,
        });
    }

    let stride_before = stride_before(&tensor.shape, dim);
    let stride_after = stride_after(&tensor.shape, dim);
    let mut sorted = tensor.data.clone();
    let mut indices = vec![0.0f64; tensor.data.len()];
    let mut buffer: Vec<(usize, (f64, f64))> = Vec::with_capacity(dim_len);

    for after in 0..stride_after {
        for before in 0..stride_before {
            buffer.clear();
            for k in 0..dim_len {
                let idx = before + k * stride_before + after * stride_before * dim_len;
                let value = tensor.data[idx];
                buffer.push((k, value));
            }
            buffer.sort_by(|a, b| compare_complex_values(a.1, b.1, args));
            for (pos, (original_index, value)) in buffer.iter().enumerate() {
                let target = before + pos * stride_before + after * stride_before * dim_len;
                sorted[target] = *value;
                indices[target] = (*original_index + 1) as f64;
            }
        }
    }

    let sorted_tensor = ComplexTensor::new(sorted, tensor.shape.clone())
        .map_err(|e| sort_error(format!("sort: {e}")))?;
    let index_tensor = Tensor::new(indices, tensor.shape.clone())
        .map_err(|e| sort_error(format!("sort: {e}")))?;

    Ok(SortEvaluation {
        sorted: complex_tensor_into_value(sorted_tensor),
        indices: index_tensor,
    })
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn compare_real_values(a: f64, b: f64, args: &SortArgs) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => match args.direction {
            SortDirection::Ascend => Ordering::Greater,
            SortDirection::Descend => Ordering::Less,
        },
        (false, true) => match args.direction {
            SortDirection::Ascend => Ordering::Less,
            SortDirection::Descend => Ordering::Greater,
        },
        (false, false) => compare_real_finite(a, b, args),
    }
}

fn compare_real_finite(a: f64, b: f64, args: &SortArgs) -> Ordering {
    let primary = match args.comparison {
        ComparisonMethod::Abs => {
            let abs_cmp = a.abs().partial_cmp(&b.abs()).unwrap_or(Ordering::Equal);
            if abs_cmp != Ordering::Equal {
                return match args.direction {
                    SortDirection::Ascend => abs_cmp,
                    SortDirection::Descend => abs_cmp.reverse(),
                };
            }
            Ordering::Equal
        }
        ComparisonMethod::Auto | ComparisonMethod::Real => Ordering::Equal,
    };
    if primary != Ordering::Equal {
        return primary;
    }
    match args.direction {
        SortDirection::Ascend => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
        SortDirection::Descend => b.partial_cmp(&a).unwrap_or(Ordering::Equal),
    }
}

fn compare_complex_values(a: (f64, f64), b: (f64, f64), args: &SortArgs) -> Ordering {
    match (complex_is_nan(a), complex_is_nan(b)) {
        (true, true) => Ordering::Equal,
        (true, false) => match args.direction {
            SortDirection::Ascend => Ordering::Greater,
            SortDirection::Descend => Ordering::Less,
        },
        (false, true) => match args.direction {
            SortDirection::Ascend => Ordering::Less,
            SortDirection::Descend => Ordering::Greater,
        },
        (false, false) => compare_complex_finite(a, b, args),
    }
}

fn compare_complex_finite(a: (f64, f64), b: (f64, f64), args: &SortArgs) -> Ordering {
    match args.comparison {
        ComparisonMethod::Real => compare_complex_real_imag(a, b, args.direction),
        ComparisonMethod::Abs | ComparisonMethod::Auto => {
            let abs_cmp = complex_abs(a)
                .partial_cmp(&complex_abs(b))
                .unwrap_or(Ordering::Equal);
            if abs_cmp != Ordering::Equal {
                return match args.direction {
                    SortDirection::Ascend => abs_cmp,
                    SortDirection::Descend => abs_cmp.reverse(),
                };
            }
            compare_complex_real_imag(a, b, args.direction)
        }
    }
}

fn compare_complex_real_imag(a: (f64, f64), b: (f64, f64), direction: SortDirection) -> Ordering {
    let real_cmp = match direction {
        SortDirection::Ascend => a.0.partial_cmp(&b.0),
        SortDirection::Descend => b.0.partial_cmp(&a.0),
    }
    .unwrap_or(Ordering::Equal);
    if real_cmp != Ordering::Equal {
        return real_cmp;
    }
    match direction {
        SortDirection::Ascend => a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal),
        SortDirection::Descend => b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal),
    }
}

fn complex_is_nan(value: (f64, f64)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn complex_abs(value: (f64, f64)) -> f64 {
    value.0.hypot(value.1)
}

fn stride_before(shape: &[usize], dim: usize) -> usize {
    if dim <= 1 {
        return 1;
    }
    let mut product = 1usize;
    for i in 0..(dim - 1) {
        product = product.saturating_mul(*shape.get(i).unwrap_or(&1));
    }
    product
}

fn stride_after(shape: &[usize], dim: usize) -> usize {
    if dim >= shape.len() {
        return 1;
    }
    let mut product = 1usize;
    for extent in shape.iter().skip(dim) {
        product = product.saturating_mul(*extent);
    }
    product
}

fn dimension_length(shape: &[usize], dim: usize) -> usize {
    shape.get(dim - 1).copied().unwrap_or(1)
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&extent| extent > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SortDirection {
    #[default]
    Ascend,
    Descend,
}

impl SortDirection {
    fn to_provider(self) -> ProviderSortOrder {
        match self {
            SortDirection::Ascend => ProviderSortOrder::Ascend,
            SortDirection::Descend => ProviderSortOrder::Descend,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ComparisonMethod {
    #[default]
    Auto,
    Real,
    Abs,
}

impl ComparisonMethod {
    fn to_provider(self) -> ProviderSortComparison {
        match self {
            ComparisonMethod::Auto => ProviderSortComparison::Auto,
            ComparisonMethod::Real => ProviderSortComparison::Real,
            ComparisonMethod::Abs => ProviderSortComparison::Abs,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SortArgs {
    dimension: Option<usize>,
    direction: SortDirection,
    comparison: ComparisonMethod,
}

impl SortArgs {
    fn parse(rest: &[Value]) -> crate::BuiltinResult<Self> {
        let mut args = SortArgs::default();
        let mut i = 0usize;
        while i < rest.len() {
            if args.dimension.is_none() {
                if is_dimension_placeholder(&rest[i]) {
                    i += 1;
                    continue;
                }
                match tensor::parse_dimension(&rest[i], "sort") {
                    Ok(dim) => {
                        args.dimension = Some(dim);
                        i += 1;
                        continue;
                    }
                    Err(err) => {
                        if matches!(rest[i], Value::Int(_) | Value::Num(_)) {
                            return Err(sort_error(err));
                        }
                    }
                }
            }
            if let Some(keyword) = tensor::value_to_string(&rest[i]) {
                let lowered = keyword.trim().to_ascii_lowercase();
                match lowered.as_str() {
                    "ascend" | "ascending" => {
                        args.direction = SortDirection::Ascend;
                        i += 1;
                        continue;
                    }
                    "descend" | "descending" => {
                        args.direction = SortDirection::Descend;
                        i += 1;
                        continue;
                    }
                    "comparisonmethod" => {
                        i += 1;
                        if i >= rest.len() {
                            return Err(sort_error("sort: expected a value for 'ComparisonMethod'"));
                        }
                        let raw = &rest[i];
                        let value = match raw {
                            Value::String(s) => s.clone(),
                            Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
                            Value::CharArray(ca) if ca.rows == 1 => {
                                ca.data.iter().copied().collect()
                            }
                            _ => {
                                return Err(
                                    sort_error("sort: 'ComparisonMethod' requires a string value")
                                )
                            }
                        };
                        let lowered_value = value.trim().to_ascii_lowercase();
                        args.comparison = match lowered_value.as_str() {
                            "auto" => ComparisonMethod::Auto,
                            "real" => ComparisonMethod::Real,
                            "abs" | "magnitude" => ComparisonMethod::Abs,
                            other => {
                                return Err(sort_error(format!(
                                    "sort: unsupported ComparisonMethod '{other}'"
                                ))
                                .into())
                            }
                        };
                        i += 1;
                        continue;
                    }
                    "missingplacement" => {
                        return Err(sort_error(
                            "sort: the 'MissingPlacement' option is not supported yet",
                        )
                        .into());
                    }
                    _ => {}
                }
            }
            return Err(sort_error(format!("sort: unrecognised argument {:?}", rest[i])));
        }
        Ok(args)
    }
}

fn is_dimension_placeholder(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.data.is_empty(),
        Value::LogicalArray(logical) => logical.data.is_empty(),
        _ => false,
    }
}

pub struct SortEvaluation {
    sorted: Value,
    indices: Tensor,
}

impl SortEvaluation {
    pub fn into_sorted_value(self) -> Value {
        self.sorted
    }

    pub fn into_values(self) -> (Value, Value) {
        let indices = tensor::tensor_into_value(self.indices);
        (self.sorted, indices)
    }

    pub fn indices_value(&self) -> Value {
        tensor::tensor_into_value(self.indices.clone())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ComplexTensor, IntValue, Tensor, Value};

    fn error_message(flow: crate::RuntimeControlFlow) -> String {
        match flow {
            crate::RuntimeControlFlow::Error(err) => err.message().to_string(),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_vector_default() {
        let tensor = Tensor::new(vec![3.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let result = sort_builtin(Value::Tensor(tensor), Vec::new()).expect("sort");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
                assert_eq!(t.shape, vec![3, 1]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_descend_direction() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result =
            sort_builtin(Value::Tensor(tensor), vec![Value::from("descend")]).expect("sort");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![4.0, 3.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_matrix_default_dim1() {
        let tensor = Tensor::new(vec![4.0, 2.0, 1.0, 5.0, 6.0, 3.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![2.0, 4.0, 1.0, 5.0, 3.0, 6.0]);
                assert_eq!(t.shape, vec![2, 3]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 1.0, 1.0, 2.0, 2.0, 1.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_matrix_along_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 2.0, 2.0, 5.0], vec![2, 3]).unwrap();
        let eval =
            evaluate(Value::Tensor(tensor), &[Value::Int(IntValue::I32(2))]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0]);
                assert_eq!(t.shape, vec![2, 3]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_dimension_placeholder_then_dim() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 2.0], vec![2, 2]).unwrap();
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[
                Value::Tensor(placeholder),
                Value::Int(IntValue::I32(2)),
                Value::from("descend"),
            ],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.data, vec![4.0, 3.0, 1.0, 2.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_descend_then_dimension() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 2.0, 2.0, 5.0], vec![2, 3]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[Value::from("descend"), Value::Int(IntValue::I32(1))],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 1.0, 4.0, 2.0, 5.0, 2.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_returns_indices() {
        let tensor = Tensor::new(vec![4.0, 1.0, 9.0, 2.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 4.0, 9.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 4.0, 1.0, 3.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_with_nan_handling() {
        let tensor = Tensor::new(vec![f64::NAN, 4.0, 1.0, 2.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert!(t.data[3].is_nan());
                assert_eq!(&t.data[0..3], &[1.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let eval_desc =
            evaluate(Value::Tensor(tensor), &[Value::from("descend")]).expect("evaluate");
        let (sorted_desc, _) = eval_desc.into_values();
        match sorted_desc {
            Value::Tensor(t) => {
                assert!(t.data[0].is_nan());
                assert_eq!(&t.data[1..], &[4.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_by_absolute_value() {
        let tensor = Tensor::new(vec![-8.0, -1.0, 3.0, -2.0], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[Value::from("ComparisonMethod"), Value::from("abs")],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.data, vec![-1.0, -2.0, 3.0, -8.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_by_absolute_value_descend() {
        let tensor = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4, 1]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor),
            &[
                Value::from("descend"),
                Value::from("ComparisonMethod"),
                Value::from("abs"),
            ],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.data, vec![4.0, -3.0, 2.0, -1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_complex_auto_abs() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.5), (0.0, -1.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::ComplexTensor(t) => {
                assert_eq!(t.data, vec![(0.0, -1.0), (1.0, 2.0), (-3.0, 0.5)])
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 1.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_complex_real_descend() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.0), (1.0, -1.0)], vec![3, 1]).unwrap();
        let eval = evaluate(
            Value::ComplexTensor(tensor),
            &[
                Value::from("descend"),
                Value::from("ComparisonMethod"),
                Value::from("real"),
            ],
        )
        .expect("evaluate");
        let (sorted, _) = eval.into_values();
        match sorted {
            Value::ComplexTensor(t) => {
                assert_eq!(t.data, vec![(1.0, 2.0), (1.0, -1.0), (-3.0, 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_stable_with_duplicates() {
        let tensor = Tensor::new(vec![2.0, 2.0, 1.0, 2.0], vec![4, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 2.0, 2.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert_eq!(t.data, vec![3.0, 1.0, 2.0, 4.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_empty_tensor() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => {
                assert!(t.data.is_empty());
                assert_eq!(t.shape, tensor.shape);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert!(t.data.is_empty()),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_dim_greater_than_ndims() {
        let tensor = Tensor::new(vec![4.0, 2.0, 3.0, 1.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(tensor.clone()),
            &[Value::Int(IntValue::I32(3))],
        )
        .expect("evaluate");
        let (sorted, indices) = eval.into_values();
        match sorted {
            Value::Tensor(t) => assert_eq!(t.data, tensor.data),
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => assert!(t.data.iter().all(|v| (*v - 1.0).abs() < f64::EPSILON)),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_invalid_argument_errors() {
        let err = error_message(sort_builtin(
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            vec![Value::from("missingplacement"), Value::from("first")],
        )
        .unwrap_err());
        assert!(err.contains("MissingPlacement"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_invalid_comparison_method_errors() {
        let err = error_message(sort_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            vec![Value::from("ComparisonMethod"), Value::from("unknown")],
        )
        .unwrap_err());
        assert!(err.contains("ComparisonMethod"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_invalid_comparison_method_value_errors() {
        let err = error_message(sort_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            vec![
                Value::from("ComparisonMethod"),
                Value::Int(IntValue::I32(1)),
            ],
        )
        .unwrap_err());
        assert!(
            err.contains("requires a string value"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_dimension_zero_errors() {
        let err = error_message(sort_builtin(
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            vec![Value::Num(0.0)],
        )
        .unwrap_err());
        assert!(
            err.contains("dimension must be >= 1"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sort_gpu_round_trip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 1.0, 2.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("evaluate");
            let (sorted, indices) = eval.into_values();
            match sorted {
                Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 3.0]),
                other => panic!("expected tensor, got {other:?}"),
            }
            match indices {
                Value::Tensor(t) => assert_eq!(t.data, vec![2.0, 3.0, 1.0]),
                other => panic!("expected tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sort_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![4.0, 1.0, 3.0, 2.0], vec![4, 1]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cpu sort");
        let (cpu_sorted, cpu_indices) = cpu_eval.into_values();

        let gpu_view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&gpu_view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu sort");
        let (gpu_sorted, gpu_indices) = gpu_eval.into_values();

        let cpu_sorted_tensor = match cpu_sorted {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected CPU sorted value {other:?}"),
        };
        let cpu_indices_tensor = match cpu_indices {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected CPU indices value {other:?}"),
        };
        let gpu_sorted_tensor = match gpu_sorted {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected GPU sorted value {other:?}"),
        };
        let gpu_indices_tensor = match gpu_indices {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected GPU indices value {other:?}"),
        };

        assert_eq!(gpu_sorted_tensor.data, cpu_sorted_tensor.data);
        assert_eq!(gpu_indices_tensor.data, cpu_indices_tensor.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
