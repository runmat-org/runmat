//! MATLAB-compatible `range` builtin with GPU-aware semantics for RunMat.

use std::collections::HashSet;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "range",
        wasm_path = "crate::builtins::array::creation::range"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "range"
category: "array/creation"
keywords: ["range", "max", "min", "spread", "omitnan", "gpu"]
summary: "Compute the difference between the maximum and minimum values (max - min)."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses provider min/max reductions when available; falls back to the host when omitnan is requested or when multi-dimension reductions are unsupported."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::creation::range::tests"
  integration: "builtins::array::creation::range::tests::range_gpu_provider_roundtrip"
  gpu: "builtins::array::creation::range::tests::range_wgpu_dim2_matches_cpu"
---

# What does the `range` function do in MATLAB / RunMat?
`range(X)` returns the span (maximum minus minimum) of the elements in `X`. For arrays,
the calculation happens along the first non-singleton dimension unless you specify one.
The output keeps MATLAB's column-major shape conventions.

## How does the `range` function behave in MATLAB / RunMat?
- Operates on scalars, vectors, matrices, and N-D tensors.
- `range(X)` measures the spread along the first non-singleton dimension.
- `range(X, dim)` targets a specific dimension.
- `range(X, vecdim)` reduces across multiple dimensions at once.
- `range(X, 'all')` collapses all elements to a single scalar result.
- `range(___)` is equivalent to `max(___) - min(___)` over the requested slice.
- `vecdim` accepts a row or column vector of positive integers; duplicate entries are ignored.
- `range(___, 'omitnan')` ignores `NaN` values (resulting in `NaN` if a slice has no finite values).
- `range(___, 'includenan')` (default) returns `NaN` for any slice containing `NaN`.
- Works with `gpuArray` inputs; the runtime keeps residency on the GPU when provider hooks are available.

## Examples of using the `range` function in MATLAB / RunMat

### Measuring column-wise range of a matrix

```matlab
A = [1 4 2; 3 7 5];
spread = range(A);
```

Expected output:

```matlab
% Each column's maximum minus minimum
spread = [2 3 3];
```

### Selecting a dimension explicitly

```matlab
A = [1 4 2; 3 7 5];
row_spread = range(A, 2);
```

Expected output:

```matlab
row_spread = [3; 4];
```

### Collapsing all elements with 'all'

```matlab
temperatures = [68 72 75; 70 74 78];
overall = range(temperatures, 'all');
```

Expected output:

```matlab
overall = 10;
```

### Ignoring NaNs when computing range

```matlab
samples = [2 NaN 5; 4 6 NaN];
spread = range(samples, 1, 'omitnan');
```

Expected output:

```matlab
spread = [2 0 0];
```

### Keeping gpuArray results on the device

```matlab
G = gpuArray([1 10 3; 2 8 4]);
column_spread = range(G);
gather(column_spread);
```

Expected behaviour:

```matlab
ans =
     1     2     1
```

### Handling higher-dimensional inputs

```matlab
data = reshape(1:24, [3, 4, 2]);
slice_spread = range(data, [1 2]);
```

Expected output:

```matlab
slice_spread(:,:,1) =
    11

slice_spread(:,:,2) =
    11
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You generally do **not** need to call `gpuArray` explicitly in RunMat. When the input is
already on the GPU, the planner keeps the result on-device if the active provider exposes
`reduce_min`, `reduce_max`, the corresponding `*_dim` reductions, and elementwise subtraction.
Current providers specialise 2-D reductions; multi-axis reductions, tensors with more than two
dimensions, or `'omitnan'` requests gather back to the host for correctness. When a required hook
is missing, RunMat gathers the data automatically, computes on the CPU, and returns a standard tensor.

## FAQ

### What does `range(X)` return for scalars?
The range of a scalar is always `0`, because the maximum and minimum coincide.

### How are NaN values treated by default?
By default (`'includenan'`), any slice containing `NaN` yields `NaN`. Use `'omitnan'` to ignore them.

### Can I supply multiple dimensions?
Yes. Pass a vector such as `[1 3]` to reduce across both the first and third dimensions simultaneously.

### What happens if I request a dimension larger than `ndims(X)`?
RunMat mirrors MATLAB: a size-1 dimension produces zeros (or `NaN` when the lone value is `NaN`).

### Does `range` support integer and logical inputs?
Yes. The inputs are promoted to double precision internally, matching MATLAB semantics.

### How does `range` interact with `gpuArray`?
The runtime asks the active provider for min/max reductions. When unavailable, it gathers the data and computes on the CPU.

### Why is the output sometimes a scalar and sometimes a vector?
The output collapses the selected dimensions to size `1`, leaving other dimensions unchanged. A single remaining element becomes a scalar.

### How can I inspect the range across all elements quickly?
Use `range(X, 'all')` to flatten all dimensions into a single scalar spread.

## See Also
[max](../../math/reduction/max), [min](../../math/reduction/min), [sum](../../math/reduction/sum), [mean](../../math/reduction/mean), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `range` function is available at: [`crates/runmat-runtime/src/builtins/array/creation/range.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/range.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::array::creation::range")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "range",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_min_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_max_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_min",
        },
        ProviderHook::Reduction {
            name: "reduce_max",
        },
        ProviderHook::Binary {
            name: "elem_sub",
            commutative: false,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes: "Requires provider min/max reductions plus elem_sub; omitnan and multi-axis reductions gather to host when hooks are absent.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::array::creation::range")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "range",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "No fused kernel today; the runtime composes provider min/max reductions or gathers to host.",
};

#[runtime_builtin(
    name = "range",
    category = "array/creation",
    summary = "Compute the difference between the maximum and minimum values.",
    keywords = "range,max,min,spread,gpu",
    accel = "reduction",
    wasm_path = "crate::builtins::array::creation::range"
)]
fn range_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (dim_selection, nan_mode) = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => range_gpu(handle, dim_selection, nan_mode),
        other => range_host(other, dim_selection, nan_mode),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NanMode {
    Include,
    Omit,
}

#[derive(Clone, PartialEq, Eq)]
enum DimSelection {
    Auto,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Clone)]
struct ResolvedDims {
    dims_in_bounds: Vec<usize>,
    #[allow(dead_code)]
    dims_out_of_bounds: Vec<usize>,
}

fn parse_arguments(args: &[Value]) -> Result<(DimSelection, NanMode), String> {
    let mut selection = DimSelection::Auto;
    let mut nan_mode = NanMode::Include;
    let mut selection_set = false;

    for arg in args {
        if let Some(mode) = parse_nan_flag(arg)? {
            nan_mode = mode;
            continue;
        }

        if is_all_flag(arg)? {
            if selection_set && !matches!(selection, DimSelection::Auto) {
                return Err(
                    "range: 'all' cannot be combined with an explicit dimension".to_string()
                );
            }
            selection = DimSelection::All;
            selection_set = true;
            continue;
        }

        if selection_set && !matches!(selection, DimSelection::Auto) {
            return Err("range: too many dimension arguments".to_string());
        }

        selection = parse_dim_spec(arg)?;
        selection_set = true;
    }

    Ok((selection, nan_mode))
}

fn parse_nan_flag(value: &Value) -> Result<Option<NanMode>, String> {
    let text = match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    };
    let Some(text) = text else {
        return Ok(None);
    };
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "omitnan" => Ok(Some(NanMode::Omit)),
        "includenan" => Ok(Some(NanMode::Include)),
        _ => Ok(None),
    }
}

fn is_all_flag(value: &Value) -> Result<bool, String> {
    let text = match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    };
    Ok(text
        .map(|s| s.trim().eq_ignore_ascii_case("all"))
        .unwrap_or(false))
}

fn parse_dim_spec(value: &Value) -> Result<DimSelection, String> {
    match value {
        Value::Int(i) => {
            let dim = i.to_i64();
            if dim < 1 {
                return Err("range: dimension must be >= 1".to_string());
            }
            Ok(DimSelection::Dim(dim as usize))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("range: dimension must be finite".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err("range: dimension must be an integer".to_string());
            }
            if rounded < 1.0 {
                return Err("range: dimension must be >= 1".to_string());
            }
            Ok(DimSelection::Dim(rounded as usize))
        }
        Value::Tensor(t) => parse_dim_tensor(t),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)?;
            parse_dim_tensor(&tensor)
        }
        Value::GpuTensor(_) => Err(
            "range: dimension arguments must reside on the host (numeric or string)".to_string(),
        ),
        other => Err(format!(
            "range: unsupported dimension argument type {:?}",
            other
        )),
    }
}

fn parse_dim_tensor(tensor: &Tensor) -> Result<DimSelection, String> {
    if tensor.data.is_empty() {
        return Ok(DimSelection::Auto);
    }
    if !is_vector_shape(&tensor.shape) {
        return Err("range: dimension vector must be a row or column vector".to_string());
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        if !value.is_finite() {
            return Err("range: dimensions must be finite".to_string());
        }
        let rounded = value.round();
        if (rounded - value).abs() > f64::EPSILON {
            return Err("range: dimensions must contain integers".to_string());
        }
        if rounded < 1.0 {
            return Err("range: dimension indices must be >= 1".to_string());
        }
        dims.push(rounded as usize);
    }
    Ok(DimSelection::Vec(dims))
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => shape.iter().filter(|&&d| d > 1).count() <= 1,
    }
}

fn range_host(value: Value, selection: DimSelection, nan_mode: NanMode) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("range", value)?;
    let resolved = resolve_dims(&tensor.shape, &selection)?;
    let result = compute_range_tensor(&tensor, &resolved, nan_mode)?;
    Ok(tensor::tensor_into_value(result))
}

fn range_gpu(
    handle: GpuTensorHandle,
    selection: DimSelection,
    nan_mode: NanMode,
) -> Result<Value, String> {
    if matches!(nan_mode, NanMode::Omit) {
        return range_gpu_fallback(&handle, selection, nan_mode);
    }

    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return range_gpu_fallback(&handle, selection, nan_mode);
    };

    let resolved = resolve_dims(&handle.shape, &selection)?;

    if should_use_global_reduce(&handle.shape, &selection, &resolved) {
        if let Some(diff) = range_gpu_all(provider, &handle) {
            return Ok(Value::GpuTensor(diff));
        }
        return range_gpu_fallback(&handle, selection, nan_mode);
    }

    if resolved.dims_in_bounds.len() != 1 {
        return range_gpu_fallback(&handle, selection, nan_mode);
    }

    let dim = resolved.dims_in_bounds[0];
    let expected_shape = reduced_shape(&handle.shape, &[dim]);

    if expected_shape.is_empty() {
        return range_gpu_fallback(&handle, selection, nan_mode);
    }

    if let Some(diff) = range_gpu_single_dim(provider, &handle, dim, &expected_shape) {
        return Ok(Value::GpuTensor(diff));
    }

    range_gpu_fallback(&handle, selection, nan_mode)
}

fn reduced_shape(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut out = shape.to_vec();
    for &dim in dims {
        if dim < out.len() {
            out[dim] = 1;
        }
    }
    out
}

fn should_use_global_reduce(
    shape: &[usize],
    selection: &DimSelection,
    resolved: &ResolvedDims,
) -> bool {
    if shape.is_empty() {
        return false;
    }
    matches!(selection, DimSelection::All) || resolved.dims_in_bounds.len() == shape.len()
}

fn range_gpu_all(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
) -> Option<GpuTensorHandle> {
    let min_handle = provider.reduce_min(handle).ok()?;
    let max_handle = match provider.reduce_max(handle) {
        Ok(h) => h,
        Err(_) => {
            let _ = provider.free(&min_handle);
            return None;
        }
    };
    let diff = match provider.elem_sub(&max_handle, &min_handle) {
        Ok(h) => h,
        Err(_) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return None;
        }
    };
    let _ = provider.free(&min_handle);
    let _ = provider.free(&max_handle);
    Some(diff)
}

fn range_gpu_single_dim(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    dim_zero_based: usize,
    expected_shape: &[usize],
) -> Option<GpuTensorHandle> {
    let mut candidates = Vec::with_capacity(2);
    candidates.push(dim_zero_based);
    if let Some(next) = dim_zero_based.checked_add(1) {
        if next != dim_zero_based {
            candidates.push(next);
        }
    }
    let mut seen = HashSet::new();
    for candidate in candidates {
        if !seen.insert(candidate) {
            continue;
        }
        let min_result = match provider.reduce_min_dim(handle, candidate) {
            Ok(res) => res,
            Err(_) => continue,
        };
        if min_result.values.shape != expected_shape {
            let _ = provider.free(&min_result.values);
            let _ = provider.free(&min_result.indices);
            continue;
        }
        let max_result = match provider.reduce_max_dim(handle, candidate) {
            Ok(res) => res,
            Err(_) => {
                let _ = provider.free(&min_result.values);
                let _ = provider.free(&min_result.indices);
                continue;
            }
        };
        if max_result.values.shape != expected_shape {
            let _ = provider.free(&min_result.values);
            let _ = provider.free(&min_result.indices);
            let _ = provider.free(&max_result.values);
            let _ = provider.free(&max_result.indices);
            continue;
        }
        let diff = match provider.elem_sub(&max_result.values, &min_result.values) {
            Ok(handle) => handle,
            Err(_) => {
                let _ = provider.free(&min_result.values);
                let _ = provider.free(&min_result.indices);
                let _ = provider.free(&max_result.values);
                let _ = provider.free(&max_result.indices);
                continue;
            }
        };
        let _ = provider.free(&min_result.values);
        let _ = provider.free(&min_result.indices);
        let _ = provider.free(&max_result.values);
        let _ = provider.free(&max_result.indices);
        return Some(diff);
    }
    None
}

fn range_gpu_fallback(
    handle: &GpuTensorHandle,
    selection: DimSelection,
    nan_mode: NanMode,
) -> Result<Value, String> {
    let tensor = gpu_helpers::gather_tensor(handle)?;
    range_host(Value::Tensor(tensor), selection, nan_mode)
}

fn resolve_dims(shape: &[usize], selection: &DimSelection) -> Result<ResolvedDims, String> {
    let dims_1_based: Vec<usize> = match selection {
        DimSelection::Auto => vec![default_dimension_from_shape(shape)],
        DimSelection::Dim(d) => vec![*d],
        DimSelection::Vec(v) => {
            if v.is_empty() {
                vec![default_dimension_from_shape(shape)]
            } else {
                v.clone()
            }
        }
        DimSelection::All => {
            let ndims = if shape.is_empty() { 1 } else { shape.len() };
            (1..=ndims).collect()
        }
    };

    let mut seen = HashSet::new();
    let mut dims_in_bounds = Vec::new();
    let mut dims_out_of_bounds = Vec::new();
    let ndims = shape.len();

    for dim1 in dims_1_based {
        if dim1 == 0 {
            return Err("range: dimension indices must be >= 1".to_string());
        }
        if !seen.insert(dim1) {
            continue;
        }
        let zero = dim1 - 1;
        if zero < ndims {
            dims_in_bounds.push(zero);
        } else {
            dims_out_of_bounds.push(dim1);
        }
    }

    dims_in_bounds.sort_unstable();
    dims_out_of_bounds.sort_unstable();

    Ok(ResolvedDims {
        dims_in_bounds,
        dims_out_of_bounds,
    })
}

fn compute_range_tensor(
    tensor: &Tensor,
    dims: &ResolvedDims,
    nan_mode: NanMode,
) -> Result<Tensor, String> {
    let mut shape = tensor.shape.clone();
    if shape.is_empty() {
        shape = vec![tensor.rows, tensor.cols];
    }

    if dims.dims_in_bounds.is_empty() {
        let output_shape = shape.clone();
        let mut output = Vec::with_capacity(tensor.data.len());
        for &value in &tensor.data {
            if value.is_nan() {
                output.push(f64::NAN);
            } else {
                output.push(0.0);
            }
        }
        return Tensor::new(output, output_shape).map_err(|e| format!("range: {e}"));
    }

    let mut output_shape = shape.clone();
    for &dim in &dims.dims_in_bounds {
        if dim < output_shape.len() {
            output_shape[dim] = 1;
        }
    }

    let out_len = tensor::element_count(&output_shape);
    if out_len == 0 {
        return Tensor::new(vec![], output_shape).map_err(|e| format!("range: {e}"));
    }

    let mut mins = vec![f64::INFINITY; out_len];
    let mut maxs = vec![f64::NEG_INFINITY; out_len];
    let mut saw_value = vec![false; out_len];
    let mut saw_nan = vec![false; out_len];

    let mut coords = vec![0usize; shape.len()];
    let mut out_coords = vec![0usize; shape.len()];
    let mut reduce_mask = vec![false; shape.len()];
    for &dim in &dims.dims_in_bounds {
        if dim < reduce_mask.len() {
            reduce_mask[dim] = true;
        }
    }

    for (linear, &value) in tensor.data.iter().enumerate() {
        linear_to_multi(linear, &shape, &mut coords);
        for (i, coord) in coords.iter().enumerate() {
            out_coords[i] = if reduce_mask[i] { 0 } else { *coord };
        }
        let out_idx = multi_to_linear(&out_coords, &output_shape);
        if value.is_nan() {
            if matches!(nan_mode, NanMode::Include) {
                saw_nan[out_idx] = true;
            }
            continue;
        }
        if !saw_value[out_idx] {
            mins[out_idx] = value;
            maxs[out_idx] = value;
            saw_value[out_idx] = true;
        } else {
            if value < mins[out_idx] {
                mins[out_idx] = value;
            }
            if value > maxs[out_idx] {
                maxs[out_idx] = value;
            }
        }
    }

    let mut output = vec![0.0; out_len];
    for idx in 0..out_len {
        if saw_nan[idx] {
            output[idx] = f64::NAN;
        } else if saw_value[idx] {
            let diff = maxs[idx] - mins[idx];
            output[idx] = if diff == 0.0 { 0.0 } else { diff };
        } else {
            output[idx] = f64::NAN;
        }
    }

    Tensor::new(output, output_shape).map_err(|e| format!("range: {e}"))
}

fn linear_to_multi(index: usize, shape: &[usize], out: &mut [usize]) {
    let mut remainder = index;
    for (dim, &size) in shape.iter().enumerate() {
        if size == 0 {
            out[dim] = 0;
        } else {
            out[dim] = remainder % size;
            remainder /= size;
        }
    }
}

fn multi_to_linear(coords: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for (dim, &size) in shape.iter().enumerate() {
        if size == 0 {
            continue;
        }
        index += coords[dim] * stride;
        stride *= size;
    }
    index
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntValue;

    #[test]
    fn range_scalar_zero() {
        let result = range_builtin(Value::Num(42.0), Vec::new()).expect("range");
        match result {
            Value::Num(n) => assert_eq!(n, 0.0),
            other => panic!("expected scalar zero, got {other:?}"),
        }
    }

    #[test]
    fn range_vector_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0], vec![1, 3]).unwrap();
        let result = range_builtin(Value::Tensor(tensor), Vec::new()).expect("range");
        match result {
            Value::Num(n) => assert_eq!(n, 3.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn range_matrix_columnwise() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 7.0, 2.0, 5.0], vec![2, 3]).unwrap();
        let result = range_builtin(Value::Tensor(tensor), Vec::new()).expect("range");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![2.0, 3.0, 3.0]);
            }
            other => panic!("expected row vector, got {other:?}"),
        }
    }

    #[test]
    fn range_matrix_rowwise() {
        let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 7.0, 2.0, 5.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = range_builtin(Value::Tensor(tensor), args).expect("range");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![3.0, 4.0]);
            }
            other => panic!("expected column vector, got {other:?}"),
        }
    }

    #[test]
    fn range_all_collapse() {
        let tensor =
            Tensor::new((1..=6).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 3]).unwrap();
        let args = vec![Value::from("all")];
        let result = range_builtin(Value::Tensor(tensor), args).expect("range");
        match result {
            Value::Num(n) => assert_eq!(n, 5.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn range_omit_nan() {
        let tensor = Tensor::new(vec![2.0, 4.0, f64::NAN, 6.0, 5.0, f64::NAN], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(1)), Value::from("omitnan")];
        let result = range_builtin(Value::Tensor(tensor), args).expect("range");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![2.0, 0.0, 0.0]);
            }
            other => panic!("expected row vector, got {other:?}"),
        }
    }

    #[test]
    fn range_includenan_returns_nan() {
        let tensor = Tensor::new(vec![2.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result =
            range_builtin(Value::Tensor(tensor), vec![Value::from("includenan")]).expect("range");
        match result {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[test]
    fn range_all_omitnan() {
        let tensor = Tensor::new(vec![2.0, f64::NAN, 10.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("all"), Value::from("omitnan")];
        let result = range_builtin(Value::Tensor(tensor), args).expect("range");
        match result {
            Value::Num(n) => assert_eq!(n, 8.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn range_dim_beyond_ndims_returns_zeros_or_nan() {
        let tensor = Tensor::new(vec![1.0, 4.0, f64::NAN, 7.0], vec![2, 2]).unwrap();
        let args = vec![Value::Int(IntValue::I32(3))];
        let result = range_builtin(Value::Tensor(tensor), args).expect("range");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data[0], 0.0);
                assert_eq!(t.data[1], 0.0);
                assert!(t.data[2].is_nan());
                assert_eq!(t.data[3], 0.0);
            }
            other => panic!("expected matrix result, got {other:?}"),
        }
    }

    #[test]
    fn range_multiple_dimensions() {
        let tensor = Tensor::new(
            (1..=24).map(|v| v as f64).collect::<Vec<_>>(),
            vec![3, 4, 2],
        )
        .unwrap();
        let dims = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result =
            range_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("range");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1, 2]);
                assert_eq!(t.data, vec![11.0, 11.0]);
            }
            other => panic!("expected 1x1x2 tensor, got {other:?}"),
        }
    }

    #[test]
    fn range_vecdim_column_vector() {
        let tensor = Tensor::new(
            (1..=12).map(|v| v as f64).collect::<Vec<_>>(),
            vec![3, 2, 2],
        )
        .unwrap();
        let dims = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result =
            range_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("range");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1, 1]);
                assert_eq!(t.data, vec![11.0]);
            }
            Value::Num(n) => assert_eq!(n, 11.0),
            other => panic!("expected scalar tensor or scalar value, got {other:?}"),
        }
    }

    #[test]
    fn range_invalid_dimension_non_integer() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = range_builtin(Value::Tensor(tensor), vec![Value::Num(1.5)])
            .expect_err("expected dimension error");
        assert!(err.contains("integer"));
    }

    #[test]
    fn range_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 3.0, 4.0, 7.0, 2.0, 5.0], vec![2, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = range_builtin(Value::GpuTensor(handle), Vec::new()).expect("range");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.data, vec![2.0, 3.0, 3.0]);
        });
    }

    #[test]
    fn range_gpu_row_vector_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                range_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("range");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.data, vec![6.0]);
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn range_gpu_all_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 6.0, 3.0, 10.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = range_builtin(Value::GpuTensor(handle.clone()), vec![Value::from("all")])
                .expect("range");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.data, vec![9.0]);
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn range_gpu_omit_nan_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 4.0, f64::NAN, 6.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::from("omitnan")];
            let result = range_builtin(Value::GpuTensor(handle), args).expect("range");
            match result {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![1, 2]);
                    assert_eq!(t.data, vec![2.0, 0.0]);
                }
                other => panic!("expected host tensor fallback, got {other:?}"),
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn range_wgpu_dim2_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 6.0, 3.0, 5.0], vec![2, 3]).unwrap();
        let cpu = range_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Int(IntValue::I32(2))],
        )
        .expect("cpu range");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu = range_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::Int(IntValue::I32(2))],
        )
        .expect("gpu range");
        let cpu_tensor = test_support::gather(cpu).expect("gather cpu");
        let gpu_tensor = test_support::gather(gpu).expect("gather gpu");
        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gpu_tensor.data.iter().zip(cpu_tensor.data.iter()) {
            assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
        }
        let _ = provider.free(&handle);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
