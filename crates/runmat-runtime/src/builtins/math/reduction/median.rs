//! MATLAB-compatible `median` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "median",
        builtin_path = "crate::builtins::math::reduction::median"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "median"
category: "math/reduction"
keywords: ["median", "reduction", "omitnan", "includenan", "statistics", "gpu"]
summary: "Median of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Executes on-device when providers expose median reducers; otherwise falls back to host for compatibility."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::median::tests"
  integration: "builtins::math::reduction::median::tests::median_gpu_provider_roundtrip"
---

# What does the `median` function do in MATLAB / RunMat?
`median(x)` returns the middle value of scalars, vectors, matrices, and higher-dimensional tensors.
When no dimension is supplied, the reduction runs along the first non-singleton dimension.

## How does the `median` function behave in MATLAB / RunMat?
- `median(X)` on an `m × n` matrix returns a row vector (`1 × n`) containing the column medians.
- `median(X, 2)` returns a column vector (`m × 1`) containing the row medians.
- `median(X, 'all')` reduces across every element in `X` and returns a scalar.
- `median(X, vecdim)` accepts a row or column vector of dimensions (for example `[1 3]`) and reduces each listed axis in a single call while preserving the others.
- Even-length slices return the average of the two center elements after sorting.
- Logical inputs are promoted to double precision (`true → 1.0`, `false → 0.0`) before computing the median.
- `median(..., 'omitnan')` ignores `NaN` values; if every element is `NaN`, the result is `NaN`.
- `median(..., 'includenan')` (default) propagates `NaN` when any value in the slice is `NaN`.
- Empty slices return `NaN` values that preserve MATLAB-compatible shape semantics.
- Dimensions larger than `ndims(X)` leave the input unchanged.

## `median` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that already live on the GPU remain on the device.
The runtime asks providers for a dedicated median reduction. If the provider cannot supply one
or cannot honour `omitnan`, RunMat gathers GPU data back to the host and executes the CPU path to
preserve MATLAB semantics (including even-length averaging and ordering ties).

## Examples of using the `median` function in MATLAB / RunMat

### Finding the median of an odd-length vector

```matlab
x = [7 2 9 4 5];
m = median(x);
```

Expected output:

```matlab
m = 5;
```

### Computing the median of an even-length vector with averaging

```matlab
x = [1 4 9 10];
m = median(x);
```

Expected output:

```matlab
m = 6.5;
```

### Column-wise median of a matrix

```matlab
A = [1 3 5; 7 9 11; 2 4 6];
colMedians = median(A);
```

Expected output:

```matlab
colMedians = [2 4 6];
```

### Row medians while ignoring NaN values

```matlab
A = [1 NaN 3; 4 5 NaN];
rowMedians = median(A, 2, 'omitnan');
```

Expected output:

```matlab
rowMedians = [2; 4.5];
```

### Median of all elements in a matrix

```matlab
A = reshape(1:6, [3 2]);
overall = median(A, 'all');
```

Expected output:

```matlab
overall = 3.5;
```

### Median of all elements on a GPU-backed tensor

```matlab
G = gpuArray(randn(2048, 2048));
overall = median(G, 'omitnan');
result = gather(overall);
```

The planner keeps the tensor on the device for upstream work, gathers for the median, and returns the scalar result.

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, in the above example,
the result of the `median` call will already be on the GPU when the fusion planner has detected a net benefit to
operating the fused expression it is part of on the GPU.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency,
you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the
core language, as their toolbox is a separate commercial product, MathWorks MATLAB users need to call `gpuArray`
to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.

## FAQ

### When should I use the `median` function?
Use `median` when you need a robust measure of central tendency that resists outliers better than the mean.

### How are NaN values handled?
Use `'omitnan'` to ignore `NaN` entries. The default `'includenan'` propagates `NaN` if any element in the slice is `NaN`.

### Does `median` convert logical values?
Yes. Logical arrays are converted to double precision before the median is taken, matching MATLAB.

### What happens with even-length slices?
Even-length slices return the arithmetic mean of the two middle values after sorting, matching MATLAB behaviour.

### Can I compute the median along a specific dimension?
Yes. Pass a dimension index (`>= 1`) as the second argument: `median(A, 2)` reduces across rows.

### What if I ask for a dimension larger than `ndims(A)`?
RunMat treats trailing dimensions as having size 1, so the result is the original array unchanged.

### Does `median` fully execute on the GPU today?
When the active provider implements the `reduce_median*` hooks (e.g., the WGPU backend), the reduction stays on the device.
Otherwise the runtime gathers to the CPU to guarantee MATLAB-compatible semantics.

## See Also
[mean](./mean), [sum](./sum), [prod](./prod), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `median` function is available at: [`crates/runmat-runtime/src/builtins/math/reduction/median.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/median.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::median")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "median",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_median_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_median",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute medians entirely on device; runtimes fall back to host when hooks are missing or omitnan is requested.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::median")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "median",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes:
        "Fusion planner gathers to the host; future kernels may expose order-statistic reductions.",
};

#[derive(Clone)]
enum MedianAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

struct ParsedArguments {
    axes: MedianAxes,
    nan_mode: ReductionNaN,
}

#[runtime_builtin(
    name = "median",
    category = "math/reduction",
    summary = "Median of scalars, vectors, matrices, or N-D tensors.",
    keywords = "median,reduction,omitnan,includenan,statistics,gpu",
    accel = "reduction",
    builtin_path = "crate::builtins::math::reduction::median"
)]
fn median_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => (median_gpu(handle, &parsed)).map_err(Into::into),
        other => (median_host(other, &parsed)).map_err(Into::into),
    }
}

fn parse_arguments(args: &[Value]) -> Result<ParsedArguments, String> {
    let mut axes = MedianAxes::Default;
    let mut axes_set = false;
    let mut nan_mode = ReductionNaN::Include;

    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];

        if let Some(keyword) = keyword_of(arg) {
            match keyword.as_str() {
                "omitnan" => {
                    nan_mode = ReductionNaN::Omit;
                    idx += 1;
                    continue;
                }
                "includenan" => {
                    nan_mode = ReductionNaN::Include;
                    idx += 1;
                    continue;
                }
                "all" => {
                    if axes_set && !matches!(axes, MedianAxes::Default) {
                        return Err(
                            "median: 'all' cannot be combined with an explicit dimension"
                                .to_string(),
                        );
                    }
                    axes = MedianAxes::All;
                    axes_set = true;
                    idx += 1;
                    continue;
                }
                "" => {
                    return Err("median: keyword arguments must not be empty strings".to_string());
                }
                _ => {
                    if let Some(original) = value_as_str(arg) {
                        return Err(format!("median: unrecognised argument '{original}'"));
                    } else {
                        return Err(format!("median: unrecognised argument {arg:?}"));
                    }
                }
            }
        }

        if !axes_set || matches!(axes, MedianAxes::Default) {
            if let Some(selection) = parse_axes(arg)? {
                if matches!(selection, MedianAxes::All) {
                    if axes_set && !matches!(axes, MedianAxes::Default) {
                        return Err(
                            "median: 'all' cannot be combined with an explicit dimension"
                                .to_string(),
                        );
                    }
                    axes = MedianAxes::All;
                } else {
                    axes = selection;
                }
                axes_set = true;
                idx += 1;
                continue;
            }
        } else if parse_axes(arg)?.is_some() {
            return Err("median: multiple dimension specifications provided".to_string());
        }

        return Err(format!("median: unrecognised argument {arg:?}"));
    }

    Ok(ParsedArguments { axes, nan_mode })
}

fn median_host(value: Value, args: &ParsedArguments) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("median", value)?;
    let reduced = median_tensor(tensor, args.axes.clone(), args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn median_gpu(handle: GpuTensorHandle, args: &ParsedArguments) -> Result<Value, String> {
    if args.nan_mode == ReductionNaN::Include {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Some(device_result) = median_gpu_try(provider, &handle, &args.axes) {
                return Ok(Value::GpuTensor(device_result));
            }
        }
    }

    let gathered = gpu_helpers::gather_tensor(&handle)?;
    let reduced = median_tensor(gathered, args.axes.clone(), args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn median_gpu_try(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    axes: &MedianAxes,
) -> Option<GpuTensorHandle> {
    match axes {
        MedianAxes::Default => {
            if handle.shape.is_empty() {
                Some(handle.clone())
            } else {
                let dim = default_dimension_from_shape(&handle.shape);
                reduce_median_dim_gpu(provider, handle.clone(), dim)
            }
        }
        MedianAxes::Dim(dim) => reduce_median_dim_gpu(provider, handle.clone(), *dim),
        MedianAxes::Vec(dims) => {
            let mut result = handle.clone();
            let mut dims_sorted = dims.clone();
            dims_sorted.sort_unstable();
            dims_sorted.dedup();
            for dim in dims_sorted {
                result = reduce_median_dim_gpu(provider, result, dim)?;
            }
            Some(result)
        }
        MedianAxes::All => {
            if handle.shape.is_empty() {
                Some(handle.clone())
            } else {
                provider
                    .reduce_median(handle)
                    .map_err(|err| {
                        log::trace!("median: provider reduce_median fallback triggered: {err}");
                        err
                    })
                    .ok()
            }
        }
    }
}

fn reduce_median_dim_gpu(
    provider: &dyn AccelProvider,
    handle: GpuTensorHandle,
    dim: usize,
) -> Option<GpuTensorHandle> {
    if dim == 0 {
        return None;
    }
    if handle.shape.len() < dim {
        return Some(handle);
    }
    provider
        .reduce_median_dim(&handle, dim - 1)
        .map_err(|err| {
            log::trace!("median: provider reduce_median_dim fallback triggered: {err}");
            err
        })
        .ok()
}

fn median_tensor(
    tensor: Tensor,
    axes: MedianAxes,
    nan_mode: ReductionNaN,
) -> Result<Tensor, String> {
    match axes {
        MedianAxes::Default => {
            let dim = default_dimension(&tensor);
            reduce_tensor_median_dim(&tensor, dim, nan_mode)
        }
        MedianAxes::Dim(dim) => reduce_tensor_median_dim(&tensor, dim, nan_mode),
        MedianAxes::Vec(mut dims) => {
            let mut current = tensor;
            dims.sort_unstable();
            dims.dedup();
            if dims.is_empty() {
                let dim = default_dimension(&current);
                current = reduce_tensor_median_dim(&current, dim, nan_mode)?;
                return Ok(current);
            }
            for dim in dims {
                current = reduce_tensor_median_dim(&current, dim, nan_mode)?;
            }
            Ok(current)
        }
        MedianAxes::All => {
            if tensor.shape.is_empty() {
                Ok(tensor)
            } else {
                let mut current = tensor;
                let rank = current.shape.len();
                for dim in 1..=rank {
                    current = reduce_tensor_median_dim(&current, dim, nan_mode)?;
                }
                Ok(current)
            }
        }
    }
}

fn parse_axes(value: &Value) -> Result<Option<MedianAxes>, String> {
    if let Some(text) = value_as_str(value) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err("median: dimension string must not be empty".to_string());
        }
        let lowered = trimmed.to_ascii_lowercase();
        return match lowered.as_str() {
            "all" => Ok(Some(MedianAxes::All)),
            "omitnan" | "includenan" => Ok(None),
            _ => Err(format!("median: unrecognised argument '{trimmed}'")),
        };
    }

    match value {
        Value::Tensor(t) => {
            if t.data.is_empty() {
                return Ok(Some(MedianAxes::Default));
            }
            if t.data.len() == 1 {
                let scalar = Value::Num(t.data[0]);
                let dim = tensor::parse_dimension(&scalar, "median")?;
                return Ok(Some(MedianAxes::Dim(dim)));
            }
            let dims = parse_dimension_vector(t)?;
            if dims.is_empty() {
                Ok(Some(MedianAxes::Default))
            } else {
                Ok(Some(MedianAxes::Vec(dims)))
            }
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)?;
            if tensor.data.is_empty() {
                return Ok(Some(MedianAxes::Default));
            }
            if tensor.data.len() == 1 {
                let scalar = Value::Num(tensor.data[0]);
                let dim = tensor::parse_dimension(&scalar, "median")?;
                return Ok(Some(MedianAxes::Dim(dim)));
            }
            let dims = parse_dimension_vector(&tensor)?;
            if dims.is_empty() {
                Ok(Some(MedianAxes::Default))
            } else {
                Ok(Some(MedianAxes::Vec(dims)))
            }
        }
        Value::Int(_) | Value::Num(_) => {
            let dim = tensor::parse_dimension(value, "median")?;
            Ok(Some(MedianAxes::Dim(dim)))
        }
        Value::GpuTensor(_) => Err("median: dimension arguments cannot be GPU tensors".to_string()),
        Value::Bool(_) => Err("median: dimension must be numeric".to_string()),
        _ => Ok(None),
    }
}

fn parse_dimension_vector(tensor: &Tensor) -> Result<Vec<usize>, String> {
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &entry in &tensor.data {
        if !entry.is_finite() {
            return Err("median: dimension entries must be finite integers".to_string());
        }
        let rounded = entry.round();
        if (rounded - entry).abs() > f64::EPSILON {
            return Err("median: dimension entries must be integers".to_string());
        }
        if rounded < 1.0 {
            return Err("median: dimension entries must be >= 1".to_string());
        }
        dims.push(rounded as usize);
    }
    Ok(dims)
}

fn value_as_str(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn reduce_tensor_median_dim(
    tensor: &Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
) -> Result<Tensor, String> {
    if dim == 0 {
        return Err("median: dimension must be >= 1".to_string());
    }

    if tensor.shape.is_empty() {
        let value = tensor.data.first().copied().unwrap_or(f64::NAN);
        return Tensor::new(vec![value], vec![1, 1]).map_err(|e| format!("median: {e}"));
    }

    if dim > tensor.shape.len() {
        return Ok(tensor.clone());
    }

    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let Some(output_shape) = reduction_shape(&tensor.shape, dim) else {
        return Ok(tensor.clone());
    };

    if reduce_len == 0 || tensor.data.is_empty() {
        let fill = vec![f64::NAN; tensor::element_count(&output_shape)];
        return Tensor::new(fill, output_shape).map_err(|e| format!("median: {e}"));
    }

    if reduce_len == 1 {
        return Tensor::new(tensor.data.clone(), tensor.shape.clone())
            .map_err(|e| format!("median: {e}"));
    }

    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let mut output = vec![0.0f64; tensor::element_count(&output_shape)];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut slice = Vec::with_capacity(reduce_len);
            let mut saw_nan = false;

            for k in 0..reduce_len {
                let idx = before + k * stride_before + after * stride_before * reduce_len;
                let value = tensor.data[idx];
                match nan_mode {
                    ReductionNaN::Include => {
                        if value.is_nan() {
                            saw_nan = true;
                            break;
                        }
                        slice.push(value);
                    }
                    ReductionNaN::Omit => {
                        if value.is_nan() {
                            continue;
                        }
                        slice.push(value);
                    }
                }
            }

            let out_idx = after * stride_before + before;

            if saw_nan {
                output[out_idx] = f64::NAN;
                continue;
            }

            if slice.is_empty() {
                output[out_idx] = f64::NAN;
                continue;
            }

            let median = compute_median_inplace(&mut slice);
            output[out_idx] = median;
        }
    }

    Tensor::new(output, output_shape).map_err(|e| format!("median: {e}"))
}

pub fn compute_median_inplace(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| partial_cmp_f64(*a, *b));
    let len = values.len();
    if len % 2 == 1 {
        values[len / 2]
    } else {
        let upper = values[len / 2];
        let lower = values[len / 2 - 1];
        0.5 * (lower + upper)
    }
}

fn partial_cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Less)
}

fn reduction_shape(shape: &[usize], dim: usize) -> Option<Vec<usize>> {
    if dim == 0 {
        return None;
    }
    if shape.is_empty() {
        return Some(vec![1, 1]);
    }
    if dim > shape.len() {
        return None;
    }
    let mut out = shape.to_vec();
    out[dim - 1] = 1;
    Some(out)
}

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, v| acc.saturating_mul(v))
}

fn default_dimension(tensor: &Tensor) -> usize {
    default_dimension_from_shape(&tensor.shape)
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
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::IntValue;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_scalar_num() {
        let result = median_builtin(Value::Num(5.0), Vec::new()).expect("median");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_vector_odd_length() {
        let tensor = Tensor::new(vec![7.0, 2.0, 9.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_vector_even_length() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 10.0], vec![4, 1]).unwrap();
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        assert_eq!(result, Value::Num(6.5));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 7.0, 2.0, 9.0, 5.0, 11.0], vec![3, 2]).expect("tensor");
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![2.0, 9.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0], vec![3, 2]).expect("tensor");
        let result = median_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))])
            .expect("median");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.data, vec![4.0, 6.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_all_across_matrix() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).unwrap();
        let result =
            median_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("median");
        match result {
            Value::Num(v) => assert!((v - 3.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_vecdim_multiple_axes() {
        let tensor =
            Tensor::new((1..=8).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result =
            median_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("median");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2, 1]);
                assert_eq!(out.data.len(), 2);
                assert!((out.data[0] - 3.5).abs() < 1e-12);
                assert!((out.data[1] - 5.5).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_with_omit_nan() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result =
            median_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("median");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_with_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        match result {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_empty_returns_nan() {
        let tensor = Tensor::new(vec![], vec![0, 1]).unwrap();
        let result = median_builtin(Value::Tensor(tensor), Vec::new()).expect("median");
        match result {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected NaN scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = tensor.clone();
        let result = median_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(5))])
            .expect("median");
        match result {
            Value::Tensor(out) => assert_eq!(out, original),
            Value::Num(n) => assert_eq!(n, original.data[0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_rejects_unknown_keyword() {
        let err = median_builtin(Value::Num(1.0), vec![Value::from("like")]).unwrap_err();
        assert!(
            err.contains("unrecognised argument"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = median_builtin(Value::GpuTensor(handle), Vec::new()).expect("median");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.data[0], 6.5);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn median_gpu_omit_nan_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = median_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")])
                .expect("median");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.data[0], 3.0);
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
    #[cfg(feature = "wgpu")]
    fn median_wgpu_dim_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 5.0, 9.0, 2.0, 6.0, 10.0], vec![3, 2]).unwrap();
        let args_dim1 = ParsedArguments {
            axes: MedianAxes::Dim(1),
            nan_mode: ReductionNaN::Include,
        };
        let cpu = median_host(Value::Tensor(tensor.clone()), &args_dim1).expect("cpu median");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_value = median_gpu(handle, &args_dim1).expect("gpu median");
        let gathered = test_support::gather(gpu_value).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 5e-5,
                };
                for (a, b) in ct.data.iter().zip(gt.data.iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }

        // Global median ('all') remains consistent
        let args_all = ParsedArguments {
            axes: MedianAxes::All,
            nan_mode: ReductionNaN::Include,
        };
        let cpu_all =
            median_host(Value::Tensor(tensor.clone()), &args_all).expect("cpu median all");
        let gpu_all = median_gpu(
            runmat_accelerate_api::provider()
                .unwrap()
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                })
                .expect("upload"),
            &args_all,
        )
        .expect("gpu median all");
        let gathered_all = test_support::gather(gpu_all).expect("gather");
        match cpu_all {
            Value::Num(a) => {
                assert_eq!(gathered_all.data.len(), 1);
                assert!((a - gathered_all.data[0]).abs() < 1e-12);
            }
            Value::Tensor(t) => {
                assert_eq!(t.data.len(), gathered_all.data.len());
                for (a, b) in t.data.iter().zip(gathered_all.data.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("unexpected CPU output for all: {other:?}"),
        }
    }
}
