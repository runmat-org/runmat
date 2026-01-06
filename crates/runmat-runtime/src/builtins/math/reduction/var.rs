//! MATLAB-compatible `var` builtin with GPU-aware semantics for RunMat.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, ProviderNanMode, ProviderStdNormalization,
};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{extract_dims, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "var",
        builtin_path = "crate::builtins::math::reduction::var"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "var"
category: "math/reduction"
keywords: ["var", "variance", "statistics", "gpu", "omitnan", "all"]
summary: "Variance of scalars, vectors, matrices, or N-D tensors with MATLAB-compatible options."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to host when 'omitnan' is requested or the provider lacks std/elementwise kernels for the squaring step."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::var::tests"
  integration: "builtins::math::reduction::var::tests::var_gpu_provider_roundtrip"
  gpu: "builtins::math::reduction::var::tests::var_wgpu_dim1_matches_cpu"
---

# What does the `var` function do in MATLAB / RunMat?
`var(x)` measures the spread of the elements in `x` by returning their variance. By default, RunMat matches MATLAB’s sample definition (dividing by `n-1`) and works along the first non-singleton dimension.

## How does the `var` function behave in MATLAB / RunMat?
- `var(X)` on an `m × n` matrix returns a `1 × n` row vector with the sample variance of each column.
- `var(X, 1)` switches to population normalisation (`n` in the denominator). Use `var(X, 0)` or `var(X, [])` to keep the default sample behaviour.
- `var(X, flag, dim)` lets you pick both the normalisation (`flag = 0` sample, `1` population, or `[]`) and the dimension to reduce. `var(X, flag, 'all')` collapses every dimension, while `var(X, flag, vecdim)` accepts a dimension vector such as `[1 3]` and reduces all listed axes in a single call.
- Strings like `'omitnan'` and `'includenan'` decide whether `NaN` values are skipped or propagated.
- Logical inputs are promoted to double precision before reduction so that results follow MATLAB’s numeric rules.
- Empty slices return `NaN` with MATLAB-compatible shapes. Scalars return `0`, regardless of the normalisation mode.
- Dimensions greater than `ndims(X)` leave the input unchanged.
- Weighted variances (`flag` as a vector) are not implemented yet; RunMat reports a descriptive error when they are requested.

Complex tensors are not currently supported; convert them to real magnitudes manually before calling `var`.

## `var` Function GPU Execution Behaviour
When RunMat Accelerate is active, device-resident tensors remain on the GPU whenever the provider implements the relevant hooks. Providers that expose `reduce_std_dim`/`reduce_std` execute the reduction in-place on the device; the runtime squares the resulting standard deviation tensor with an elementwise multiply in device memory. Whenever `'omitnan'`, multi-axis reductions, or unsupported shapes are requested, RunMat transparently gathers the data to the host, computes the result there, and materialises the variance tensor before returning.

## Examples of using the `var` function in MATLAB / RunMat

### Sample variance of a vector

```matlab
x = [1 2 3 4 5];
v = var(x);                 % uses flag = 0 (sample) by default
```

Expected output:

```matlab
v = 2.5;
```

### Population variance of each column

```matlab
A = [1 3 5; 2 4 6];
vpop = var(A, 1);           % divide by n instead of n-1
```

Expected output:

```matlab
vpop = [0.25 0.25 0.25];
```

### Collapsing every dimension at once

```matlab
B = reshape(1:12, [3 4]);
overall = var(B, 0, 'all');
```

Expected output:

```matlab
overall = 13;
```

### Reducing across multiple dimensions

```matlab
C = cat(3, [1 2; 3 4], [5 6; 7 8]);
sliceVariance = var(C, [], [1 3]);   % keep columns, reduce rows & pages
```

Expected output:

```matlab
sliceVariance = [6.6667 6.6667];
```

### Ignoring NaN values

```matlab
D = [1 NaN 3; 2 4 NaN];
rowVariance = var(D, 0, 2, 'omitnan');
```

Expected output:

```matlab
rowVariance = [2; 2];
```

### Variance on the GPU without manual `gpuArray`

```matlab
G = rand(1024, 512);
spread = var(G, 1, 'all');   % stays on the GPU when the provider supports std reductions
```

`spread` remains device-resident. Use `gather(spread)` if you need the result on the host.

### Preserving default behaviour with an empty normalisation flag

```matlab
C = [1 2; 3 4];
rowVariance = var(C, [], 2);
```

Expected output:

```matlab
rowVariance = [0.5; 0.5];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Usually you do not need to call `gpuArray` manually. The fusion planner keeps tensors on the GPU across fused expressions and gathers them only when necessary. For explicit control or MATLAB compatibility, you can still call `gpuArray`/`gather` yourself.

## FAQ

### What values can I pass as the normalisation flag?
Use `0` (or `[]`) for the sample definition, `1` for population. RunMat rejects non-scalar weight vectors and reports that weighted variances are not implemented yet.

### How can I collapse multiple dimensions?
Pass a vector of dimensions such as `var(A, [], [1 3])`. You can also use `'all'` to collapse every dimension into a single scalar.

### How do `'omitnan'` and `'includenan'` work?
`'omitnan'` skips NaN values; if every element in a slice is NaN the result is NaN. `'includenan'` (the default) propagates a single NaN to the output slice.

### What happens if I request a dimension greater than `ndims(X)`?
RunMat returns the input unchanged so that MATLAB-compatible code relying on that behaviour continues to work. Scalars still return `0` to follow MATLAB’s edge-case semantics.

### Are complex inputs supported?
Not yet. RunMat currently requires real inputs for `var`. Convert complex data to magnitude or separate real/imaginary parts before calling the builtin.

## See Also
[std](./std), [mean](./mean), [sum](./sum), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `var` function is available at: [`crates/runmat-runtime/src/builtins/math/reduction/var.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/var.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::var")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "var",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_std_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_std",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers compute variance via standard-deviation reductions followed by an in-device squaring pass.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::var")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "var",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion currently gathers to the host; future kernels can reuse the variance accumulator directly.",
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VarNormalization {
    Sample,
    Population,
}

#[derive(Clone, Debug)]
enum VarAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Clone)]
struct ParsedArguments {
    axes: VarAxes,
    normalization: VarNormalization,
    nan_mode: ReductionNaN,
}

enum NormParse {
    NotMatched,
    Placeholder,
    Value(VarNormalization),
    Weighted,
}

#[runtime_builtin(
    name = "var",
    category = "math/reduction",
    summary = "Variance of scalars, vectors, matrices, or N-D tensors.",
    keywords = "var,variance,statistics,gpu,omitnan,all",
    accel = "reduction",
    builtin_path = "crate::builtins::math::reduction::var"
)]
fn var_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let parsed = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => var_gpu(handle, &parsed),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("var: complex inputs are not supported yet".to_string())
        }
        other => var_host(other, &parsed),
    }
}

fn parse_arguments(args: &[Value]) -> Result<ParsedArguments, String> {
    let mut axes = VarAxes::Default;
    let mut axes_set = false;
    let mut normalization = VarNormalization::Sample;
    let mut normalization_consumed = false;
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
                    if axes_set && !matches!(axes, VarAxes::Default) {
                        return Err(
                            "var: 'all' cannot be combined with an explicit dimension".to_string()
                        );
                    }
                    axes = VarAxes::All;
                    axes_set = true;
                    idx += 1;
                    continue;
                }
                _ => {
                    return Err(format!("var: unrecognised option '{keyword}'"));
                }
            }
        }

        if !normalization_consumed {
            match parse_normalization(arg)? {
                NormParse::Value(norm) => {
                    normalization = norm;
                    normalization_consumed = true;
                    idx += 1;
                    continue;
                }
                NormParse::Placeholder => {
                    normalization_consumed = true;
                    idx += 1;
                    continue;
                }
                NormParse::Weighted => {
                    return Err("var: weighted variance is not implemented yet".to_string());
                }
                NormParse::NotMatched => {}
            }
        }

        if !axes_set || matches!(axes, VarAxes::Default) {
            if let Some(selection) = parse_axes(arg)? {
                if axes_set && !matches!(axes, VarAxes::Default) {
                    return Err("var: multiple dimension specifications provided".to_string());
                }
                if matches!(selection, VarAxes::All)
                    && axes_set
                    && !matches!(axes, VarAxes::Default)
                {
                    return Err(
                        "var: 'all' cannot be combined with an explicit dimension".to_string()
                    );
                }
                axes = selection;
                axes_set = true;
                idx += 1;
                continue;
            }
        } else if parse_axes(arg)?.is_some() {
            return Err("var: multiple dimension specifications provided".to_string());
        }

        return Err(format!("var: unrecognised argument {arg:?}"));
    }

    Ok(ParsedArguments {
        axes,
        normalization,
        nan_mode,
    })
}

fn parse_normalization(value: &Value) -> Result<NormParse, String> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.data.is_empty() {
                return Ok(NormParse::Placeholder);
            }
            if tensor.data.len() == 1 {
                return parse_normalization_scalar(tensor.data[0]);
            }
            Ok(NormParse::Weighted)
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Ok(NormParse::Placeholder);
            }
            if logical.data.len() == 1 {
                let flag = if logical.data[0] != 0 { 1.0 } else { 0.0 };
                return parse_normalization_scalar(flag);
            }
            Ok(NormParse::Weighted)
        }
        Value::Bool(flag) => Ok(NormParse::Value(if *flag {
            VarNormalization::Population
        } else {
            VarNormalization::Sample
        })),
        Value::Int(i) => match i.to_i64() {
            0 => Ok(NormParse::Value(VarNormalization::Sample)),
            1 => Ok(NormParse::Value(VarNormalization::Population)),
            _ => Err("var: normalisation flag must be 0, 1, or []".to_string()),
        },
        Value::Num(n) => parse_normalization_scalar(*n),
        Value::GpuTensor(_) => Err("var: normalisation flag must reside on the host".to_string()),
        _ => Ok(NormParse::NotMatched),
    }
}

fn parse_normalization_scalar(value: f64) -> Result<NormParse, String> {
    if !value.is_finite() {
        return Err("var: normalisation flag must be finite".to_string());
    }
    if (value - 0.0).abs() < f64::EPSILON {
        return Ok(NormParse::Value(VarNormalization::Sample));
    }
    if (value - 1.0).abs() < f64::EPSILON {
        return Ok(NormParse::Value(VarNormalization::Population));
    }
    Err("var: normalisation flag must be 0, 1, or []".to_string())
}

fn parse_axes(value: &Value) -> Result<Option<VarAxes>, String> {
    if let Some(keyword) = keyword_of(value) {
        if keyword == "all" {
            return Ok(Some(VarAxes::All));
        }
        return Ok(None);
    }

    let Some(dims) = extract_dims(value, "var")? else {
        return Ok(None);
    };
    if dims.is_empty() {
        return Err("var: dimension vector must not be empty".to_string());
    }
    let mut cleaned = Vec::with_capacity(dims.len());
    for dim in dims {
        if dim == 0 {
            return Err("var: dimensions must be >= 1".to_string());
        }
        cleaned.push(dim);
    }
    if cleaned.len() == 1 {
        Ok(Some(VarAxes::Dim(cleaned[0])))
    } else {
        Ok(Some(VarAxes::Vec(cleaned)))
    }
}

fn var_host(value: Value, args: &ParsedArguments) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("var", value)?;
    let reduced = var_tensor(tensor, &args.axes, args.normalization, args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn var_tensor(
    tensor: Tensor,
    axes: &VarAxes,
    normalization: VarNormalization,
    nan_mode: ReductionNaN,
) -> Result<Tensor, String> {
    let (dims, had_request) = resolve_axes(&tensor.shape, axes)?;
    if dims.is_empty() {
        if had_request && tensor.data.len() == 1 {
            return var_scalar_tensor(&tensor, nan_mode);
        }
        return Ok(tensor);
    }
    var_tensor_reduce(&tensor, &dims, normalization, nan_mode)
}

fn var_scalar_tensor(tensor: &Tensor, nan_mode: ReductionNaN) -> Result<Tensor, String> {
    let value = tensor.data.first().copied().unwrap_or(f64::NAN);
    let result = if value.is_nan() {
        f64::NAN
    } else {
        match nan_mode {
            ReductionNaN::Include | ReductionNaN::Omit => 0.0,
        }
    };
    Tensor::new(vec![result], vec![1, 1]).map_err(|e| format!("var: {e}"))
}

fn var_tensor_reduce(
    tensor: &Tensor,
    dims: &[usize],
    normalization: VarNormalization,
    nan_mode: ReductionNaN,
) -> Result<Tensor, String> {
    let mut dims_sorted = dims.to_vec();
    dims_sorted.sort_unstable();
    dims_sorted.dedup();
    if dims_sorted.is_empty() {
        return Ok(tensor.clone());
    }

    let output_shape = reduced_shape(&tensor.shape, &dims_sorted);
    let out_len = tensor::element_count(&output_shape);
    if tensor.data.is_empty() {
        let fill = vec![f64::NAN; out_len];
        return Tensor::new(fill, output_shape).map_err(|e| format!("var: {e}"));
    }

    let mut counts = vec![0usize; out_len];
    let mut means = vec![0.0f64; out_len];
    let mut m2 = vec![0.0f64; out_len];
    let mut saw_nan = vec![false; out_len];
    let mut coords = vec![0usize; tensor.shape.len()];
    let mut out_coords = vec![0usize; tensor.shape.len()];
    let mut reduce_mask = vec![false; tensor.shape.len()];
    for &dim in &dims_sorted {
        if dim < reduce_mask.len() {
            reduce_mask[dim] = true;
        }
    }

    for (linear, &value) in tensor.data.iter().enumerate() {
        linear_to_multi(linear, &tensor.shape, &mut coords);
        for (i, coord) in coords.iter().enumerate() {
            out_coords[i] = if reduce_mask[i] { 0 } else { *coord };
        }
        let out_idx = multi_to_linear(&out_coords, &output_shape);
        if value.is_nan() {
            if matches!(nan_mode, ReductionNaN::Include) {
                saw_nan[out_idx] = true;
            }
            continue;
        }

        let mean = &mut means[out_idx];
        let m2_slot = &mut m2[out_idx];
        counts[out_idx] += 1;
        let count = counts[out_idx];
        let delta = value - *mean;
        *mean += delta / (count as f64);
        let delta2 = value - *mean;
        *m2_slot += delta * delta2;
    }

    let mut output = vec![0.0f64; out_len];
    for idx in 0..out_len {
        output[idx] =
            if (saw_nan[idx] && matches!(nan_mode, ReductionNaN::Include)) || counts[idx] == 0 {
                f64::NAN
            } else {
                let count = counts[idx];
                match normalization {
                    VarNormalization::Sample => {
                        if count > 1 {
                            (m2[idx] / (count - 1) as f64).max(0.0)
                        } else {
                            0.0
                        }
                    }
                    VarNormalization::Population => (m2[idx] / (count as f64)).max(0.0),
                }
            };
    }

    Tensor::new(output, output_shape).map_err(|e| format!("var: {e}"))
}

fn resolve_axes(shape: &[usize], axes: &VarAxes) -> Result<(Vec<usize>, bool), String> {
    match axes {
        VarAxes::Default => {
            if shape.is_empty() {
                Ok((Vec::new(), true))
            } else {
                let dim = default_dimension_from_shape(shape);
                let zero = dim.saturating_sub(1);
                if zero < shape.len() {
                    Ok((vec![zero], true))
                } else {
                    Ok((Vec::new(), true))
                }
            }
        }
        VarAxes::Dim(dim) => {
            if *dim == 0 {
                return Err("var: dimension must be >= 1".to_string());
            }
            let zero = dim - 1;
            if zero < shape.len() {
                Ok((vec![zero], true))
            } else {
                Ok((Vec::new(), true))
            }
        }
        VarAxes::Vec(dims) => {
            if dims.is_empty() {
                return resolve_axes(shape, &VarAxes::Default);
            }
            let mut out = Vec::with_capacity(dims.len());
            for &dim in dims {
                if dim == 0 {
                    return Err("var: dimension must be >= 1".to_string());
                }
                let zero = dim - 1;
                if zero < shape.len() {
                    out.push(zero);
                }
            }
            out.sort_unstable();
            out.dedup();
            Ok((out, true))
        }
        VarAxes::All => {
            if shape.is_empty() {
                Ok((Vec::new(), true))
            } else {
                Ok(((0..shape.len()).collect(), true))
            }
        }
    }
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
    let mut idx = 0usize;
    let mut stride = 1usize;
    for (coord, &extent) in coords.iter().zip(shape.iter()) {
        if extent == 0 {
            return 0;
        }
        idx += coord * stride;
        stride *= extent;
    }
    idx
}

fn var_gpu(handle: GpuTensorHandle, args: &ParsedArguments) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(device_value) = var_gpu_reduce(provider, &handle, args) {
            return Ok(Value::GpuTensor(device_value));
        }
    }
    var_gpu_fallback(&handle, args)
}

fn var_gpu_reduce(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    args: &ParsedArguments,
) -> Option<GpuTensorHandle> {
    let (dims, had_request) = resolve_axes(&handle.shape, &args.axes).ok()?;
    let elements = tensor::element_count(&handle.shape);
    if dims.is_empty() {
        if had_request && elements == 1 {
            return None;
        }
        return Some(handle.clone());
    }

    let normalization = match args.normalization {
        VarNormalization::Sample => ProviderStdNormalization::Sample,
        VarNormalization::Population => ProviderStdNormalization::Population,
    };
    let nan_mode = match args.nan_mode {
        ReductionNaN::Include => ProviderNanMode::Include,
        ReductionNaN::Omit => ProviderNanMode::Omit,
    };

    let std_handle = if dims.len() == handle.shape.len() {
        if handle.shape.is_empty() {
            return Some(handle.clone());
        }
        provider
            .reduce_std(handle, normalization, nan_mode)
            .map_err(|err| {
                log::trace!("var: provider reduce_std fallback triggered: {err}");
                err
            })
            .ok()?
    } else if dims.len() == 1 {
        let dim = dims[0] + 1;
        reduce_std_dim_gpu(provider, handle.clone(), dim, normalization, nan_mode)?
    } else {
        return None;
    };

    provider
        .elem_mul(&std_handle, &std_handle)
        .map_err(|err| {
            log::trace!("var: provider elem_mul fallback triggered: {err}");
            err
        })
        .ok()
}

fn reduce_std_dim_gpu(
    provider: &dyn AccelProvider,
    handle: GpuTensorHandle,
    dim: usize,
    normalization: ProviderStdNormalization,
    nan_mode: ProviderNanMode,
) -> Option<GpuTensorHandle> {
    if dim == 0 {
        return None;
    }
    if handle.shape.len() < dim {
        return Some(handle);
    }
    provider
        .reduce_std_dim(&handle, dim - 1, normalization, nan_mode)
        .map_err(|err| {
            log::trace!("var: provider reduce_std_dim fallback triggered: {err}");
            err
        })
        .ok()
}

fn var_gpu_fallback(handle: &GpuTensorHandle, args: &ParsedArguments) -> Result<Value, String> {
    let tensor = gpu_helpers::gather_tensor(handle)?;
    let reduced = var_tensor(tensor, &args.axes, args.normalization, args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
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
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_scalar_num() {
        let result = var_builtin(Value::Num(6.0), Vec::new()).expect("var");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = var_builtin(Value::Tensor(tensor), Vec::new()).expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![4.5, 4.5, 4.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_matrix_population() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result =
            var_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))]).expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_all_reduces_entire_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result = var_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("var");
        match result {
            Value::Num(v) => assert!((v - 1.666_666_666_666_666_7).abs() < 1e-12),
            other => panic!("expected scalar variance, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_vector_dimension_selection() {
        let tensor = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
        )
        .unwrap();
        let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();
        let vecdim = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(placeholder), Value::Tensor(vecdim)];
        let result = var_builtin(Value::Tensor(tensor), args).expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3, 1]);
                for value in &out.data {
                    assert!((value - (37.0 / 3.0)).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_placeholder_flag_is_accepted() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let placeholder = Tensor::new(vec![], vec![0, 0]).unwrap();
        let args = vec![Value::Tensor(placeholder), Value::Int(IntValue::I32(2))];
        let result = var_builtin(Value::Tensor(tensor), args).expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                for value in &out.data {
                    assert!((*value - 0.5).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_weighted_flag_not_supported() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let weights = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let err = var_builtin(Value::Tensor(tensor), vec![Value::Tensor(weights)])
            .expect_err("var should reject weighted inputs");
        assert!(err.contains("weighted variance"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = var_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(2))],
        )
        .expect("var");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_with_omit_nan_default_dimension() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result = var_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("var");
        match result {
            Value::Num(v) => assert!((v - 8.0).abs() < 1e-12),
            other => panic!("expected scalar variance, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_with_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = var_builtin(Value::Tensor(tensor), Vec::new()).expect("var");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = var_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(5))],
        )
        .expect("var");
        match result {
            Value::Tensor(out) => assert_eq!(out, tensor),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = var_builtin(Value::GpuTensor(handle), Vec::new()).expect("var");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.data, vec![4.5, 4.5, 4.5]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var_gpu_omit_nan_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                var_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("var");
            let gathered = test_support::gather(result).expect("gather");
            let cpu = var_host(
                Value::Tensor(tensor),
                &ParsedArguments {
                    axes: VarAxes::Default,
                    normalization: VarNormalization::Sample,
                    nan_mode: ReductionNaN::Omit,
                },
            )
            .expect("cpu var");
            match cpu {
                Value::Tensor(expected) => {
                    assert_eq!(gathered.shape, expected.shape);
                    for (a, b) in gathered.data.iter().zip(expected.data.iter()) {
                        if b.is_nan() {
                            assert!(a.is_nan());
                        } else {
                            assert!((a - b).abs() < 1e-12);
                        }
                    }
                }
                _ => panic!("expected tensor output from host path"),
            }
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
    fn var_wgpu_dim1_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let cpu = var_host(
            Value::Tensor(tensor.clone()),
            &ParsedArguments {
                axes: VarAxes::Dim(1),
                normalization: VarNormalization::Sample,
                nan_mode: ReductionNaN::Include,
            },
        )
        .unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = var_gpu(
            handle,
            &ParsedArguments {
                axes: VarAxes::Dim(1),
                normalization: VarNormalization::Sample,
                nan_mode: ReductionNaN::Include,
            },
        )
        .unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                for (a, b) in ct.data.iter().zip(gt.data.iter()) {
                    assert!((a - b).abs() < 1e-6);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
