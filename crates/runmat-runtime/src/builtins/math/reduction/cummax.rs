//! MATLAB-compatible `cummax` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_accelerate_api::{
    GpuTensorHandle, ProviderCummaxResult, ProviderNanMode, ProviderScanDirection,
};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "cummax"
category: "math/reduction"
keywords: ["cummax", "cumulative maximum", "running maximum", "reverse", "omitnan", "indices", "gpu"]
summary: "Running maximum and index tracking for scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses provider cummax_scan when available; otherwise gathers to host and computes MATLAB-compatible results."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::cummax::tests"
  integration: "builtins::math::reduction::cummax::tests::cummax_gpu_provider_roundtrip"
---

# What does the `cummax` function do in MATLAB / RunMat?
`cummax(X)` computes the cumulative maximum of the elements in `X` along a chosen dimension. It also tracks where each running maximum occurs, mirroring MATLAB's two-output behaviour.

## How does the `cummax` function behave in MATLAB / RunMat?
- By default the running maximum follows the first non-singleton dimension. Use `cummax(X, dim)` to choose a dimension explicitly; if `dim > ndims(X)`, the input is returned unchanged and the indices are ones.
- `[Y, I] = cummax(X, ...)` returns both the running maxima (`Y`) and the indices of where those maxima were observed (`I`). Indices are one-based and match MATLAB exactly.
- Add `"reverse"` (or `"forward"`) to control the scan direction. Reverse mode walks from the end of the chosen dimension back to the beginning.
- `"omitnan"` skips `NaN` values when choosing the running maximum, returning `NaN` only when every value seen so far is `NaN`. `"includenan"` (default) propagates `NaN` once one is encountered.
- Synonyms such as `"omitmissing"` / `"includemissing"` are accepted for MATLAB compatibility.
- Real and complex inputs are supported. Complex numbers are ordered using MATLAB's magnitude-and-angle rules.

## `cummax` Function GPU Execution Behaviour
When the input already resides on the GPU, RunMat calls the acceleration provider's `cummax_scan` hook. Providers that implement this hook return GPU handles for both the running maxima and their indices. If the hook is missing—or if it rejects the requested options (such as `"omitnan"` or `"reverse"`)—RunMat gathers the data to the host, computes the MATLAB-compatible result, and returns dense tensors on the CPU. Residency metadata is cleared so downstream kernels can re-promote values when profitable.

## Examples of using the `cummax` function in MATLAB / RunMat

### Tracking column-wise running maxima
```matlab
A = [4 2 7; 3 5 1];
[Y, I] = cummax(A);
```
Expected output:
```matlab
Y =
     4     2     7
     4     5     7
I =
     1     1     1
     1     2     1
```

### Requesting running maxima across rows
```matlab
A = [4 2 7; 3 5 1];
[Y, I] = cummax(A, 2);
```
Expected output:
```matlab
Y =
     4     4     7
     3     5     5
I =
     1     1     3
     1     2     2
```

### Getting cumulative maxima in reverse order
```matlab
v = [8 3 6 2];
[Y, I] = cummax(v, "reverse");
```
Expected output:
```matlab
Y = [8 6 6 2]
I = [1 3 3 4]
```

### Ignoring NaN values in running maxima
```matlab
v = [NaN 5 NaN 3];
[Y, I] = cummax(v, "omitnan");
```
Expected output:
```matlab
Y = [NaN 5 5 5]
I = [NaN 2 2 2]
```

### Capturing running maxima and indices on the GPU
```matlab
G = gpuArray([3 1 4 1 5]);
[Y, I] = cummax(G);
hostY = gather(Y);
hostI = gather(I);
```
Behaviour: When the active provider supplies `cummax_scan`, `Y` and `I` stay resident on the GPU. Otherwise RunMat gathers `G`, computes the result on the host, and returns dense CPU tensors so downstream code still matches MATLAB's results.
Expected output:
```matlab
hostY = [3 3 4 4 5];
hostI = [1 1 3 3 5];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Manual `gpuArray` calls are optional. The planner keeps tensors on the GPU when profitable, and the cummax builtin preserves residency whenever the provider implements `cummax_scan`. If the hook is unavailable, RunMat gathers to the host transparently and still returns MATLAB-compatible maxima and indices. You can still use `gpuArray` to match MATLAB scripts or force residency ahead of a tight GPU loop.

## FAQ

### Does `cummax` always return indices?
Yes. The builtin produces MATLAB-compatible indices internally. When a call requests two outputs (`[Y, I] = cummax(...)`), `I` is surfaced directly. For single-output calls the indices remain available to the runtime for later retrieval.

### How are complex numbers ordered?
Complex maxima follow MATLAB's rules: values compare by magnitude, and ties break by phase angle. `"omitnan"` treats elements with `NaN` real or imaginary parts as missing.

### What happens when all elements seen so far are `NaN` with `"omitnan"`?
The running maximum stays `NaN` and the corresponding index is `NaN` until a finite value is encountered. Once a finite value appears, subsequent `NaN`s leave the maximum and index unchanged.

### Does the `"reverse"` option change the reported indices?
Indices are still reported using 1-based positions along the chosen dimension. `"reverse"` simply walks the dimension from end to start before writing the outputs.

### What if the requested dimension exceeds `ndims(X)`?
The input is returned unchanged. Every index is `1`, matching MATLAB's treatment of singleton trailing dimensions.

## See Also
[max](./max), [cummin](./cummin), [cumsum](./cumsum), [cumprod](./cumprod), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/reduction/cummax.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/cummax.rs)
- Found a bug or behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cummax",
    op_kind: GpuOpKind::Custom("scan"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cummax_scan")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes:
        "Providers may expose prefix-max kernels that return running values and indices; the runtime gathers to host when hooks or options are unsupported.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cummax",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently lowers cummax to the runtime implementation; providers can substitute specialised scan kernels when available.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("cummax", DOC_MD);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CummaxDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CummaxNanMode {
    Include,
    Omit,
}

/// Evaluation artifact returned by `cummax` that carries both values and indices.
#[derive(Debug, Clone)]
pub struct CummaxEvaluation {
    values: Value,
    indices: Value,
}

impl CummaxEvaluation {
    /// Consume the evaluation and return only the running maxima (single-output call).
    pub fn into_value(self) -> Value {
        self.values
    }

    /// Consume the evaluation and return both maxima and indices.
    pub fn into_pair(self) -> (Value, Value) {
        (self.values, self.indices)
    }

    /// Peek at the indices without consuming the evaluation.
    pub fn indices_value(&self) -> Value {
        self.indices.clone()
    }
}

#[runtime_builtin(
    name = "cummax",
    category = "math/reduction",
    summary = "Cumulative maximum and index tracking for scalars, vectors, matrices, or N-D tensors.",
    keywords = "cummax,cumulative maximum,running maximum,reverse,omitnan,indices,gpu",
    accel = "reduction"
)]
fn cummax_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    evaluate(value, &rest).map(|eval| eval.into_value())
}

/// Evaluate the builtin once and expose both outputs (value + indices).
pub fn evaluate(value: Value, rest: &[Value]) -> Result<CummaxEvaluation, String> {
    let (dim, direction, nan_mode) = parse_arguments(rest)?;
    match value {
        Value::GpuTensor(handle) => cummax_gpu(handle, dim, direction, nan_mode),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("cummax: {e}"))?;
            let target_dim = dim.unwrap_or(1);
            let (values, indices) =
                cummax_complex_tensor(&tensor, target_dim, direction, nan_mode)?;
            Ok(CummaxEvaluation {
                values: complex_tensor_into_value(values),
                indices: tensor::tensor_into_value(indices),
            })
        }
        Value::ComplexTensor(ct) => {
            let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&ct.shape));
            let (values, indices) = cummax_complex_tensor(&ct, target_dim, direction, nan_mode)?;
            Ok(CummaxEvaluation {
                values: complex_tensor_into_value(values),
                indices: tensor::tensor_into_value(indices),
            })
        }
        other => cummax_host(other, dim, direction, nan_mode),
    }
}

fn parse_arguments(
    args: &[Value],
) -> Result<(Option<usize>, CummaxDirection, CummaxNanMode), String> {
    if args.len() > 4 {
        return Err("cummax: unsupported arguments".to_string());
    }

    let mut dim: Option<usize> = None;
    let mut direction = CummaxDirection::Forward;
    let mut direction_set = false;
    let mut nan_mode = CummaxNanMode::Include;
    let mut nan_set = false;

    for value in args {
        match value {
            Value::Int(_) | Value::Num(_) => {
                if dim.is_some() {
                    return Err("cummax: dimension specified more than once".to_string());
                }
                dim = Some(tensor::parse_dimension(value, "cummax")?);
            }
            Value::Tensor(t) if t.data.is_empty() => {
                // MATLAB allows [] placeholders; ignore them.
            }
            Value::LogicalArray(l) if l.data.is_empty() => {}
            _ => {
                if let Some(text) = tensor::value_to_string(value) {
                    let keyword = text.trim().to_ascii_lowercase();
                    match keyword.as_str() {
                        "forward" => {
                            if direction_set {
                                return Err(
                                    "cummax: direction specified more than once".to_string()
                                );
                            }
                            direction = CummaxDirection::Forward;
                            direction_set = true;
                        }
                        "reverse" => {
                            if direction_set {
                                return Err(
                                    "cummax: direction specified more than once".to_string()
                                );
                            }
                            direction = CummaxDirection::Reverse;
                            direction_set = true;
                        }
                        "omitnan" | "omitmissing" => {
                            if nan_set {
                                return Err(
                                    "cummax: missing-value handling specified more than once"
                                        .to_string(),
                                );
                            }
                            nan_mode = CummaxNanMode::Omit;
                            nan_set = true;
                        }
                        "includenan" | "includemissing" => {
                            if nan_set {
                                return Err(
                                    "cummax: missing-value handling specified more than once"
                                        .to_string(),
                                );
                            }
                            nan_mode = CummaxNanMode::Include;
                            nan_set = true;
                        }
                        "" => {
                            return Err("cummax: empty string option is not supported".to_string());
                        }
                        other => {
                            return Err(format!("cummax: unrecognised option '{other}'"));
                        }
                    }
                } else {
                    return Err(format!("cummax: unsupported argument type {value:?}"));
                }
            }
        }
    }

    Ok((dim, direction, nan_mode))
}

fn cummax_host(
    value: Value,
    dim: Option<usize>,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> Result<CummaxEvaluation, String> {
    let tensor = tensor::value_into_tensor_for("cummax", value)?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let (values, indices) = cummax_tensor(&tensor, target_dim, direction, nan_mode)?;
    Ok(CummaxEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

fn cummax_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> Result<CummaxEvaluation, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(target) = dim {
        if target == 0 {
            return Err("cummax: dimension must be >= 1".to_string());
        }
    }

    let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&handle.shape));
    if target_dim == 0 {
        return Err("cummax: dimension must be >= 1".to_string());
    }

    if target_dim > handle.shape.len() {
        let indices = ones_indices(&handle.shape)?;
        return Ok(CummaxEvaluation {
            values: Value::GpuTensor(handle),
            indices: tensor::tensor_into_value(indices),
        });
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let zero_based_dim = target_dim.saturating_sub(1);
        if zero_based_dim < handle.shape.len() {
            let provider_direction = match direction {
                CummaxDirection::Forward => ProviderScanDirection::Forward,
                CummaxDirection::Reverse => ProviderScanDirection::Reverse,
            };
            let provider_nan_mode = match nan_mode {
                CummaxNanMode::Include => ProviderNanMode::Include,
                CummaxNanMode::Omit => ProviderNanMode::Omit,
            };
            if let Ok(ProviderCummaxResult { values, indices }) = provider.cummax_scan(
                &handle,
                zero_based_dim,
                provider_direction,
                provider_nan_mode,
            ) {
                return Ok(CummaxEvaluation {
                    values: Value::GpuTensor(values),
                    indices: Value::GpuTensor(indices),
                });
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let (values, indices) = cummax_tensor(&tensor, target_dim, direction, nan_mode)?;
    Ok(CummaxEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

fn cummax_tensor(
    tensor: &Tensor,
    dim: usize,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> Result<(Tensor, Tensor), String> {
    if dim == 0 {
        return Err("cummax: dimension must be >= 1".to_string());
    }
    if tensor.data.is_empty() {
        let indices =
            Tensor::new(Vec::new(), tensor.shape.clone()).map_err(|e| format!("cummax: {e}"))?;
        return Ok((tensor.clone(), indices));
    }
    if dim > tensor.shape.len() {
        let indices = ones_indices(&tensor.shape)?;
        return Ok((tensor.clone(), indices));
    }

    let dim_index = dim - 1;
    let segment_len = tensor.shape[dim_index];
    if segment_len == 0 {
        let indices =
            Tensor::new(Vec::new(), tensor.shape.clone()).map_err(|e| format!("cummax: {e}"))?;
        return Ok((tensor.clone(), indices));
    }

    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let block = stride_before * segment_len;
    let mut values_out = vec![0.0f64; tensor.data.len()];
    let mut indices_out = vec![0.0f64; tensor.data.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            match direction {
                CummaxDirection::Forward => {
                    let mut current = 0.0f64;
                    let mut current_index = 0usize;
                    let mut has_value = false;
                    let mut nan_fixed = false;
                    let mut nan_index = 0usize;
                    for k in 0..segment_len {
                        let idx = base + before + k * stride_before;
                        let value = tensor.data[idx];
                        let position = k + 1;
                        match nan_mode {
                            CummaxNanMode::Include => {
                                if nan_fixed {
                                    values_out[idx] = f64::NAN;
                                    indices_out[idx] = nan_index as f64;
                                    continue;
                                }
                                if value.is_nan() {
                                    nan_fixed = true;
                                    nan_index = position;
                                    values_out[idx] = f64::NAN;
                                    indices_out[idx] = position as f64;
                                    continue;
                                }
                                if !has_value || value > current {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                            CummaxNanMode::Omit => {
                                if value.is_nan() {
                                    if has_value {
                                        values_out[idx] = current;
                                        indices_out[idx] = current_index as f64;
                                    } else {
                                        values_out[idx] = f64::NAN;
                                        indices_out[idx] = f64::NAN;
                                    }
                                    continue;
                                }
                                if !has_value || value > current {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                        }
                    }
                }
                CummaxDirection::Reverse => {
                    let mut current = 0.0f64;
                    let mut current_index = 0usize;
                    let mut has_value = false;
                    let mut nan_fixed = false;
                    let mut nan_index = 0usize;
                    for offset in (0..segment_len).rev() {
                        let idx = base + before + offset * stride_before;
                        let value = tensor.data[idx];
                        let position = offset + 1;
                        match nan_mode {
                            CummaxNanMode::Include => {
                                if nan_fixed {
                                    values_out[idx] = f64::NAN;
                                    indices_out[idx] = nan_index as f64;
                                    continue;
                                }
                                if value.is_nan() {
                                    nan_fixed = true;
                                    nan_index = position;
                                    values_out[idx] = f64::NAN;
                                    indices_out[idx] = position as f64;
                                    continue;
                                }
                                if !has_value || value > current {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                            CummaxNanMode::Omit => {
                                if value.is_nan() {
                                    if has_value {
                                        values_out[idx] = current;
                                        indices_out[idx] = current_index as f64;
                                    } else {
                                        values_out[idx] = f64::NAN;
                                        indices_out[idx] = f64::NAN;
                                    }
                                    continue;
                                }
                                if !has_value || value > current {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                        }
                    }
                }
            }
        }
    }

    let values_tensor =
        Tensor::new(values_out, tensor.shape.clone()).map_err(|e| format!("cummax: {e}"))?;
    let indices_tensor =
        Tensor::new(indices_out, tensor.shape.clone()).map_err(|e| format!("cummax: {e}"))?;
    Ok((values_tensor, indices_tensor))
}

fn cummax_complex_tensor(
    tensor: &ComplexTensor,
    dim: usize,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> Result<(ComplexTensor, Tensor), String> {
    if dim == 0 {
        return Err("cummax: dimension must be >= 1".to_string());
    }
    if tensor.data.is_empty() {
        let indices =
            Tensor::new(Vec::new(), tensor.shape.clone()).map_err(|e| format!("cummax: {e}"))?;
        return Ok((tensor.clone(), indices));
    }
    if dim > tensor.shape.len() {
        let indices = ones_indices(&tensor.shape)?;
        return Ok((tensor.clone(), indices));
    }

    let dim_index = dim - 1;
    let segment_len = tensor.shape[dim_index];
    if segment_len == 0 {
        let indices =
            Tensor::new(Vec::new(), tensor.shape.clone()).map_err(|e| format!("cummax: {e}"))?;
        return Ok((tensor.clone(), indices));
    }

    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let block = stride_before * segment_len;
    let mut values_out = vec![(0.0f64, 0.0f64); tensor.data.len()];
    let mut indices_out = vec![0.0f64; tensor.data.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            match direction {
                CummaxDirection::Forward => {
                    let mut current = (0.0f64, 0.0f64);
                    let mut current_index = 0usize;
                    let mut has_value = false;
                    let mut nan_fixed = false;
                    let mut nan_index = 0usize;
                    for k in 0..segment_len {
                        let idx = base + before + k * stride_before;
                        let value = tensor.data[idx];
                        let position = k + 1;
                        let value_is_nan = complex_is_nan(value);
                        match nan_mode {
                            CummaxNanMode::Include => {
                                if nan_fixed {
                                    values_out[idx] = complex_nan();
                                    indices_out[idx] = nan_index as f64;
                                    continue;
                                }
                                if value_is_nan {
                                    nan_fixed = true;
                                    nan_index = position;
                                    values_out[idx] = complex_nan();
                                    indices_out[idx] = position as f64;
                                    continue;
                                }
                                if !has_value || complex_greater(value, current) {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                            CummaxNanMode::Omit => {
                                if value_is_nan {
                                    if has_value {
                                        values_out[idx] = current;
                                        indices_out[idx] = current_index as f64;
                                    } else {
                                        values_out[idx] = complex_nan();
                                        indices_out[idx] = f64::NAN;
                                    }
                                    continue;
                                }
                                if !has_value || complex_greater(value, current) {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                        }
                    }
                }
                CummaxDirection::Reverse => {
                    let mut current = (0.0f64, 0.0f64);
                    let mut current_index = 0usize;
                    let mut has_value = false;
                    let mut nan_fixed = false;
                    let mut nan_index = 0usize;
                    for offset in (0..segment_len).rev() {
                        let idx = base + before + offset * stride_before;
                        let value = tensor.data[idx];
                        let position = offset + 1;
                        let value_is_nan = complex_is_nan(value);
                        match nan_mode {
                            CummaxNanMode::Include => {
                                if nan_fixed {
                                    values_out[idx] = complex_nan();
                                    indices_out[idx] = nan_index as f64;
                                    continue;
                                }
                                if value_is_nan {
                                    nan_fixed = true;
                                    nan_index = position;
                                    values_out[idx] = complex_nan();
                                    indices_out[idx] = position as f64;
                                    continue;
                                }
                                if !has_value || complex_greater(value, current) {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                            CummaxNanMode::Omit => {
                                if value_is_nan {
                                    if has_value {
                                        values_out[idx] = current;
                                        indices_out[idx] = current_index as f64;
                                    } else {
                                        values_out[idx] = complex_nan();
                                        indices_out[idx] = f64::NAN;
                                    }
                                    continue;
                                }
                                if !has_value || complex_greater(value, current) {
                                    has_value = true;
                                    current = value;
                                    current_index = position;
                                }
                                values_out[idx] = current;
                                indices_out[idx] = current_index as f64;
                            }
                        }
                    }
                }
            }
        }
    }

    let values_tensor =
        ComplexTensor::new(values_out, tensor.shape.clone()).map_err(|e| format!("cummax: {e}"))?;
    let indices_tensor =
        Tensor::new(indices_out, tensor.shape.clone()).map_err(|e| format!("cummax: {e}"))?;
    Ok((values_tensor, indices_tensor))
}

fn complex_greater(candidate: (f64, f64), current: (f64, f64)) -> bool {
    compare_complex_auto(candidate, current) == Ordering::Greater
}

fn complex_is_nan(value: (f64, f64)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn complex_nan() -> (f64, f64) {
    (f64::NAN, f64::NAN)
}

fn compare_complex_auto(a: (f64, f64), b: (f64, f64)) -> Ordering {
    let a_mag = magnitude_squared(a);
    let b_mag = magnitude_squared(b);
    if a_mag < b_mag {
        return Ordering::Less;
    }
    if a_mag > b_mag {
        return Ordering::Greater;
    }
    let a_angle = a.1.atan2(a.0);
    let b_angle = b.1.atan2(b.0);
    if a_angle < b_angle {
        Ordering::Less
    } else if a_angle > b_angle {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn magnitude_squared(z: (f64, f64)) -> f64 {
    z.0.mul_add(z.0, z.1 * z.1)
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn ones_indices(shape: &[usize]) -> Result<Tensor, String> {
    let len = tensor::element_count(shape);
    let data = if len == 0 {
        Vec::new()
    } else {
        vec![1.0f64; len]
    };
    Tensor::new(data, shape.to_vec()).map_err(|e| format!("cummax: {e}"))
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

fn dim_product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, value| acc.saturating_mul(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::IntValue;

    #[test]
    fn cummax_scalar_returns_value_and_index() {
        let eval = evaluate(Value::Num(7.0), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(7.0));
        assert_eq!(indices, Value::Num(1.0));
    }

    #[test]
    fn cummax_matrix_default_dimension() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 4.0, 2.0, 5.0, 7.0, 7.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.shape, vec![2, 3]);
                assert_eq!(idx.data, vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_dimension_two_tracks_rows() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![4.0, 3.0, 4.0, 5.0, 7.0, 5.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.data, vec![1.0, 1.0, 1.0, 2.0, 3.0, 2.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_reverse_direction() {
        let tensor = Tensor::new(vec![8.0, 3.0, 6.0, 2.0], vec![4, 1]).unwrap();
        let args = vec![Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.data, vec![8.0, 6.0, 6.0, 2.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.data, vec![1.0, 3.0, 3.0, 4.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_dimension_reverse_omitnan_combo() {
        let tensor =
            Tensor::new(vec![1.0, 5.0, f64::NAN, 2.0, 3.0, 4.0], vec![2, 3]).expect("tensor");
        let args = vec![
            Value::Int(IntValue::I32(2)),
            Value::from("reverse"),
            Value::from("omitnan"),
        ];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![3.0, 5.0, 3.0, 4.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.data, vec![3.0, 1.0, 3.0, 3.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_placeholder_allows_four_arguments() {
        let tensor =
            Tensor::new(vec![1.0, 5.0, f64::NAN, 2.0, 3.0, 4.0], vec![2, 3]).expect("tensor");
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).expect("placeholder");
        let args = vec![
            Value::Tensor(placeholder),
            Value::Int(IntValue::I32(2)),
            Value::from("reverse"),
            Value::from("omitnan"),
        ];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![3.0, 5.0, 3.0, 4.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.data, vec![3.0, 1.0, 3.0, 3.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_omit_nan_behaviour() {
        let tensor = Tensor::new(vec![f64::NAN, 5.0, f64::NAN, 3.0], vec![4, 1]).expect("tensor");
        let args = vec![Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert!(out.data[0].is_nan());
                assert_eq!(out.data[1], 5.0);
                assert_eq!(out.data[2], 5.0);
                assert_eq!(out.data[3], 5.0);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert!(idx.data[0].is_nan());
                assert_eq!(idx.data[1], 2.0);
                assert_eq!(idx.data[2], 2.0);
                assert_eq!(idx.data[3], 2.0);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_duplicate_direction_errors() {
        let err = evaluate(
            Value::Num(1.0),
            &[Value::from("reverse"), Value::from("forward")],
        );
        assert!(
            matches!(err, Err(message) if message.contains("direction specified more than once"))
        );
    }

    #[test]
    fn cummax_duplicate_nanflag_errors() {
        let err = evaluate(
            Value::Num(1.0),
            &[Value::from("omitnan"), Value::from("includenan")],
        );
        assert!(
            matches!(err, Err(message) if message.contains("missing-value handling specified more than once"))
        );
    }

    #[test]
    fn cummax_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.data[0], 1.0);
                assert!(out.data[1].is_nan());
                assert!(out.data[2].is_nan());
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.data[0], 1.0);
                assert_eq!(idx.data[1], 2.0);
                assert_eq!(idx.data[2], 2.0);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_dimension_greater_than_rank() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(4))];
        let eval = evaluate(Value::Tensor(tensor.clone()), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.data, tensor.data),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert!(idx.data.iter().all(|v| *v == 1.0)),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_allows_empty_dimension_placeholder() {
        let tensor = Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap();
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = [Value::Tensor(placeholder), Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.data, vec![3.0, 1.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.data, vec![1.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_dimension_zero_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = [Value::Int(IntValue::I32(0))];
        match evaluate(Value::Tensor(tensor), &args) {
            Ok(_) => panic!("expected dimension error"),
            Err(err) => assert!(err.contains("dimension must be >= 1")),
        }
    }

    #[test]
    fn cummax_reverse_omitnan_combination() {
        let tensor =
            Tensor::new(vec![f64::NAN, 4.0, 2.0, f64::NAN, 3.0], vec![5, 1]).expect("tensor");
        let args = [Value::from("reverse"), Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.data, vec![4.0, 4.0, 3.0, 3.0, 3.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.data, vec![2.0, 2.0, 5.0, 5.0, 5.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_complex_vector() {
        let tensor =
            ComplexTensor::new(vec![(3.0, 0.0), (2.0, 0.0), (2.0, 1.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::ComplexTensor(out) => {
                assert_eq!(out.data[0], (3.0, 0.0));
                assert_eq!(out.data[1], (3.0, 0.0));
                assert_eq!(out.data[2], (3.0, 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.data, vec![1.0, 1.0, 1.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummax");
            let (values, indices) = eval.into_pair();
            let gathered_values = test_support::gather(values).expect("gather values");
            let gathered_indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(gathered_values.data, vec![4.0, 4.0, 7.0, 7.0]);
            assert_eq!(gathered_indices.data, vec![1.0, 1.0, 3.0, 3.0]);
        });
    }

    #[test]
    fn cummax_gpu_dimension_exceeds_rank_returns_indices() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::Int(IntValue::I32(5))];
            let eval = evaluate(Value::GpuTensor(handle), &args).expect("cummax");
            let (values, indices) = eval.into_pair();
            let gathered_values = test_support::gather(values).expect("gather values");
            let gathered_indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(gathered_values.data, tensor.data);
            assert!(gathered_indices.data.iter().all(|v| *v == 1.0));
        });
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn cummax_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0, 5.0, 0.0], vec![3, 2]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cummax cpu");
        let (cpu_vals, cpu_idx) = cpu_eval.into_pair();
        let expected_vals = match cpu_vals {
            Value::Tensor(t) => t,
            other => panic!("expected tensor values from cpu eval, got {other:?}"),
        };
        let expected_idx = match cpu_idx {
            Value::Tensor(t) => t,
            other => panic!("expected tensor indices from cpu eval, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummax gpu");
        let (gpu_vals, gpu_idx) = gpu_eval.into_pair();

        match (&gpu_vals, &gpu_idx) {
            (Value::GpuTensor(_), Value::GpuTensor(_)) => {}
            other => panic!("expected GPU tensors, got {other:?}"),
        }

        let gathered_vals = test_support::gather(gpu_vals).expect("gather values");
        let gathered_idx = test_support::gather(gpu_idx).expect("gather indices");

        assert_eq!(gathered_vals.shape, expected_vals.shape);
        assert_eq!(gathered_vals.data, expected_vals.data);
        assert_eq!(gathered_idx.shape, expected_idx.shape);
        assert_eq!(gathered_idx.data, expected_idx.data);
    }
}
