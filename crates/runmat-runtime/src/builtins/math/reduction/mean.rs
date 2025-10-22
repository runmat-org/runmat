//! Column-major mean reduction builtin for RunMat.
//!
//! Provides MATLAB-compatible behaviour, including optional `'omitnan'`
//! handling, GPU-aware execution via RunMat Accelerate, and fusion/GPU
//! metadata for the native planner.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
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
title: "mean"
category: "math/reduction"
keywords: ["mean", "average", "reduction", "gpu", "omitnan"]
summary: "Average elements of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to host whenever 'omitnan' is requested or the active provider lacks mean reductions."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::mean::tests"
  integration: "builtins::math::reduction::mean::tests::mean_gpu_provider_roundtrip"
---

# What does the `mean` function do in MATLAB / RunMat?
`mean(x)` computes the arithmetic mean of scalars, vectors, matrices, and higher-dimensional tensors.
When no dimension is supplied, the reduction runs along the first non-singleton dimension.

## How does the `mean` function behave in MATLAB / RunMat?
- `mean(X)` on an `m × n` matrix returns a row vector (`1 × n`) with column averages.
- `mean(X, 2)` returns a column vector (`m × 1`) containing row averages.
- Logical inputs are promoted to double precision (`true → 1.0`, `false → 0.0`).
- `mean(..., 'omitnan')` ignores `NaN` values; if every element in the slice is `NaN`, the result is `NaN`.
- `mean(..., 'includenan')` (default) propagates `NaN` when any element in the slice is `NaN`.
- Empty slices produce `NaN` outputs that follow MATLAB's shape semantics.
- Dimensions larger than `ndims(X)` leave the input unchanged.

## `mean` Function GPU Execution Behaviour
When the input tensor lives on the GPU, RunMat first asks the active acceleration provider for a device-side reduction buffer via `reduce_mean_dim` / `reduce_mean`.
If the provider lacks those hooks, RunMat gathers the data from the GPU to the host and falls back to the host implementation.

## Examples of using the `mean` function in MATLAB / RunMat

### Computing the mean of a matrix

```matlab
A = [1 2 3; 4 5 6];
colMeans = mean(A);      % [2.5 3.5 4.5]
rowMeans = mean(A, 2);   % [2; 5]
```

Expected output:

```matlab
colMeans = [2.5 3.5 4.5];
rowMeans = [2; 5];
```

### Computing the mean of a vector with NaN values

```matlab
values = [1 NaN 3];
avg = mean(values, 'omitnan');   % 2
```

Expected output:

```matlab
avg = 2;
```

### Computing the row vs column means of a matrix

```matlab
A = [1 2 3; 4 5 6];
colMeans = mean(A);      % [2.5 3.5 4.5]
rowMeans = mean(A, 2);   % [2; 5]
```

Expected output:

```matlab
colMeans = [2.5 3.5 4.5];
rowMeans = [2; 5];
```

### Computing the mean of a matrix on a GPU

In RunMat:

```matlab
G = rand(1024, 1024);
result = mean(G .^ 2);
```

In MathWorks MATLAB (supported in RunMat as well):

```matlab
G = gpuArray(rand(1024, 1024));
energy = mean(G .^ 2);
result = gather(energy);
```

In both cases, the expected output is:

```matlab
result = [2.5 3.5 4.5];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, in the above example, the result of the `mean` call will already be on the GPU when the fusion planner has detected a net benefit to operating the fused expression it is part of on the GPU.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the core language, as their toolbox is a separate commercial product, MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.

## FAQ

### When should I use the `mean` function?

Use `mean` whenever you need to compute the arithmetic mean of a tensor. This is useful for calculating averages, central tendencies, or performing statistical analysis.

### Does `mean` produce double arrays by default?

Yes, by default, `mean` creates dense double-precision arrays unless you explicitly specify a type such as `'single'` or use the `'like'` argument to match a prototype array.

### What does `mean(A)` return?

If you call `mean(A)`, where `A` is an array, the result is a new array of the same shape as `A` with the mean of each slice along the first non-singleton dimension. For example, if `A` is a 2x3 matrix, `mean(A)` will return a 1x3 matrix with the mean of each column.

### How do I compute the mean of a specific dimension?

You can use the `dim` argument to specify the dimension along which to compute the mean. For example, `mean(A, 2)` will return a 2x1 matrix with the mean of each row.

## See Also
[sum](./sum), [median](./median), [cumsum](./cumsum), [gpuArray](../accel/gpu_array), [gather](../accel/gather)

## Source & Feedback
- The full source code for the implementation of the `mean` function is available at: [`crates/runmat-runtime/src/builtins/math/reduction/mean.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/mean.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mean",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_mean_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_mean",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers can specialise mean reductions; omitnan currently falls back to the host.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mean",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion fallback currently gathers to host; future kernels will divide the accumulated sum by slice size.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("mean", DOC_MD);

#[runtime_builtin(
    name = "mean",
    category = "math/reduction",
    summary = "Average elements of scalars, vectors, matrices, or N-D tensors.",
    keywords = "mean,average,reduction,gpu,omitnan",
    accel = "reduction"
)]
fn mean_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (dim, nan_mode) = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => mean_gpu(handle, dim, nan_mode),
        other => mean_host(other, dim, nan_mode),
    }
}

fn parse_arguments(args: &[Value]) -> Result<(Option<usize>, ReductionNaN), String> {
    match args.len() {
        0 => Ok((None, ReductionNaN::Include)),
        1 => {
            if let Some(mode) = parse_nan_mode(&args[0])? {
                Ok((None, mode))
            } else {
                let dim = tensor::parse_dimension(&args[0], "mean")?;
                Ok((Some(dim), ReductionNaN::Include))
            }
        }
        2 => {
            let dim = tensor::parse_dimension(&args[0], "mean")?;
            if let Some(mode) = parse_nan_mode(&args[1])? {
                Ok((Some(dim), mode))
            } else {
                Err("mean: expected 'omitnan' or 'includenan' as the third argument".to_string())
            }
        }
        _ => Err("mean: unsupported arguments".to_string()),
    }
}

fn parse_nan_mode(value: &Value) -> Result<Option<ReductionNaN>, String> {
    let text = match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    };
    let Some(text) = text else {
        return Ok(None);
    };
    let trimmed = text.trim();
    let lowered = trimmed.to_ascii_lowercase();
    match lowered.as_str() {
        "omitnan" => Ok(Some(ReductionNaN::Omit)),
        "includenan" => Ok(Some(ReductionNaN::Include)),
        _ => Err(format!("mean: unknown reduction mode '{trimmed}'")),
    }
}

fn mean_host(value: Value, dim: Option<usize>, nan_mode: ReductionNaN) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("mean", value)?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let reduced = reduce_tensor_mean_dim(&tensor, target_dim, nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn mean_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    nan_mode: ReductionNaN,
) -> Result<Value, String> {
    let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&handle.shape));

    if target_dim == 0 {
        return Err("mean: dimension must be >= 1".to_string());
    }

    let Some(target_shape) = reduction_shape(&handle.shape, target_dim) else {
        return Ok(Value::GpuTensor(handle));
    };

    if nan_mode == ReductionNaN::Include {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let zero_based = target_dim.saturating_sub(1);
            if zero_based < handle.shape.len() {
                if let Ok(device_result) = provider.reduce_mean_dim(&handle, zero_based) {
                    return Ok(Value::GpuTensor(device_result));
                }
            }
            if tensor::element_count(&target_shape) == 1 {
                if let Ok(device_result) = provider.reduce_mean(&handle) {
                    return Ok(Value::GpuTensor(device_result));
                }
            }
        }
    }

    let gathered = gpu_helpers::gather_tensor(&handle)?;
    let fallback_dim = dim.unwrap_or_else(|| default_dimension(&gathered));
    let reduced = reduce_tensor_mean_dim(&gathered, fallback_dim, nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn reduce_tensor_mean_dim(
    tensor: &Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
) -> Result<Tensor, String> {
    if dim == 0 {
        return Err("mean: dimension must be >= 1".to_string());
    }

    if tensor.shape.is_empty() {
        let value = tensor.data.get(0).copied().unwrap_or(f64::NAN);
        let result = match nan_mode {
            ReductionNaN::Include => value,
            ReductionNaN::Omit => {
                if value.is_nan() {
                    f64::NAN
                } else {
                    value
                }
            }
        };
        return Tensor::new(vec![result], vec![1, 1]).map_err(|e| format!("mean: {e}"));
    }

    let Some(output_shape) = reduction_shape(&tensor.shape, dim) else {
        return Ok(tensor.clone());
    };

    if tensor.data.is_empty() {
        let fill = vec![f64::NAN; tensor::element_count(&output_shape)];
        return Tensor::new(fill, output_shape).map_err(|e| format!("mean: {e}"));
    }

    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let out_len = tensor::element_count(&output_shape);
    let mut output = vec![0.0f64; out_len];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut sum = 0.0;
            let mut count = 0usize;
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
                        sum += value;
                    }
                    ReductionNaN::Omit => {
                        if value.is_nan() {
                            continue;
                        }
                        sum += value;
                        count += 1;
                    }
                }
            }

            let out_idx = after * stride_before + before;
            output[out_idx] = match nan_mode {
                ReductionNaN::Include => {
                    if reduce_len == 0 || saw_nan {
                        f64::NAN
                    } else {
                        sum / (reduce_len as f64)
                    }
                }
                ReductionNaN::Omit => {
                    if count == 0 {
                        f64::NAN
                    } else {
                        sum / (count as f64)
                    }
                }
            };
        }
    }

    Tensor::new(output, output_shape).map_err(|e| format!("mean: {e}"))
}

fn reduction_shape(shape: &[usize], dim: usize) -> Option<Vec<usize>> {
    if dim == 0 {
        return None;
    }
    if shape.is_empty() {
        if dim == 1 {
            return Some(vec![1, 1]);
        }
        return None;
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
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::IntValue;

    #[test]
    fn mean_scalar_num() {
        let result = mean_builtin(Value::Num(6.0), Vec::new()).expect("mean");
        assert_eq!(result, Value::Num(6.0));
    }

    #[test]
    fn mean_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = mean_builtin(Value::Tensor(tensor), Vec::new()).expect("mean");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![2.5, 3.5, 4.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn mean_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("mean");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![2.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn mean_with_omit_nan_default_dimension() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 5.0], vec![3, 1]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("mean");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn mean_with_omit_nan_all_nan_returns_nan() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("mean");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN result, got {other:?}"),
        }
    }

    #[test]
    fn mean_with_include_nan_propagates_nan() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::from("includenan")]).expect("mean");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN result, got {other:?}"),
        }
    }

    #[test]
    fn mean_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = tensor.clone();
        let result =
            mean_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(5))]).expect("mean");
        match result {
            Value::Tensor(out) => assert_eq!(out, original),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn mean_dimension_with_omit_nan() {
        let tensor =
            Tensor::new(vec![1.0, f64::NAN, 3.0, 4.0], vec![2, 2]).expect("tensor construction");
        let result = mean_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::from("omitnan")],
        )
        .expect("mean");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![1.0, 3.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn mean_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = mean_builtin(Value::GpuTensor(handle), Vec::new()).expect("mean");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.data, vec![2.5, 3.5, 4.5]);
        });
    }

    #[test]
    fn mean_gpu_omit_nan_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                mean_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("mean");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(gathered.data, vec![2.0, 4.0]);
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
    fn mean_wgpu_dim1_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![1.0, 4.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let cpu = mean_host(Value::Tensor(t.clone()), Some(1), ReductionNaN::Include).unwrap();
        let view = runmat_accelerate_api::HostTensorView { data: &t.data, shape: &t.shape };
        let h = runmat_accelerate_api::provider().unwrap().upload(&view).unwrap();
        let gpu = mean_gpu(h, Some(1), ReductionNaN::Include).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
