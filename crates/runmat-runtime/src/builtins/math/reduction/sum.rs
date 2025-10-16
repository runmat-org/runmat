//! Elementwise and reduction sum builtin for RunMat.
//!
//! This implementation mirrors MATLAB semantics, including optional `'omitnan'` handling,
//! tiered GPU execution through RunMat Accelerate, and fusion metadata for the native planner.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg_attr(not(test), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "sum"
category: "math/reduction"
keywords: ["sum", "reduction", "gpu", "omitnan"]
summary: "Sum elements of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to host when omitnan is requested or the active provider lacks reduction hooks."
fusion:
  elementwise: false
  reduction: true
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::sum::tests"
  integration: "builtins::math::reduction::sum::tests::sum_gpu_provider_roundtrip"
---

# MATLAB-Compatible `sum`
`sum(x)` adds together elements of scalars, vectors, matrices, and higher-dimensional tensors.
When no dimension is supplied, the reduction runs along the first non-singleton dimension.

## Behaviour
- `sum(X)` on an `m × n` matrix returns a row vector (`1 × n`) with column sums.
- `sum(X, 2)` returns a column vector (`m × 1`) containing row sums.
- Logical inputs are promoted to double precision (`true → 1.0`, `false → 0.0`).
- `sum(..., 'omitnan')` ignores `NaN` values; if all entries are `NaN`, the result becomes `0`.
- `sum(..., 'includenan')` (default) propagates `NaN` when any element in the slice is `NaN`.
- Empty slices return zeros with MATLAB-compatible shape semantics.
- Dimensions larger than `ndims(X)` leave the input unchanged.

## GPU Execution
When RunMat Accelerate is active, tensors that already reside on the device stay on the GPU.
Providers may implement `reduce_sum_dim` or `reduce_sum` for fast execution; otherwise
RunMat transparently gathers the data and falls back to the host implementation.
`'omitnan'` always uses the host path today because providers do not yet accept NaN policies.

## Examples

```matlab
A = [1 2 3; 4 5 6];
colSums = sum(A);      % [5 7 9]
rowSums = sum(A, 2);   % [6; 15]
```

```matlab
values = [1 NaN 3];
total = sum(values, 'omitnan');   % 4
```

```matlab
G = gpuArray(rand(1024, 1024));
energy = sum(G .^ 2);
result = gather(energy);          % column sums computed on the device when supported
```

## See Also
[`prod`], [`mean`], [`cumsum`], [`gpuArray`], [`gather`]
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sum",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Reduction {
        name: "reduce_sum_dim",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    notes:
        "Providers may specialise reduce_sum_dim / reduce_sum; omitnan falls back to the CPU path.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sum",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.get(0).ok_or(FusionError::MissingInput(0))?;
            Ok(format!("accumulator += {input};"))
        },
    }),
    emits_nan: false,
    notes: "Planner emits a standard column-major reduction template; providers can substitute custom kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[runtime_builtin(
    name = "sum",
    category = "math/reduction",
    summary = "Sum elements of scalars, vectors, matrices, or N-D tensors.",
    keywords = "sum,reduction,gpu,omitnan",
    accel = "reduction"
)]
fn sum_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (dim, nan_mode) = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => sum_gpu(handle, dim, nan_mode),
        other => sum_host(other, dim, nan_mode),
    }
}

fn parse_arguments(args: &[Value]) -> Result<(Option<usize>, ReductionNaN), String> {
    match args.len() {
        0 => Ok((None, ReductionNaN::Include)),
        1 => {
            if let Some(mode) = parse_nan_mode(&args[0])? {
                Ok((None, mode))
            } else {
                let dim = tensor::parse_dimension(&args[0], "sum")?;
                Ok((Some(dim), ReductionNaN::Include))
            }
        }
        2 => {
            let dim = tensor::parse_dimension(&args[0], "sum")?;
            if let Some(mode) = parse_nan_mode(&args[1])? {
                Ok((Some(dim), mode))
            } else {
                Err("sum: expected 'omitnan' or 'includenan' as the third argument".to_string())
            }
        }
        _ => Err("sum: unsupported arguments".to_string()),
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
        _ => Err(format!("sum: unknown reduction mode '{trimmed}'")),
    }
}

fn sum_host(value: Value, dim: Option<usize>, nan_mode: ReductionNaN) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor(value)?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let reduced = reduce_tensor_dim(&tensor, target_dim, nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn sum_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    nan_mode: ReductionNaN,
) -> Result<Value, String> {
    let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&handle.shape));

    if target_dim == 0 {
        return Err("sum: dimension must be >= 1".to_string());
    }

    let Some(target_shape) = reduction_shape(&handle.shape, target_dim) else {
        return Ok(Value::GpuTensor(handle));
    };

    if nan_mode == ReductionNaN::Include {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let zero_based = target_dim.saturating_sub(1);
            if zero_based < handle.shape.len() {
                if let Ok(device_result) = provider.reduce_sum_dim(&handle, zero_based) {
                    return Ok(Value::GpuTensor(device_result));
                }
            }

            if tensor::element_count(&target_shape) == 1 {
                if let Ok(device_result) = provider.reduce_sum(&handle) {
                    return Ok(Value::GpuTensor(device_result));
                }
            }
        }
    }

    let gathered = gpu_helpers::gather_tensor(&handle)?;
    let fallback_dim = dim.unwrap_or_else(|| default_dimension(&gathered));
    let reduced = reduce_tensor_dim(&gathered, fallback_dim, nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn reduce_tensor_dim(
    tensor: &Tensor,
    dim: usize,
    nan_mode: ReductionNaN,
) -> Result<Tensor, String> {
    if dim == 0 {
        return Err("sum: dimension must be >= 1".to_string());
    }

    if tensor.data.is_empty() {
        if let Some(shape) = reduction_shape(&tensor.shape, dim) {
            let zeros = vec![0.0; tensor::element_count(&shape)];
            return Tensor::new(zeros, shape).map_err(|e| format!("sum: {e}"));
        } else {
            return Ok(tensor.clone());
        }
    }

    if tensor.shape.is_empty() {
        let value = tensor.data[0];
        let reduced = match nan_mode {
            ReductionNaN::Include => value,
            ReductionNaN::Omit => {
                if value.is_nan() {
                    0.0
                } else {
                    value
                }
            }
        };
        return Tensor::new(vec![reduced], vec![1, 1]).map_err(|e| format!("sum: {e}"));
    }

    let Some(output_shape) = reduction_shape(&tensor.shape, dim) else {
        return Ok(tensor.clone());
    };

    let dim_index = dim - 1;
    let reduce_len = tensor.shape[dim_index];
    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);

    let out_len = tensor::element_count(&output_shape);
    let mut output = vec![0.0f64; out_len];

    for after in 0..stride_after {
        for before in 0..stride_before {
            let mut sum = 0.0;
            let mut any_value = false;
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
                        any_value = true;
                    }
                    ReductionNaN::Omit => {
                        if value.is_nan() {
                            continue;
                        }
                        sum += value;
                        any_value = true;
                    }
                }
            }

            let out_idx = after * stride_before + before;
            output[out_idx] = match nan_mode {
                ReductionNaN::Include => {
                    if saw_nan {
                        f64::NAN
                    } else if any_value {
                        sum
                    } else {
                        0.0
                    }
                }
                ReductionNaN::Omit => {
                    if any_value {
                        sum
                    } else {
                        0.0
                    }
                }
            };
        }
    }

    Tensor::new(output, output_shape).map_err(|e| format!("sum: {e}"))
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
    fn sum_scalar_num() {
        let result = sum_builtin(Value::Num(5.0), Vec::new()).expect("sum");
        assert_eq!(result, Value::Num(5.0));
    }

    #[test]
    fn sum_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), Vec::new()).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![5.0, 7.0, 9.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result =
            sum_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![6.0, 15.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_with_omit_nan_default_dimension() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("sum");
        assert_eq!(result, Value::Num(4.0));
    }

    #[test]
    fn sum_with_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), Vec::new()).expect("sum");
        match result {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[test]
    fn sum_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = tensor.clone();
        let value = Value::Tensor(tensor);
        let result = sum_builtin(value, vec![Value::Int(IntValue::I32(5))]).expect("sum");
        match result {
            Value::Tensor(out) => assert_eq!(out, original),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sum_builtin(Value::GpuTensor(handle), Vec::new()).expect("sum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.data, vec![5.0, 7.0, 9.0]);
        });
    }

    #[test]
    fn sum_gpu_omit_nan_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                sum_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("sum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(gathered.data, vec![2.0, 4.0]);
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
