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
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
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

# What does the `sum` function do in MATLAB / RunMat?
`sum(x)` adds together elements of scalars, vectors, matrices, and higher-dimensional tensors.
When no dimension is supplied, the reduction runs along the first non-singleton dimension.

## How does the `sum` function behave in MATLAB / RunMat?
- `sum(X)` on an `m × n` matrix returns a row vector (`1 × n`) with column sums.
- `sum(X, 2)` returns a column vector (`m × 1`) containing row sums.
- Logical inputs are promoted to double precision (`true → 1.0`, `false → 0.0`).
- `sum(..., 'omitnan')` ignores `NaN` values; if all entries are `NaN`, the result becomes `0`.
- `sum(..., 'includenan')` (default) propagates `NaN` when any element in the slice is `NaN`.
- Empty slices return zeros with MATLAB-compatible shape semantics.
- Dimensions larger than `ndims(X)` leave the input unchanged.

## `sum` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that already reside on the device stay on the GPU.
If the provider lacks those hooks, RunMat gathers the data from the GPU to the host and falls back to the host implementation.

## Examples of using the `sum` function in MATLAB / RunMat

### Summing the elements of a matrix

```matlab
A = [1 2 3; 4 5 6];
colSums = sum(A);
rowSums = sum(A, 2);
```

Expected output:

```matlab
colSums = [5 7 9];
rowSums = [6; 15];
```

### Summing the elements of a vector with NaN values

```matlab
values = [1 NaN 3];
total = sum(values, 'omitnan');
```

Expected output:
```matlab
total = 4;
```

### Summing the elements of a matrix on a GPU

In RunMat:

```matlab
G = rand(1024, 1024);
result = sum(G .^ 2);
```

In MathWorks MATLAB (supported in RunMat as well):

```matlab
G = gpuArray(rand(1024, 1024));
energy = sum(G .^ 2);
result = gather(energy);
```

In both cases, the expected output is:

```matlab
result = [2.5 3.5 4.5];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, in the above example, the result of the `sum` call will already be on the GPU when the fusion planner has detected a net benefit to operating the fused expression it is part of on the GPU.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the core language, as their toolbox is a separate commercial product, MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.

## FAQ

### When should I use the `sum` function?

Use `sum` whenever you need to add together the elements of a tensor. This is useful for calculating totals, sums of squares, or performing statistical analysis.

### Does `sum` produce double arrays by default?

Yes, by default, `sum` creates dense double-precision arrays unless you explicitly specify a type such as `'single'` or use the `'like'` argument to match a prototype array.

### What does `sum(A)` return?

If you call `sum(A)`, where `A` is an array, the result is a new array of the same shape as `A` with the sum of each slice along the first non-singleton dimension. For example, if `A` is a 2x3 matrix, `sum(A)` will return a 1x3 matrix with the sum of each column.

### How do I compute the sum of a specific dimension?

You can use the `dim` argument to specify the dimension along which to compute the sum. For example, `sum(A, 2)` will return a 2x1 matrix with the sum of each row.

## See Also
[prod](./prod), [mean](./mean), [cumsum](./cumsum), [gpuArray](../accel/gpu_array), [gather](../accel/gather)

## Source & Feedback
- The full source code for the implementation of the `sum` function is available at: [`crates/runmat-runtime/src/builtins/math/reduction/sum.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/sum.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
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
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
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

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("sum", DOC_MD);

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
            // Accept either order: (dim, mode) or (mode, dim)
            if let Some(mode) = parse_nan_mode(&args[0])? {
                // (mode, dim)
                let dim = tensor::parse_dimension(&args[1], "sum")?;
                Ok((Some(dim), mode))
            } else {
                // (dim, mode)
                let dim = tensor::parse_dimension(&args[0], "sum")?;
                if let Some(mode) = parse_nan_mode(&args[1])? {
                    Ok((Some(dim), mode))
                } else {
                    Err("sum: expected 'omitnan' or 'includenan' as the third argument".to_string())
                }
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
    } else if nan_mode == ReductionNaN::Omit {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let axis = target_dim.saturating_sub(1);
            if handle.shape.len() == 2 && axis <= 1 {
                let rows = *handle.shape.get(0).unwrap_or(&1);
                let cols = *handle.shape.get(1).unwrap_or(&1);
                let (reduce_len, num_slices, axis_is_row) = if axis == 0 {
                    (rows, cols, false)
                } else {
                    (cols, rows, true)
                };
                let output_shape = reduction_shape(&handle.shape, target_dim)
                    .unwrap_or_else(|| vec![num_slices]);
                let scalar_ty = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F32 => "f32",
                    runmat_accelerate_api::ProviderPrecision::F64 => "f64",
                };
                // Minimal WGSL with omitnan=true; column-major addressing
                let mut shader = String::new();
                shader.push_str(&format!("struct Tensor {{ data: array<{scalar_ty}>; }}\n"));
                shader.push_str("struct MParams { nrows: u32, ncols: u32, ld: u32, flags: u32 }\n\n");
                shader.push_str("@group(0) @binding(0) var<storage, read> input0: Tensor;\n");
                shader.push_str("@group(0) @binding(1) var<storage, read_write> output: Tensor;\n");
                shader.push_str("@group(0) @binding(2) var<uniform> params: MParams;\n\n");
                shader.push_str("@compute @workgroup_size(256)\n");
                if axis_is_row {
                    shader.push_str(
                        "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n",
                    );
                    shader.push_str("  let row = wid.x; if (row >= params.nrows) { return; }\n");
                    shader.push_str(&format!(
                        "  var acc: {scalar_ty} = {}0.0;\n",
                        if scalar_ty == "f64" { "f64(" } else { "" }
                    ));
                    if scalar_ty == "f64" { shader.push_str("  // close f64 literal\n"); }
                    shader.push_str("  var c = lid.x;\n  while (c < params.ncols) {\n    let v = input0.data[row + (c * params.ld)];\n    if (!isNan(v)) { acc = acc + v; }\n    c += 256u;\n  }\n");
                } else {
                    shader.push_str(
                        "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n",
                    );
                    shader.push_str("  let col = wid.x; if (col >= params.ncols) { return; }\n");
                    shader.push_str(&format!(
                        "  var acc: {scalar_ty} = {}0.0;\n",
                        if scalar_ty == "f64" { "f64(" } else { "" }
                    ));
                    if scalar_ty == "f64" { shader.push_str("  // close f64 literal\n"); }
                    shader.push_str("  var r = lid.x;\n  while (r < params.nrows) {\n    let v = input0.data[(col * params.ld) + r];\n    if (!isNan(v)) { acc = acc + v; }\n    r += 256u;\n  }\n");
                }
                shader.push_str("  var<workgroup> tile: array<f32, 256u>;\n  tile[lid.x] = acc;\n  workgroupBarrier();\n");
                shader.push_str("  var off = 128u;\n  loop { if (off == 0u) { break; } if (lid.x < off) {\n    let a = tile[lid.x]; let b = tile[lid.x + off];\n    tile[lid.x] = a + b;\n  } workgroupBarrier(); off = off / 2u; }\n");
                if axis_is_row {
                    shader.push_str("  if (lid.x == 0u) { output.data[row] = tile[0u]; }\n}\n");
    } else {
                    shader.push_str("  if (lid.x == 0u) { output.data[col] = tile[0u]; }\n}\n");
                }
                if let Ok(device_result) = provider.fused_reduction(
                    &shader,
                    std::slice::from_ref(&handle),
                    &output_shape,
                    reduce_len,
                    num_slices,
                    256,
                ) {
                    return Ok(Value::GpuTensor(device_result));
                }
            }
        }
        // If provider path failed, fall back to host
        let gathered = gpu_helpers::gather_tensor(&handle)?;
        let reduced = reduce_tensor_dim(&gathered, target_dim, nan_mode)?;
        return Ok(tensor::tensor_into_value(reduced));
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
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
