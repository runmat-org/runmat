//! MATLAB-compatible `diff` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "diff")]
pub const DOC_MD: &str = r#"---
title: "diff"
category: "math/reduction"
keywords: ["diff", "difference", "finite difference", "nth difference", "gpu"]
summary: "Forward finite differences of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Calls the provider's `diff_dim` hook; falls back to host when that hook is unavailable."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::diff::tests"
  integration: "builtins::math::reduction::diff::tests::diff_gpu_provider_roundtrip"
---

# What does the `diff` function do in MATLAB / RunMat?
`diff(X)` computes forward finite differences along the first dimension of `X` whose size exceeds 1.
For vectors, this is simply the difference between adjacent elements. Higher-order differences are
obtained by repeating this process.

## How does the `diff` function behave in MATLAB / RunMat?
- `diff(X)` walks along the first non-singleton dimension. Column vectors therefore differentiate
  down the rows, while row vectors operate across columns.
- `diff(X, N)` computes the Nth forward difference. `N = 0` returns `X` unchanged. Each order reduces
  the length of the working dimension by one, so the output length becomes `max(len - N, 0)`.
- `diff(X, N, dim)` lets you choose the dimension explicitly. Passing `[]` for `N` or `dim` keeps the
  defaults, and dimensions larger than `ndims(X)` behave like length-1 axes (so any positive order
  yields an empty result).
- Real, logical, and character inputs promote to double precision tensors before differencing.
  Complex inputs retain their complex type, with forward differences applied to both the real and
  imaginary parts independently.
- Empty slices propagate: if the selected dimension has length 0 or 1, the corresponding axis in the
  output has length 0.

## `diff` Function GPU Execution Behaviour
When the operand already resides on the GPU, RunMat asks the active acceleration provider for a
finite-difference kernel via `diff_dim`. The WGPU backend implements this hook, so forward differences
execute entirely on the device and the result stays resident on the GPU. Providers that have not wired
`diff_dim` yet transparently gather the data, run the CPU implementation, and hand the result back to
the planner so subsequent kernels can re-promote it when beneficial.

## Examples of using the `diff` function in MATLAB / RunMat

### Computing first differences of a vector
```matlab
v = [3 4 9 15];
d1 = diff(v);
```
Expected output:
```matlab
d1 = [1 5 6];
```

### Taking second-order differences
```matlab
v = [1 4 9 16 25];
d2 = diff(v, 2);
```
Expected output:
```matlab
d2 = [2 2 2];
```

### Selecting the working dimension explicitly
```matlab
A = [1 2 3; 4 5 6];
rowDiff = diff(A, 1, 2);
```
Expected output:
```matlab
rowDiff =
     1     1
     1     1
```

### Running `diff` on GPU arrays
```matlab
G = gpuArray([1 4 9 16]);
gDiff = diff(G);
result = gather(gDiff);
```
Expected output:
```matlab
result = [3 5 7];
```

### N exceeding the dimension length returns an empty array
```matlab
v = (1:3)';
emptyResult = diff(v, 5);
```
Expected output:
```matlab
emptyResult =
  0×1 empty double column vector
```

### Applying `diff` to character data
```matlab
codes = diff('ACEG');
```
Expected output:
```matlab
codes = [2 2 2];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Manual `gpuArray` promotion is optional. RunMat keeps tensors on the GPU when providers implement
the relevant hooks and the planner predicts a benefit. With the WGPU backend registered, `diff`
executes fully on the GPU and returns a device-resident tensor. When the hook is missing, RunMat
gathers transparently, computes on the CPU, and keeps residency metadata consistent so fused
expressions can re-promote values when profitable.

## FAQ

### Does `diff` change the size of the input?
`diff` reduces the length along the working dimension by `N`. All other dimensions are preserved.
If the working dimension is shorter than `N`, the result is empty. With the WGPU backend the empty
result remains GPU-resident.

### How are higher-order differences computed?
RunMat applies the first-order forward difference repeatedly. This mirrors MATLAB’s definition and
produces the same numerical results.

### Can I pass `[]` for the order or dimension arguments?
Yes. An empty array keeps the default value (`N = 1`, first non-singleton dimension).

### Does `diff` support complex numbers?
Yes. Differences are taken on the real and imaginary parts independently, and the result remains
complex unless it becomes empty.

### What happens for character or logical inputs?
Characters and logical values are promoted to double precision differences, matching MATLAB.

### Will the GPU path produce the same results as the CPU path?
Yes. When a provider lacks a finite-difference kernel, RunMat gathers the data and computes on the CPU
to preserve MATLAB semantics exactly. Otherwise, the WGPU backend produces identical results on the GPU.

## See Also
[cumsum](./cumsum), [sum](./sum), [cumprod](./cumprod), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/reduction/diff.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/diff.rs)
- Found a bug or behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "diff",
    op_kind: GpuOpKind::Custom("finite-difference"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("diff_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers surface finite-difference kernels through `diff_dim`; the WGPU backend keeps tensors on the device.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diff",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently delegates to the runtime implementation; providers can override with custom kernels.",
};

#[runtime_builtin(
    name = "diff",
    category = "math/reduction",
    summary = "Forward finite differences of scalars, vectors, matrices, or N-D tensors.",
    keywords = "diff,difference,finite difference,nth difference,gpu",
    accel = "diff"
)]
fn diff_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (order, dim) = parse_arguments(&rest)?;
    if order == 0 {
        return Ok(value);
    }

    match value {
        Value::Tensor(tensor) => {
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("diff", value)?;
            diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor {
                data: vec![(re, im)],
                shape: vec![1, 1],
                rows: 1,
                cols: 1,
            };
            diff_complex_tensor(tensor, order, dim).map(complex_tensor_into_value)
        }
        Value::ComplexTensor(tensor) => {
            diff_complex_tensor(tensor, order, dim).map(complex_tensor_into_value)
        }
        Value::CharArray(chars) => diff_char_array(chars, order, dim),
        Value::GpuTensor(handle) => diff_gpu(handle, order, dim),
        other => Err(format!(
            "diff: unsupported input type {:?}; expected numeric, logical, or character data",
            other
        )),
    }
}

fn parse_arguments(args: &[Value]) -> Result<(usize, Option<usize>), String> {
    match args.len() {
        0 => Ok((1, None)),
        1 => {
            let order = parse_order(&args[0])?;
            Ok((order.unwrap_or(1), None))
        }
        2 => {
            let order = parse_order(&args[0])?.unwrap_or(1);
            let dim = parse_dimension_arg(&args[1])?;
            Ok((order, dim))
        }
        _ => Err("diff: unsupported arguments".to_string()),
    }
}

fn parse_order(value: &Value) -> Result<Option<usize>, String> {
    if is_empty_array(value) {
        return Ok(None);
    }
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err("diff: order must be a non-negative integer scalar".to_string());
            }
            Ok(Some(raw as usize))
        }
        Value::Num(n) => parse_numeric_order(*n).map(Some),
        Value::Tensor(t) if t.data.len() == 1 => parse_numeric_order(t.data[0]).map(Some),
        Value::Bool(b) => Ok(Some(if *b { 1 } else { 0 })),
        other => Err(format!(
            "diff: order must be a non-negative integer scalar, got {:?}",
            other
        )),
    }
}

fn parse_numeric_order(value: f64) -> Result<usize, String> {
    if !value.is_finite() {
        return Err("diff: order must be finite".to_string());
    }
    if value < 0.0 {
        return Err("diff: order must be a non-negative integer scalar".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("diff: order must be a non-negative integer scalar".to_string());
    }
    Ok(rounded as usize)
}

fn parse_dimension_arg(value: &Value) -> Result<Option<usize>, String> {
    if is_empty_array(value) {
        return Ok(None);
    }
    match value {
        Value::Int(_) | Value::Num(_) => tensor::parse_dimension(value, "diff").map(Some),
        Value::Tensor(t) if t.data.len() == 1 => {
            tensor::parse_dimension(&Value::Num(t.data[0]), "diff").map(Some)
        }
        other => Err(format!(
            "diff: dimension must be a positive integer scalar, got {:?}",
            other
        )),
    }
}

fn is_empty_array(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if t.data.is_empty())
}

fn diff_gpu(handle: GpuTensorHandle, order: usize, dim: Option<usize>) -> Result<Value, String> {
    let working_dim = dim.unwrap_or_else(|| default_dimension(&handle.shape));
    if working_dim == 0 {
        return Err("diff: dimension must be >= 1".to_string());
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(device_result) = provider.diff_dim(&handle, order, working_dim.saturating_sub(1))
        {
            return Ok(Value::GpuTensor(device_result));
        }
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
    diff_tensor_host(tensor, order, Some(working_dim)).map(tensor::tensor_into_value)
}

fn diff_char_array(chars: CharArray, order: usize, dim: Option<usize>) -> Result<Value, String> {
    if order == 0 {
        return Ok(Value::CharArray(chars));
    }
    let shape = vec![chars.rows, chars.cols];
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, shape).map_err(|e| format!("diff: {e}"))?;
    diff_tensor_host(tensor, order, dim).map(tensor::tensor_into_value)
}

pub fn diff_tensor_host(
    tensor: Tensor,
    order: usize,
    dim: Option<usize>,
) -> Result<Tensor, String> {
    let mut current = tensor;
    let mut working_dim = dim.unwrap_or_else(|| default_dimension(&current.shape));
    for _ in 0..order {
        current = diff_tensor_once(current, working_dim)?;
        if current.data.is_empty() {
            break;
        }
        // Preserve explicit dimension if the caller provided one; update when defaulting and shape shrinks.
        if dim.is_none() && dimension_length(&current.shape, working_dim) == 0 {
            working_dim = default_dimension(&current.shape);
        }
    }
    Ok(current)
}

fn diff_complex_tensor(
    tensor: ComplexTensor,
    order: usize,
    dim: Option<usize>,
) -> Result<ComplexTensor, String> {
    let mut current = tensor;
    let mut working_dim = dim.unwrap_or_else(|| default_dimension(&current.shape));
    for _ in 0..order {
        current = diff_complex_tensor_once(current, working_dim)?;
        if current.data.is_empty() {
            break;
        }
        if dim.is_none() && dimension_length(&current.shape, working_dim) == 0 {
            working_dim = default_dimension(&current.shape);
        }
    }
    Ok(current)
}

fn diff_tensor_once(tensor: Tensor, dim: usize) -> Result<Tensor, String> {
    let Tensor {
        data, mut shape, ..
    } = tensor;
    let dim_index = dim.saturating_sub(1);
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let len_dim = shape[dim_index];
    let mut output_shape = shape.clone();
    if len_dim <= 1 || data.is_empty() {
        output_shape[dim_index] = output_shape[dim_index].saturating_sub(1);
        return Tensor::new(Vec::new(), output_shape).map_err(|e| format!("diff: {e}"));
    }
    output_shape[dim_index] = len_dim - 1;
    let stride_before = product(&shape[..dim_index]);
    let stride_after = product(&shape[dim_index + 1..]);
    let output_len = stride_before * (len_dim - 1) * stride_after;
    let mut out = Vec::with_capacity(output_len);

    for after in 0..stride_after {
        let after_base = after * stride_before * len_dim;
        for before in 0..stride_before {
            for k in 0..(len_dim - 1) {
                let idx0 = before + after_base + k * stride_before;
                let idx1 = idx0 + stride_before;
                out.push(data[idx1] - data[idx0]);
            }
        }
    }

    Tensor::new(out, output_shape).map_err(|e| format!("diff: {e}"))
}

fn diff_complex_tensor_once(tensor: ComplexTensor, dim: usize) -> Result<ComplexTensor, String> {
    let ComplexTensor {
        data, mut shape, ..
    } = tensor;
    let dim_index = dim.saturating_sub(1);
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let len_dim = shape[dim_index];
    let mut output_shape = shape.clone();
    if len_dim <= 1 || data.is_empty() {
        output_shape[dim_index] = output_shape[dim_index].saturating_sub(1);
        return ComplexTensor::new(Vec::new(), output_shape).map_err(|e| format!("diff: {e}"));
    }
    output_shape[dim_index] = len_dim - 1;
    let stride_before = product(&shape[..dim_index]);
    let stride_after = product(&shape[dim_index + 1..]);
    let mut out = Vec::with_capacity(stride_before * (len_dim - 1) * stride_after);

    for after in 0..stride_after {
        let after_base = after * stride_before * len_dim;
        for before in 0..stride_before {
            for k in 0..(len_dim - 1) {
                let idx0 = before + after_base + k * stride_before;
                let idx1 = idx0 + stride_before;
                let (re0, im0) = data[idx0];
                let (re1, im1) = data[idx1];
                out.push((re1 - re0, im1 - im0));
            }
        }
    }

    ComplexTensor::new(out, output_shape).map_err(|e| format!("diff: {e}"))
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&dim| dim > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn dimension_length(shape: &[usize], dim: usize) -> usize {
    let dim_index = dim.saturating_sub(1);
    if dim_index < shape.len() {
        shape[dim_index]
    } else {
        1
    }
}

fn product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, val| acc.saturating_mul(val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[test]
    fn diff_row_vector_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let result = diff_builtin(Value::Tensor(tensor), Vec::new()).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![3.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn diff_column_vector_second_order() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = diff_builtin(Value::Tensor(tensor), args).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![2.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn diff_matrix_along_columns() {
        let tensor = Tensor::new(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap();
        let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(2))];
        let result = diff_builtin(Value::Tensor(tensor), args).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(out.data, vec![1.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn diff_handles_empty_when_order_exceeds_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(5))];
        let result = diff_builtin(Value::Tensor(tensor), args).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape[0], 0);
                assert!(out.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn diff_char_array_promotes_to_double() {
        let chars = CharArray::new("ACEG".chars().collect(), 1, 4).unwrap();
        let result = diff_builtin(Value::CharArray(chars), Vec::new()).expect("diff");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![2.0, 2.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn diff_complex_tensor_preserves_type() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (3.0, 2.0), (6.0, 5.0)], vec![1, 3]).unwrap();
        let result = diff_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("diff");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec![(2.0, 1.0), (3.0, 3.0)]);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[test]
    fn diff_zero_order_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(0))];
        let result = diff_builtin(Value::Tensor(tensor.clone()), args).expect("diff");
        assert_eq!(result, Value::Tensor(tensor));
    }

    #[test]
    fn diff_accepts_empty_order_argument() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
        let baseline = diff_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("diff");
        let empty = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = diff_builtin(Value::Tensor(tensor), vec![Value::Tensor(empty)]).expect("diff");
        assert_eq!(result, baseline);
    }

    #[test]
    fn diff_accepts_empty_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![1, 4]).unwrap();
        let baseline = diff_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("diff");
        let empty = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = diff_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::Tensor(empty)],
        )
        .expect("diff");
        assert_eq!(result, baseline);
    }

    #[test]
    fn diff_rejects_negative_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(-1))];
        let err = diff_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert!(err.contains("non-negative"));
    }

    #[test]
    fn diff_rejects_non_integer_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Num(1.5)];
        let err = diff_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert!(err.contains("non-negative integer"));
    }

    #[test]
    fn diff_rejects_invalid_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(0))];
        let err = diff_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert!(err.contains("dimension must be >= 1"));
    }

    #[test]
    fn diff_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = diff_builtin(Value::GpuTensor(handle), Vec::new()).expect("diff");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.data, vec![3.0, 5.0]);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn diff_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![4, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];

        let cpu_result = diff_builtin(Value::Tensor(tensor.clone()), args.clone()).expect("diff");
        let expected = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = diff_builtin(Value::GpuTensor(handle), args).expect("diff");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gathered.shape, expected.shape);
        let tol = if matches!(
            provider.precision(),
            runmat_accelerate_api::ProviderPrecision::F32
        ) {
            1e-5
        } else {
            1e-12
        };
        for (a, b) in gathered.data.iter().zip(expected.data.iter()) {
            assert!((a - b).abs() < tol, "|{a} - {b}| >= {tol}");
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
