//! MATLAB-compatible `cumprod` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, ProviderNanMode, ProviderScanDirection};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow};

const NAME: &str = "cumprod";

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "cumprod",
        builtin_path = "crate::builtins::math::reduction::cumprod"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "cumprod"
category: "math/reduction"
keywords: ["cumprod", "cumulative product", "running product", "reverse", "omitnan", "gpu"]
summary: "Cumulative product of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to host multiplication when the active provider lacks prefix-product hooks. The output always matches the input shape."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::cumprod::tests"
  integration: "builtins::math::reduction::cumprod::tests::cumprod_gpu_provider_roundtrip"
---

# What does the `cumprod` function do in MATLAB / RunMat?
`cumprod(X)` multiplies elements cumulatively along a chosen dimension. The result has the same size as `X`, and each element stores the running product.

## How does the `cumprod` function behave in MATLAB / RunMat?
- By default, the running product is taken along the first dimension whose length is greater than `1`.
- `cumprod(X, dim)` lets you choose the dimension explicitly; if `dim > ndims(X)`, the input is returned unchanged.
- Passing `[]` for the dimension argument keeps the default dimension (MATLAB treats it as a placeholder).
- `cumprod(..., "reverse")` accumulates from the end toward the beginning; `"forward"` (default) works from start to finish.
- `cumprod(..., "omitnan")` treats `NaN` values as missing. Empty prefixes yield `1`, the multiplicative identity.
- Synonyms such as `"omitmissing"` / `"includemissing"` are accepted for MATLAB compatibility.
- The function supports real or complex scalars and dense tensors. Logical inputs are promoted to double precision.

## `cumprod` Function GPU Execution Behavior
When data already lives on the GPU, RunMat asks the active acceleration provider for a device-side prefix-product implementation. The runtime calls the `cumprod_scan` hook with the chosen dimension, direction, and NaN mode. Providers that lack this hook—or that report an error for the requested options—trigger a gather to host memory, perform the cumulative product on the CPU, and return the dense tensor result. Residency metadata is cleared so downstream operations can re-promote the tensor when profitable.

## Examples of using the `cumprod` function in MATLAB / RunMat

### Running products down each column (default dimension)
```matlab
A = [1 2 3; 4 5 6];
columnProducts = cumprod(A);
```

Expected output:
```matlab
columnProducts =
     1     2     3
     4    10    18
```

### Tracking cumulative products across rows
```matlab
A = [1 2 3; 4 5 6];
rowProducts = cumprod(A, 2);
```

Expected output:
```matlab
rowProducts =
     1     2     6
     4    20   120
```

### Reversing the accumulation direction
```matlab
v = [2 3 4 5];
reverseProducts = cumprod(v, "reverse");
```

Expected output:
```matlab
reverseProducts =
   120    60    20     5
```

### Ignoring NaN values while multiplying
```matlab
v = [2 NaN 4 NaN 3];
running = cumprod(v, "omitnan");
```

Expected output:
```matlab
running =
     2     2     8     8    24
```

### Computing a cumulative product inside a GPU workflow
```matlab
G = gpuArray(1 + 0.1*rand(1, 5));
totals = cumprod(G);
hostResult = gather(totals);
```

Expected behavior:
- `totals` stays on the GPU when the provider implements prefix products.
- Otherwise RunMat gathers `G` to the host, performs the accumulation, and returns the CPU result.

## GPU residency in RunMat (Do I need `gpuArray`?)
Manual `gpuArray` calls are optional. RunMat promotes tensors automatically when the planner predicts a benefit, keeping fused expressions resident on the device. Explicit `gpuArray` is still supported for MATLAB compatibility or when you want to guarantee GPU residency before entering a critical loop.

## FAQ

### Does `cumprod` change the size of the input?
No. The output always equals the input shape.

### What happens if I request a dimension larger than `ndims(X)`?
The input is returned unchanged, matching MATLAB behaviour.

### How does `"omitnan"` treat leading NaN values?
They are ignored, so the cumulative product uses the multiplicative identity `1` until a finite value appears.

### Can I combine `"reverse"` and `"omitnan"`?
Yes. The options can be specified in any order and RunMat mirrors MATLAB’s results.

### Does the GPU path respect `"omitnan"`?
Only when the active provider offers a native prefix-product kernel with missing-value support. Otherwise the runtime gathers to the host to preserve MATLAB semantics.

## See Also
[prod](./prod), [cumsum](./cumsum), [sum](./sum), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/reduction/cumprod.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/cumprod.rs)
- Found a bug or behavioral difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::cumprod")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cumprod",
    op_kind: GpuOpKind::Custom("scan"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cumprod_scan")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may expose device prefix-product kernels; the runtime gathers to host when hooks are absent or options are unsupported.",
};

fn cumprod_error(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message).with_builtin(NAME).build().into()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::cumprod")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cumprod",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently lowers cumprod to the runtime implementation; providers can substitute specialised scan kernels.",
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CumprodDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CumprodNanMode {
    Include,
    Omit,
}

#[runtime_builtin(
    name = "cumprod",
    category = "math/reduction",
    summary = "Cumulative product of scalars, vectors, matrices, or N-D tensors.",
    keywords = "cumprod,cumulative product,running product,reverse,omitnan,gpu",
    accel = "reduction",
    builtin_path = "crate::builtins::math::reduction::cumprod"
)]
fn cumprod_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let (dim, direction, nan_mode) = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => cumprod_gpu(handle, dim, direction, nan_mode),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cumprod_error(format!("cumprod: {e}")))?;
            let target_dim = dim.unwrap_or(1);
            let result = cumprod_complex_tensor(&tensor, target_dim, direction, nan_mode)?;
            Ok(complex_tensor_into_value(result))
        }
        Value::ComplexTensor(ct) => {
            let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&ct.shape));
            let result = cumprod_complex_tensor(&ct, target_dim, direction, nan_mode)?;
            Ok(complex_tensor_into_value(result))
        }
        other => cumprod_host(other, dim, direction, nan_mode),
    }
}

fn parse_arguments(
    args: &[Value],
) -> BuiltinResult<(Option<usize>, CumprodDirection, CumprodNanMode)> {
    if args.len() > 3 {
        return Err(cumprod_error("cumprod: unsupported arguments"));
    }

    let mut dim: Option<usize> = None;
    let mut direction = CumprodDirection::Forward;
    let mut direction_set = false;
    let mut nan_mode = CumprodNanMode::Include;
    let mut nan_set = false;

    for value in args {
        match value {
            Value::Int(_) | Value::Num(_) => {
                if dim.is_some() {
                    return Err(cumprod_error("cumprod: dimension specified more than once"));
                }
                dim = Some(tensor::parse_dimension(value, "cumprod").map_err(|err| cumprod_error(err))?);
            }
            Value::Tensor(t) if t.data.is_empty() => {
                // MATLAB allows [] as a placeholder for the default dimension; ignore it.
            }
            Value::LogicalArray(la) if la.data.is_empty() => {
                // Treat logical [] the same way.
            }
            _ => {
                if let Some(text) = tensor::value_to_string(value) {
                    let keyword = text.trim().to_ascii_lowercase();
                    match keyword.as_str() {
                        "forward" => {
                            if direction_set {
                                return Err(
                                    "cumprod: direction specified more than once".to_string()
                                );
                            }
                            direction = CumprodDirection::Forward;
                            direction_set = true;
                        }
                        "reverse" => {
                            if direction_set {
                                return Err(
                                    "cumprod: direction specified more than once".to_string()
                                );
                            }
                            direction = CumprodDirection::Reverse;
                            direction_set = true;
                        }
                        "omitnan" | "omitmissing" => {
                            if nan_set {
                                return Err(
                                    "cumprod: missing-value handling specified more than once"
                                        .to_string(),
                                );
                            }
                            nan_mode = CumprodNanMode::Omit;
                            nan_set = true;
                        }
                        "includenan" | "includemissing" => {
                            if nan_set {
                                return Err(
                                    "cumprod: missing-value handling specified more than once"
                                        .to_string(),
                                );
                            }
                            nan_mode = CumprodNanMode::Include;
                            nan_set = true;
                        }
                        "" => {
                            return Err(cumprod_error("cumprod: empty string option is not supported"));
                        }
                        other => {
                            return Err(cumprod_error(format!("cumprod: unrecognised option '{other}'")));
                        }
                    }
                } else {
                    return Err(cumprod_error(format!("cumprod: unsupported argument type {value:?}")));
                }
            }
        }
    }

    Ok((dim, direction, nan_mode))
}

fn cumprod_host(
    value: Value,
    dim: Option<usize>,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cumprod", value).map_err(|err| cumprod_error(err))?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let result = cumprod_tensor(&tensor, target_dim, direction, nan_mode)?;
    Ok(tensor::tensor_into_value(result))
}

fn cumprod_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
) -> BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if matches!(direction, CumprodDirection::Reverse) && matches!(nan_mode, CumprodNanMode::Omit) {
        let tensor = gpu_helpers::gather_tensor(&handle)?;
        let fallback_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&tensor.shape));
        let result = cumprod_tensor(&tensor, fallback_dim, direction, nan_mode)?;
        return Ok(tensor::tensor_into_value(result));
    }

    if let Some(target) = dim {
        if target == 0 {
            return Err(cumprod_error("cumprod: dimension must be >= 1"));
        }
        if target > handle.shape.len() {
            return Ok(Value::GpuTensor(handle));
        }
    }

    let fallback_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&handle.shape));
    if fallback_dim == 0 {
        return Err(cumprod_error("cumprod: dimension must be >= 1"));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let zero_based_dim = fallback_dim.saturating_sub(1);
        if zero_based_dim < handle.shape.len() {
            let provider_direction = match direction {
                CumprodDirection::Forward => ProviderScanDirection::Forward,
                CumprodDirection::Reverse => ProviderScanDirection::Reverse,
            };
            let provider_nan_mode = match nan_mode {
                CumprodNanMode::Include => ProviderNanMode::Include,
                CumprodNanMode::Omit => ProviderNanMode::Omit,
            };
            if let Ok(device_result) = provider.cumprod_scan(
                &handle,
                zero_based_dim,
                provider_direction,
                provider_nan_mode,
            ) {
                return Ok(Value::GpuTensor(device_result));
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let result = cumprod_tensor(&tensor, fallback_dim, direction, nan_mode)?;
    Ok(tensor::tensor_into_value(result))
}

fn cumprod_tensor(
    tensor: &Tensor,
    dim: usize,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
) -> BuiltinResult<Tensor> {
    if dim == 0 {
        return Err(cumprod_error("cumprod: dimension must be >= 1"));
    }
    if tensor.data.is_empty() || dim > tensor.shape.len() {
        return Ok(tensor.clone());
    }

    let dim_index = dim - 1;
    let segment_len = tensor.shape[dim_index];
    if segment_len == 0 {
        return Ok(tensor.clone());
    }

    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let block = stride_before * segment_len;
    let mut output = vec![0.0f64; tensor.data.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            match direction {
                CumprodDirection::Forward => {
                    let mut prod = 1.0f64;
                    let mut prod_is_nan = false;
                    for k in 0..segment_len {
                        let idx = base + before + k * stride_before;
                        let value = tensor.data[idx];
                        match nan_mode {
                            CumprodNanMode::Include => {
                                if prod_is_nan {
                                    output[idx] = f64::NAN;
                                    continue;
                                }
                                if value.is_nan() {
                                    prod_is_nan = true;
                                    output[idx] = f64::NAN;
                                } else {
                                    prod *= value;
                                    output[idx] = prod;
                                }
                            }
                            CumprodNanMode::Omit => {
                                if !value.is_nan() {
                                    prod *= value;
                                }
                                output[idx] = prod;
                            }
                        }
                    }
                }
                CumprodDirection::Reverse => {
                    let mut prod = 1.0f64;
                    let mut prod_is_nan = false;
                    for offset in (0..segment_len).rev() {
                        let idx = base + before + offset * stride_before;
                        let value = tensor.data[idx];
                        match nan_mode {
                            CumprodNanMode::Include => {
                                if prod_is_nan {
                                    output[idx] = f64::NAN;
                                    continue;
                                }
                                if value.is_nan() {
                                    prod_is_nan = true;
                                    output[idx] = f64::NAN;
                                } else {
                                    prod *= value;
                                    output[idx] = prod;
                                }
                            }
                            CumprodNanMode::Omit => {
                                if !value.is_nan() {
                                    prod *= value;
                                }
                                output[idx] = prod;
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::new(output, tensor.shape.clone()).map_err(|e| cumprod_error(format!("cumprod: {e}")))
}

fn cumprod_complex_tensor(
    tensor: &ComplexTensor,
    dim: usize,
    direction: CumprodDirection,
    nan_mode: CumprodNanMode,
) -> BuiltinResult<ComplexTensor> {
    if dim == 0 {
        return Err(cumprod_error("cumprod: dimension must be >= 1"));
    }
    if tensor.data.is_empty() || dim > tensor.shape.len() {
        return Ok(tensor.clone());
    }

    let dim_index = dim - 1;
    let segment_len = tensor.shape[dim_index];
    if segment_len == 0 {
        return Ok(tensor.clone());
    }

    let stride_before = dim_product(&tensor.shape[..dim_index]);
    let stride_after = dim_product(&tensor.shape[dim..]);
    let block = stride_before * segment_len;
    let mut output = vec![(0.0f64, 0.0f64); tensor.data.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            match direction {
                CumprodDirection::Forward => {
                    let mut prod = (1.0f64, 0.0f64);
                    let mut prod_is_nan = false;
                    for k in 0..segment_len {
                        let idx = base + before + k * stride_before;
                        let value = tensor.data[idx];
                        let value_is_nan = value.0.is_nan() || value.1.is_nan();
                        match nan_mode {
                            CumprodNanMode::Include => {
                                if prod_is_nan {
                                    output[idx] = (f64::NAN, f64::NAN);
                                    continue;
                                }
                                if value_is_nan {
                                    prod_is_nan = true;
                                    output[idx] = (f64::NAN, f64::NAN);
                                } else {
                                    prod = complex_mul(prod, value);
                                    output[idx] = prod;
                                }
                            }
                            CumprodNanMode::Omit => {
                                if !value_is_nan {
                                    prod = complex_mul(prod, value);
                                }
                                output[idx] = prod;
                            }
                        }
                    }
                }
                CumprodDirection::Reverse => {
                    let mut prod = (1.0f64, 0.0f64);
                    let mut prod_is_nan = false;
                    for offset in (0..segment_len).rev() {
                        let idx = base + before + offset * stride_before;
                        let value = tensor.data[idx];
                        let value_is_nan = value.0.is_nan() || value.1.is_nan();
                        match nan_mode {
                            CumprodNanMode::Include => {
                                if prod_is_nan {
                                    output[idx] = (f64::NAN, f64::NAN);
                                    continue;
                                }
                                if value_is_nan {
                                    prod_is_nan = true;
                                    output[idx] = (f64::NAN, f64::NAN);
                                } else {
                                    prod = complex_mul(prod, value);
                                    output[idx] = prod;
                                }
                            }
                            CumprodNanMode::Omit => {
                                if !value_is_nan {
                                    prod = complex_mul(prod, value);
                                }
                                output[idx] = prod;
                            }
                        }
                    }
                }
            }
        }
    }

    ComplexTensor::new(output, tensor.shape.clone()).map_err(|e| cumprod_error(format!("cumprod: {e}")))
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
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
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor as BuiltinsTensor};

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_scalar_num() {
        let result = cumprod_builtin(Value::Num(7.0), Vec::new()).expect("cumprod scalar");
        assert_eq!(result, Value::Num(7.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_matrix_default_dimension() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = cumprod_builtin(Value::Tensor(tensor), Vec::new()).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![1.0, 4.0, 2.0, 10.0, 3.0, 18.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_matrix_dimension_two() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = cumprod_builtin(Value::Tensor(tensor), args).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![1.0, 4.0, 2.0, 20.0, 6.0, 120.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_reverse_direction() {
        let tensor = BuiltinsTensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![4, 1]).unwrap();
        let result =
            cumprod_builtin(Value::Tensor(tensor), vec![Value::from("reverse")]).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![120.0, 60.0, 20.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_omit_nan_forward() {
        let tensor =
            BuiltinsTensor::new(vec![f64::NAN, 2.0, f64::NAN, 4.0], vec![4, 1]).expect("tensor");
        let result =
            cumprod_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![1.0, 2.0, 2.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_include_nan_propagates() {
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = cumprod_builtin(Value::Tensor(tensor), Vec::new()).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert!((out.data[0] - 1.0).abs() < 1e-12);
                assert!(out.data[1].is_nan());
                assert!(out.data[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_dim_greater_than_ndims_returns_input() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(5))];
        let result = cumprod_builtin(Value::Tensor(tensor.clone()), args).expect("cumprod");
        match result {
            Value::Tensor(out) => assert_eq!(out, tensor),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_complex_tensor() {
        let data = vec![(1.0, 2.0), (3.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 1]).unwrap();
        let result = cumprod_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("cumprod");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert!((out.data[0].0 - 1.0).abs() < 1e-12);
                assert!((out.data[0].1 - 2.0).abs() < 1e-12);
                assert!((out.data[1].0 - 5.0).abs() < 1e-12);
                assert!((out.data[1].1 - 5.0).abs() < 1e-12);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cumprod_builtin(Value::GpuTensor(handle), Vec::new()).expect("cumprod");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 2.0, 6.0, 24.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_reverse_with_omit_nan() {
        let tensor =
            BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0, 2.0], vec![4, 1]).expect("tensor");
        let result = cumprod_builtin(
            Value::Tensor(tensor),
            vec![Value::from("omitnan"), Value::from("reverse")],
        )
        .expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![8.0, 8.0, 8.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_accepts_omitmissing_synonym() {
        let tensor =
            BuiltinsTensor::new(vec![f64::NAN, 2.0, 3.0, f64::NAN], vec![4, 1]).expect("tensor");
        let result = cumprod_builtin(Value::Tensor(tensor), vec![Value::from("omitmissing")])
            .expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![1.0, 2.0, 6.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_accepts_includemissing_synonym() {
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0], vec![3, 1]).unwrap();
        let result = cumprod_builtin(Value::Tensor(tensor), vec![Value::from("includemissing")])
            .expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert!((out.data[0] - 1.0).abs() < 1e-12);
                assert!(out.data[1].is_nan());
                assert!(out.data[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_dimension_placeholder_is_ignored() {
        let tensor = BuiltinsTensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let placeholder = BuiltinsTensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let args = vec![Value::Tensor(placeholder), Value::from("reverse")];
        let result = cumprod_builtin(Value::Tensor(tensor), args).expect("cumprod");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 4.0, 10.0, 5.0, 18.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cumprod_dimension_zero_errors() {
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = cumprod_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))]);
        match result {
            Err(RuntimeControlFlow::Error(err)) => {
                assert!(
                    err.message().contains("dimension must be >= 1"),
                    "unexpected result: {err}"
                );
            }
            Err(RuntimeControlFlow::Suspend(_)) => panic!("unexpected suspension"),
            Ok(_) => panic!("expected error"),
        }
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
    fn cumprod_wgpu_forward_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let tensor = BuiltinsTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();

        let cpu = cumprod_host(
            Value::Tensor(tensor.clone()),
            Some(1),
            CumprodDirection::Forward,
            CumprodNanMode::Include,
        )
        .unwrap();

        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = cumprod_gpu(
            handle,
            Some(1),
            CumprodDirection::Forward,
            CumprodNanMode::Include,
        )
        .expect("cumprod gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");

        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                assert_eq!(gathered.data, ct.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cumprod_wgpu_reverse_omitnan_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let tensor = BuiltinsTensor::new(vec![1.0, f64::NAN, 4.0, 2.0], vec![4, 1]).unwrap();

        let cpu = cumprod_host(
            Value::Tensor(tensor.clone()),
            Some(1),
            CumprodDirection::Reverse,
            CumprodNanMode::Omit,
        )
        .unwrap();

        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = cumprod_gpu(
            handle,
            Some(1),
            CumprodDirection::Reverse,
            CumprodNanMode::Omit,
        )
        .expect("cumprod gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");

        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gathered.data.iter().zip(ct.data.iter()) {
                    if b.is_nan() {
                        assert!(a.is_nan());
                    } else {
                        assert!((a - b).abs() < tol);
                    }
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
