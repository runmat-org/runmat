//! MATLAB-compatible `fftshift` builtin with GPU-aware semantics for RunMat.
//!
//! `fftshift` recenters zero-frequency components for outputs produced by FFTs.

use super::common::{apply_shift, build_shift_plan, compute_shift_dims, ShiftKind};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "fftshift")]
pub const DOC_MD: &str = r#"---
title: "fftshift"
category: "math/fft"
keywords: ["fftshift", "fourier transform", "frequency centering", "spectrum", "gpu"]
summary: "Shift zero-frequency components to the center of a spectrum."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses the provider circshift hook when available; otherwise gathers once, shifts on the host, and re-uploads."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::fft::fftshift::tests"
  integration: "builtins::math::fft::fftshift::tests::fftshift_gpu_roundtrip"
---

# What does the `fftshift` function do in MATLAB / RunMat?
`fftshift(X)` circularly shifts the output of an FFT so that the zero-frequency
component moves to the center of each transformed dimension. This makes spectra
easier to inspect and aligns with MATLAB's plotting conventions.

## How does the `fftshift` function behave in MATLAB / RunMat?
- When called without a dimension list, `fftshift` shifts along every dimension
  by `floor(size(X, dim) / 2)`.
- `fftshift(X, dims)` shifts only the specified dimensions. `dims` can be a
  scalar, vector, or logical mask.
- Dimensions of length 0 or 1 are left unchanged.
- Inputs can be real or complex and may already reside on the GPU (`gpuArray`).

## `fftshift` Function GPU Execution Behaviour
RunMat asks the active acceleration provider to execute `fftshift` via the
`circshift` hook (with predetermined offsets). If the provider cannot satisfy
the request, the tensor is gathered exactly once, shifted on the host, and
optionally re-uploaded. Scalars remain on their existing device.

## Examples of using the `fftshift` function in MATLAB / RunMat

### Centering the spectrum of a 1-D FFT result with even length

```matlab
x = [0 1 2 3 4 5 6 7];
fx = fft(x);
y = fftshift(fx);
```

Expected output:

```matlab
y = [4 5 6 7 0 1 2 3];
```

### Handling odd-length vectors

```matlab
x = 1:5;
y = fftshift(x);
```

Expected output:

```matlab
y = [4 5 1 2 3];
```

### Centering both axes of a 2-D FFT

```matlab
A = [1 2 3; 4 5 6];
C = fftshift(A);
```

Expected output:

```matlab
C =
     6     4     5
     3     1     2
```

### Shifting only one dimension of a matrix

```matlab
A = [1 2 3; 4 5 6];
rowCentered = fftshift(A, 1);   % shift rows only
```

Expected output:

```matlab
rowCentered =
     4     5     6
     1     2     3
```

### Applying `fftshift` to a gpuArray spectrum

```matlab
G = gpuArray(0:7);
centered = fftshift(G);
H = gather(centered);
```

Expected output:

```matlab
H = [4 5 6 7 0 1 2 3];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
RunMat's auto-offload keeps FFT spectra on the GPU whenever the active provider
exposes the `circshift` hook. In that case `fftshift` runs entirely on the
device and downstream fused operations continue without a gather. If no GPU
provider is registered—or it does not expose `circshift`—RunMat gathers the data
once, performs the host shift, and uploads the result back to the device so
subsequent work can still benefit from acceleration. You can always call
`gpuArray` explicitly when you need MATLAB compatibility or want to guarantee a
specific residency boundary.

## FAQ

### When should I call `fftshift`?
Call `fftshift` whenever you need to center FFT outputs before visualising
spectra, computing radial averages, or applying filters that expect zero
frequency in the middle of the array.

### Does `fftshift` modify the phase or magnitude of the FFT?
No. `fftshift` only reorders the samples. Magnitudes, phases, and the overall
information content remain unchanged.

### How do I undo `fftshift`?
Use `ifftshift`, which performs the inverse rearrangement. The sequence
`ifftshift(fftshift(X))` returns `X` for all supported inputs.

### Can I apply `fftshift` to only one dimension?
Yes. Pass a dimension index or vector, e.g. `fftshift(X, 2)` to shift column
channels only.

### Does `fftshift` work with gpuArray inputs?
Yes. RunMat keeps data on the GPU whenever the provider exposes the `circshift`
hook, matching MATLAB's `gpuArray` behaviour.

### How does `fftshift` handle empty inputs?
Empty arrays are returned unchanged with identical shape metadata.

### Can I use `fftshift` on logical arrays?
Yes. Logical arrays are shifted without changing their logical element type.

## See Also
[fft](./fft), [ifft](./ifft), [ifftshift](./ifftshift), [circshift](../../array/shape/circshift), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/fft/fftshift.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/fftshift.rs)
- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fftshift",
    op_kind: GpuOpKind::Custom("fftshift"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("circshift")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to provider circshift kernels when available; otherwise gathers once and shifts on the host.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fftshift",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not currently fused; treated as an explicit data shuffling operation.",
};

#[runtime_builtin(
    name = "fftshift",
    category = "math/fft",
    summary = "Shift zero-frequency components to the center of a spectrum.",
    keywords = "fftshift,fourier transform,frequency centering,spectrum,gpu",
    accel = "custom"
)]
fn fftshift_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() > 1 {
        return Err("fftshift: too many input arguments".to_string());
    }
    let dims_arg = rest.first();

    match value {
        Value::Tensor(tensor) => {
            let dims = compute_shift_dims(&tensor.shape, dims_arg, "fftshift")?;
            fftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)
        }
        Value::ComplexTensor(ct) => {
            let dims = compute_shift_dims(&ct.shape, dims_arg, "fftshift")?;
            fftshift_complex_tensor(ct, &dims).map(Value::ComplexTensor)
        }
        Value::LogicalArray(array) => {
            let dims = compute_shift_dims(&array.shape, dims_arg, "fftshift")?;
            fftshift_logical(array, &dims).map(Value::LogicalArray)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("fftshift: {e}"))?;
            let dims = compute_shift_dims(&tensor.shape, dims_arg, "fftshift")?;
            fftshift_complex_tensor(tensor, &dims).map(|result| {
                if result.data.len() == 1 {
                    let (r, i) = result.data[0];
                    Value::Complex(r, i)
                } else {
                    Value::ComplexTensor(result)
                }
            })
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("fftshift", value)?;
            let dims = compute_shift_dims(&tensor.shape, dims_arg, "fftshift")?;
            fftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)
        }
        Value::GpuTensor(handle) => {
            let dims = compute_shift_dims(&handle.shape, dims_arg, "fftshift")?;
            fftshift_gpu(handle, &dims)
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
            Err("fftshift: expected numeric or logical input".to_string())
        }
        Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err("fftshift: unsupported input type".to_string()),
    }
}

fn fftshift_tensor(tensor: Tensor, dims: &[usize]) -> Result<Tensor, String> {
    let Tensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return Tensor::new(data, shape).map_err(|e| format!("fftshift: {e}"));
    }
    let rotated = apply_shift(&data, &plan.ext_shape, &plan.positive)
        .map_err(|e| format!("fftshift: {e}"))?;
    Tensor::new(rotated, shape).map_err(|e| format!("fftshift: {e}"))
}

fn fftshift_complex_tensor(tensor: ComplexTensor, dims: &[usize]) -> Result<ComplexTensor, String> {
    let ComplexTensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return ComplexTensor::new(data, shape).map_err(|e| format!("fftshift: {e}"));
    }
    let rotated = apply_shift(&data, &plan.ext_shape, &plan.positive)
        .map_err(|e| format!("fftshift: {e}"))?;
    ComplexTensor::new(rotated, shape).map_err(|e| format!("fftshift: {e}"))
}

fn fftshift_logical(array: LogicalArray, dims: &[usize]) -> Result<LogicalArray, String> {
    let LogicalArray { data, shape } = array;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Fft);
    if data.is_empty() || plan.is_noop() {
        return LogicalArray::new(data, shape).map_err(|e| format!("fftshift: {e}"));
    }
    let rotated = apply_shift(&data, &plan.ext_shape, &plan.positive)
        .map_err(|e| format!("fftshift: {e}"))?;
    LogicalArray::new(rotated, shape).map_err(|e| format!("fftshift: {e}"))
}

fn fftshift_gpu(handle: GpuTensorHandle, dims: &[usize]) -> Result<Value, String> {
    let plan = build_shift_plan(&handle.shape, dims, ShiftKind::Fft);
    if plan.is_noop() {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut working = handle.clone();
        if plan.ext_shape != working.shape {
            match provider.reshape(&working, &plan.ext_shape) {
                Ok(reshaped) => working = reshaped,
                Err(_) => return fftshift_gpu_fallback(handle, dims),
            }
        }
        if let Ok(mut out) = provider.circshift(&working, &plan.provider) {
            if plan.ext_shape != handle.shape {
                match provider.reshape(&out, &handle.shape) {
                    Ok(restored) => out = restored,
                    Err(_) => {
                        let mut coerced = out.clone();
                        coerced.shape = handle.shape.clone();
                        out = coerced;
                    }
                }
            }
            return Ok(Value::GpuTensor(out));
        }
    }

    fftshift_gpu_fallback(handle, dims)
}

fn fftshift_gpu_fallback(handle: GpuTensorHandle, dims: &[usize]) -> Result<Value, String> {
    let host_tensor = gpu_helpers::gather_tensor(&handle)?;
    let shifted = fftshift_tensor(host_tensor, dims)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &shifted.data,
            shape: &shifted.shape,
        };
        return provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| format!("fftshift: {e}"));
    }
    Ok(tensor::tensor_into_value(shifted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ComplexTensor, IntValue, LogicalArray, Tensor};

    #[test]
    fn fftshift_even_length_vector() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![8, 1]);
                assert_eq!(out.data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_odd_length_vector() {
        let tensor = Tensor::new((1..=5).map(|v| v as f64).collect(), vec![5, 1]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![5, 1]);
                assert_eq!(out.data, vec![4.0, 5.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_matrix_rows_only() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))])
            .expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_matrix_columns_only_via_vector_dims() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let dims = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![3.0, 6.0, 1.0, 4.0, 2.0, 5.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_matrix_rows_only_logical_mask() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mask = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), vec![Value::LogicalArray(mask)])
            .expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_matrix_all_dims() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![6.0, 3.0, 4.0, 1.0, 5.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_with_empty_dimension_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(Vec::new(), vec![0, 1]).unwrap();
        let original = tensor.clone();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, original.shape);
                assert_eq!(out.data, original.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_dimension_beyond_rank_is_ignored() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
        let dims = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let original = tensor.clone();
        let result =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("fftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, original.shape);
                assert_eq!(out.data, original.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_logical_array_input_supported() {
        let logical = LogicalArray::new(vec![1, 0, 0, 0], vec![4, 1]).unwrap();
        let result = fftshift_builtin(Value::LogicalArray(logical), Vec::new()).expect("fftshift");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(out.data, vec![0, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_complex_tensor() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            vec![4, 1],
        )
        .unwrap();
        let result = fftshift_builtin(Value::ComplexTensor(tensor), Vec::new()).unwrap();
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(
                    out.data,
                    vec![(2.0, 2.0), (3.0, 3.0), (0.0, 0.0), (1.0, 1.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_complex_scalar_passthrough() {
        let result = fftshift_builtin(Value::Complex(1.0, -2.0), Vec::new()).expect("fftshift");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn fftshift_rejects_zero_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = fftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
            .unwrap_err();
        assert!(
            err.contains("dimension indices must be >= 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fftshift_rejects_non_integer_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = fftshift_builtin(Value::Tensor(tensor), vec![Value::Num(1.5)]).unwrap_err();
        assert!(
            err.contains("dimensions must be integers"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fftshift_rejects_non_numeric_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err =
            fftshift_builtin(Value::Tensor(tensor), vec![Value::from("invalid")]).unwrap_err();
        assert!(
            err.contains("dimension indices must be numeric"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fftshift_rejects_non_vector_dimension_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = fftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).unwrap_err();
        assert!(
            err.contains("dimension vectors must be row or column vectors"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fftshift_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("fftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![8, 1]);
            assert_eq!(gathered.data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn fftshift_gpu_with_explicit_dims() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let dims = Value::Int(IntValue::I32(1));
            let result = fftshift_builtin(Value::GpuTensor(handle), vec![dims]).expect("fftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
            assert_eq!(gathered.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn fftshift_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let cpu =
            fftshift_tensor(tensor.clone(), &(0..tensor.shape.len()).collect::<Vec<_>>()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = fftshift_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }
}
