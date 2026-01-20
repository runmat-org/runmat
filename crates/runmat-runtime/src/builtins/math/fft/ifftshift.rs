//! MATLAB-compatible `ifftshift` builtin with GPU-aware semantics for RunMat.
//!
//! `ifftshift` moves the zero-frequency component back to the origin, undoing
//! the reordering performed by `fftshift` and preparing spectra for inverse FFTs.

use super::common::{apply_shift, build_shift_plan, compute_shift_dims, ShiftKind};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "ifftshift",
        builtin_path = "crate::builtins::math::fft::ifftshift"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ifftshift"
category: "math/fft"
keywords: ["ifftshift", "inverse fft shift", "frequency alignment", "gpu"]
summary: "Undo fftshift by moving the zero-frequency component back to the origin."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider circshift hooks when available; otherwise gathers, reorders on the host, and uploads again."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::fft::ifftshift::tests"
  integration: "builtins::math::fft::ifftshift::tests::ifftshift_gpu_roundtrip"
---

# What does the `ifftshift` function do in MATLAB / RunMat?
`ifftshift(X)` circularly shifts data so that the DC (zero-frequency) component
returns to index 1 along each transformed dimension. It is the inverse of
`fftshift` and is commonly used immediately before calling `ifft`.

## How does the `ifftshift` function behave in MATLAB / RunMat?
- When no dimensions are specified, every axis is shifted by `ceil(size(X, dim) / 2)`.
- Passing a list of dimensions restricts shifting to those axes; zeros and ones
  are treated as no-ops.
- Odd-length dimensions shift by one additional element compared with
  `fftshift`, matching MATLAB's `ifftshift` parity rules.
- Works for real, complex, logical, and GPU-resident tensors.
- Empty arrays and scalars are returned unchanged.

## `ifftshift` Function GPU Execution Behaviour
RunMat first attempts to execute the shift entirely on the GPU using the
provider's `circshift` hook. If that is unavailable, RunMat gathers the data
exactly once, performs the reorder on the host, and uploads the result so
downstream computations can continue on the device. Scalars remain on their
current device to avoid unnecessary transfers.

## Examples of using the `ifftshift` function in MATLAB / RunMat

### Undoing fftshift on an even-length spectrum

```matlab
x = 0:7;
y = ifftshift(x);
```

Expected output:

```matlab
y = [4 5 6 7 0 1 2 3];
```

### Undoing fftshift on an odd-length spectrum

```matlab
x = 1:5;
y = ifftshift(x);
```

Expected output:

```matlab
y = [3 4 5 1 2];
```

### Preparing data for inverse FFT

```matlab
fx = fft(rand(1, 8));
centered = fftshift(fx);
restored = ifftshift(centered);
signal = ifft(restored);
```

Expected behaviour: `restored` matches `fx` and `signal` equals the original
time-domain data (within floating-point tolerance).

### Shifting only selected dimensions

```matlab
A = reshape(1:12, 3, 4);
rowsShifted = ifftshift(A, 1);   % shift rows only
colsShifted = ifftshift(A, 2);   % shift columns only
```

Expected output:

```matlab
rowsShifted =
     2     5     8    11
     3     6     9    12
     1     4     7    10

colsShifted =
     7    10     1     4
     8    11     2     5
     9    12     3     6
```

### Using ifftshift with gpuArray data

```matlab
G = gpuArray(0:7);
shifted = ifftshift(G);
host = gather(shifted);
```

Expected output:

```matlab
host = [4 5 6 7 0 1 2 3];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You can rely on RunMat's auto-offload to keep FFT data on the GPU. When a
provider exposes `circshift`, `ifftshift` executes entirely on the device and
the result remains GPU-resident. If no provider is registered—or it lacks
`circshift`—RunMat gathers once, applies the host reorder, and uploads the
result so the rest of the computation can still run accelerated. You can call
`gpuArray` explicitly when you need MATLAB parity or to enforce a residency
boundary.

## FAQ

### When should I call `ifftshift`?
Call `ifftshift` right before `ifft` (or `ifftn`) after you have applied
`fftshift`-based processing in the frequency domain. It restores the DC
component to index 1.

### Is `ifftshift` always the inverse of `fftshift`?
Yes. For any supported input, `ifftshift(fftshift(X))` returns `X`, including
odd-length dimensions where the shift counts differ.

### Does `ifftshift` modify data values?
No. It only reorders elements. Magnitudes, phases, and overall content stay the
same.

### Can I restrict `ifftshift` to specific axes?
Yes. Pass a dimension index, vector of indices, or logical mask exactly like in
MATLAB.

### Does `ifftshift` support gpuArray inputs?
Yes. RunMat keeps GPU data on-device whenever possible and falls back to a
single gather/upload cycle otherwise.

## See Also
[fftshift](./fftshift), [fft](./fft), [ifft](./ifft), [circshift](./circshift), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/fft/ifftshift.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/ifftshift.rs)
- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifftshift")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifftshift",
    op_kind: GpuOpKind::Custom("ifftshift"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("circshift")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Delegates to provider circshift kernels; falls back to host when the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifftshift")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifftshift",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Explicit data movement; not fused with surrounding elementwise graphs.",
};

const BUILTIN_NAME: &str = "ifftshift";

fn ifftshift_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "ifftshift",
    category = "math/fft",
    summary = "Undo fftshift by moving the zero-frequency component back to the origin.",
    keywords = "ifftshift,inverse fft shift,frequency alignment,gpu",
    accel = "custom",
    builtin_path = "crate::builtins::math::fft::ifftshift"
)]
fn ifftshift_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(ifftshift_error("ifftshift: too many input arguments"));
    }
    let dims_arg = rest.first();

    match value {
        Value::Tensor(tensor) => {
            let dims = compute_shift_dims(&tensor.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::ComplexTensor(ct) => {
            let dims = compute_shift_dims(&ct.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_complex_tensor(ct, &dims).map(Value::ComplexTensor)?)
        }
        Value::LogicalArray(array) => {
            let dims = compute_shift_dims(&array.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_logical(array, &dims).map(Value::LogicalArray)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| ifftshift_error(format!("ifftshift: {e}")))?;
            let dims = compute_shift_dims(&tensor.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_complex_tensor(tensor, &dims).map(|result| {
                if result.data.len() == 1 {
                    let (r, i) = result.data[0];
                    Value::Complex(r, i)
                } else {
                    Value::ComplexTensor(result)
                }
            })?)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
                .map_err(|e| ifftshift_error(e))?;
            let dims = compute_shift_dims(&tensor.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => {
            let dims = compute_shift_dims(&handle.shape, dims_arg, BUILTIN_NAME)?;
            Ok(ifftshift_gpu(handle, &dims)?)
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => Err(
            ifftshift_error("ifftshift: expected numeric or logical input"),
        ),
        Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(ifftshift_error("ifftshift: unsupported input type")),
    }
}

fn ifftshift_tensor(tensor: Tensor, dims: &[usize]) -> BuiltinResult<Tensor> {
    let Tensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Ifft);
    if data.is_empty() || plan.is_noop() {
        return Tensor::new(data, shape).map_err(|e| ifftshift_error(format!("ifftshift: {e}")));
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    Tensor::new(rotated, shape).map_err(|e| ifftshift_error(format!("ifftshift: {e}")))
}

fn ifftshift_complex_tensor(tensor: ComplexTensor, dims: &[usize]) -> BuiltinResult<ComplexTensor> {
    let ComplexTensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Ifft);
    if data.is_empty() || plan.is_noop() {
        return ComplexTensor::new(data, shape)
            .map_err(|e| ifftshift_error(format!("ifftshift: {e}")));
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    ComplexTensor::new(rotated, shape).map_err(|e| ifftshift_error(format!("ifftshift: {e}")))
}

fn ifftshift_logical(array: LogicalArray, dims: &[usize]) -> BuiltinResult<LogicalArray> {
    let LogicalArray { data, shape } = array;
    let plan = build_shift_plan(&shape, dims, ShiftKind::Ifft);
    if data.is_empty() || plan.is_noop() {
        return LogicalArray::new(data, shape)
            .map_err(|e| ifftshift_error(format!("ifftshift: {e}")));
    }
    let rotated = apply_shift(BUILTIN_NAME, &data, &plan.ext_shape, &plan.positive)?;
    LogicalArray::new(rotated, shape).map_err(|e| ifftshift_error(format!("ifftshift: {e}")))
}

fn ifftshift_gpu(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let plan = build_shift_plan(&handle.shape, dims, ShiftKind::Ifft);
    if plan.is_noop() {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut working = handle.clone();
        if plan.ext_shape != working.shape {
            match provider.reshape(&working, &plan.ext_shape) {
                Ok(reshaped) => working = reshaped,
                Err(_) => return ifftshift_gpu_fallback(handle, dims),
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

    ifftshift_gpu_fallback(handle, dims)
}

fn ifftshift_gpu_fallback(handle: GpuTensorHandle, dims: &[usize]) -> BuiltinResult<Value> {
    let host_tensor = gpu_helpers::gather_tensor(&handle)?;
    let shifted = ifftshift_tensor(host_tensor, dims)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &shifted.data,
            shape: &shifted.shape,
        };
        return provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| ifftshift_error(format!("ifftshift: {e}")));
    }
    Ok(tensor::tensor_into_value(shifted))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::common::{apply_shift, build_shift_plan, ShiftKind};
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{ComplexTensor, IntValue, LogicalArray, Tensor};

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_even_length_vector() {
        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![8, 1]);
                assert_eq!(out.data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_odd_length_vector() {
        let tensor = Tensor::new((1..=5).map(|v| v as f64).collect(), vec![5, 1]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), Vec::new()).expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![5, 1]);
                assert_eq!(out.data, vec![3.0, 4.0, 5.0, 1.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_inverts_fftshift() {
        let tensor = Tensor::new((0..7).map(|v| v as f64).collect(), vec![7, 1]).unwrap();
        let fft_plan = build_shift_plan(&tensor.shape, &[0], ShiftKind::Fft);
        let fft_data = apply_shift(
            "ifftshift",
            &tensor.data,
            &fft_plan.ext_shape,
            &fft_plan.positive,
        )
        .expect("apply_shift");
        let fft_tensor = Tensor::new(fft_data, tensor.shape.clone()).unwrap();

        let restored =
            ifftshift_builtin(Value::Tensor(fft_tensor), Vec::new()).expect("ifftshift restore");
        match restored {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![7, 1]);
                assert_eq!(out.data, tensor.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_dimension_subset() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let dims = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result =
            ifftshift_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![2.0, 5.0, 3.0, 6.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_logical_mask_dimensions() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mask = LogicalArray::new(vec![0, 1], vec![2, 1]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), vec![Value::LogicalArray(mask)])
            .expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![2.0, 5.0, 3.0, 6.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_empty_dimension_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let empty_dims = Tensor::new(vec![], vec![0, 1]).unwrap();
        let result = ifftshift_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(empty_dims)],
        )
        .expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                assert_eq!(out.data, tensor.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_dimension_zero_error() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(
            ifftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
                .unwrap_err(),
        );
        assert!(
            err.contains("dimension indices must be >= 1"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_logical_array_supported() {
        let logical = LogicalArray::new(vec![1, 0, 0, 0], vec![4, 1]).unwrap();
        let result =
            ifftshift_builtin(Value::LogicalArray(logical), Vec::new()).expect("ifftshift");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(out.data, vec![0, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_rejects_non_numeric_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(
            ifftshift_builtin(Value::Tensor(tensor), vec![Value::from("invalid")]).unwrap_err(),
        );
        assert!(
            err.contains("dimension indices must be numeric"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                ifftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("ifftshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![8, 1]);
            assert_eq!(gathered.data, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
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
    fn ifftshift_complex_tensor() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            vec![4, 1],
        )
        .unwrap();
        let result = ifftshift_builtin(Value::ComplexTensor(tensor), Vec::new()).unwrap();
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifftshift_matrix_rows_only_via_int() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = ifftshift_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))])
            .expect("ifftshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ifftshift_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![8, 1]).unwrap();
        let cpu = ifftshift_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let handle = provider.upload(&view).expect("upload");

        let gpu_value =
            ifftshift_builtin(Value::GpuTensor(handle), Vec::new()).expect("ifftshift gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");

        match cpu {
            Value::Tensor(host) => {
                assert_eq!(gathered.shape, host.shape);
                assert_eq!(gathered.data, host.data);
            }
            other => panic!("expected tensor cpu result, got {other:?}"),
        }
    }
}
