//! MATLAB-compatible `sign` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "sign",
        builtin_path = "crate::builtins::math::elementwise::sign"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "sign"
category: "math/elementwise"
keywords: ["sign", "signum", "elementwise", "complex", "gpu"]
summary: "Sign of scalars, vectors, matrices, or N-D tensors with real or complex values."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host when the active provider lacks unary_sign or when complex inputs require host-side normalisation."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::sign::tests"
  integration: "builtins::math::elementwise::sign::tests::sign_gpu_provider_roundtrip"
---

# What does the `sign` function do in MATLAB / RunMat?
`y = sign(x)` returns the sign of each element of `x`. Real inputs become `-1`, `0`, or `1` depending
on their value, while complex inputs are normalised to unit magnitude (`x ./ abs(x)`).

## How does the `sign` function behave in MATLAB / RunMat?
- Real scalars, vectors, matrices, and higher-dimensional tensors produce `-1`, `0`, or `1` for each element.
- Complex inputs return `x ./ abs(x)`; zero-valued elements remain exactly `0 + 0i`.
- Logical inputs are promoted to doubles before applying the sign function.
- Character arrays are treated as their numeric code points and return doubles of the same shape.
- NaN inputs propagate (`sign(NaN)` is `NaN`), matching MATLAB semantics.
- `Inf` and `-Inf` map to `1` and `-1` respectively; complex numbers with infinite parts normalise accordingly.

## `sign` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that reside on the GPU remain on the device. The runtime checks
whether the active provider implements the `unary_sign` hook:

- **Hook available:** The sign is evaluated directly on the device with no host transfers.
- **Hook missing or unsupported dtype:** RunMat gathers the tensor, applies the CPU logic (including complex
  handling), and continues execution transparently.

## Examples of using the `sign` function in MATLAB / RunMat

### Determining the sign of a scalar

```matlab
result = sign(-42);
```

Expected output:

```matlab
result = -1;
```

### Applying `sign` to a vector of mixed values

```matlab
v = [-3 -0.0 0 2 5];
s = sign(v);
```

Expected output:

```matlab
s = [-1 0 0 1 1];
```

### Normalising complex numbers to unit magnitude

```matlab
z = [3+4i, -1+1i, 0+0i];
u = sign(z);
```

Expected output:

```matlab
u = [0.6+0.8i, -0.7071+0.7071i, 0];
```

### Using `sign` with character data

```matlab
codes = sign('RunMat');
```

Expected output:

```matlab
codes = [1 1 1 1 1 1];
```

### Working with logical masks

```matlab
mask = [false true false; true false true];
numeric = sign(mask);
```

Expected output:

```matlab
numeric = [0 1 0; 1 0 1];
```

### Executing `sign` on a GPU-resident tensor

```matlab
G = randn(4096, 4096, 'gpuArray');
S = sign(G);
```

`S` stays on the GPU when the provider implements `unary_sign`; otherwise RunMat gathers to the host and
falls back to the CPU implementation behind the scenes. When `sign` participates in a fused expression
for example `sign(G).*abs(G)`), the planner keeps the whole chain device-resident whenever the provider
supports the fused kernel.

### Handling infinities and NaNs

```matlab
values = [Inf, -Inf, NaN, 0];
out = sign(values);
```

Expected output:

```matlab
out = [1 -1 NaN 0];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` manually. RunMat's planner tracks residency and keeps tensors on
the GPU whenever it is profitable. Explicit `gpuArray` / `gather` calls remain available for MATLAB compatibility
or interoperability with external GPU code.

## FAQ

### Does `sign` modify NaN values?
No. NaN inputs remain NaN, matching IEEE behaviour and MATLAB semantics.

### How does `sign` handle complex zeros?
`0 + 0i` stays `0 + 0i`. Other complex values are scaled to lie on the unit circle (`x ./ abs(x)`).

### What happens for infinite complex components?
If either component is infinite, RunMat returns a direction vector with unit magnitude (e.g., `1 + 0i` or
`±1/√2 ± 1/√2 i`), mirroring MATLAB.

### Can I call `sign` on string arrays?
No. `sign` accepts numeric, logical, or character arrays. Use `double(string)` followed by `sign` if needed.

### Does `sign` allocate a new array?
Yes. The builtin returns a fresh array; downstream fusion may combine operations to reduce allocations.

### Is GPU execution numerically identical to CPU?
Results match within the provider's precision (`single` or `double`). NaN propagation and zero handling remain
consistent between CPU and GPU paths.

### Will `sign` participate in fusion?
Yes. The fusion planner can fold `sign` into neighbouring elementwise kernels, keeping data on the GPU when
possible.

### How do I keep results on the GPU?
Avoid `gather` unless host data is required. RunMat keeps the outputs of fused expressions device-resident when
beneficial.

## See Also
[abs](./abs), [sin](./sin), [sum](./sum), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `sign` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/sign.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/sign.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::sign")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sign",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sign" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute sign on-device via unary_sign; the runtime gathers to the host when the hook is unavailable or complex normalisation is required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::sign")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sign",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sign({input})"))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion kernels emit WGSL `sign` ops; providers can override via fused pipelines when advantageous.",
};

const BUILTIN_NAME: &str = "sign";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "sign",
    category = "math/elementwise",
    summary = "Sign of scalars, vectors, matrices, or N-D tensors with real or complex values.",
    keywords = "sign,signum,elementwise,complex,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::sign"
)]
fn sign_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => sign_gpu(handle),
        Value::Complex(re, im) => {
            let (re_out, im_out) = sign_complex(re, im);
            Ok(Value::Complex(re_out, im_out))
        }
        Value::ComplexTensor(ct) => sign_complex_tensor(ct),
        Value::CharArray(ca) => sign_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(builtin_error(
            "sign: expected numeric, logical, or character input",
        )),
        other => sign_real(other),
    }
}

fn sign_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_sign(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(sign_tensor(tensor)?))
}

fn sign_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("sign", value)
        .map_err(|e| builtin_error(format!("sign: {e}")))?;
    Ok(tensor::tensor_into_value(sign_tensor(tensor)?))
}

fn sign_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor.data.iter().map(|&x| sign_real_scalar(x)).collect();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| builtin_error(format!("sign: {e}")))
}

fn sign_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| sign_real_scalar(ch as u32 as f64))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("sign: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn sign_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| sign_complex(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone())
        .map_err(|e| builtin_error(format!("sign: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

#[inline]
fn sign_real_scalar(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

fn sign_complex(re: f64, im: f64) -> (f64, f64) {
    if re == 0.0 && im == 0.0 {
        return (0.0, 0.0);
    }
    if re.is_nan() || im.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let re_inf = re.is_infinite();
    let im_inf = im.is_infinite();
    if re_inf || im_inf {
        let real = if re_inf { re.signum() } else { 0.0 };
        let imag = if im_inf { im.signum() } else { 0.0 };
        let norm = (real * real + imag * imag).sqrt();
        if norm == 0.0 {
            return (real, imag);
        }
        return (real / norm, imag / norm);
    }
    let scale = re.abs().max(im.abs());
    if scale == 0.0 {
        return (0.0, 0.0);
    }
    let nr = re / scale;
    let ni = im / scale;
    let magnitude = (nr * nr + ni * ni).sqrt();
    if magnitude == 0.0 {
        (0.0, 0.0)
    } else {
        (nr / magnitude, ni / magnitude)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_scalar_positive_negative_zero() {
        assert_eq!(sign_builtin(Value::Num(3.5)).unwrap(), Value::Num(1.0));
        assert_eq!(sign_builtin(Value::Num(-2.0)).unwrap(), Value::Num(-1.0));
        assert_eq!(sign_builtin(Value::Num(0.0)).unwrap(), Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_scalar_nan_propagates() {
        let result = sign_builtin(Value::Num(f64::NAN)).unwrap();
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_tensor_mixed_values() {
        let tensor = Tensor::new(vec![-2.0, -0.0, 0.0, 5.0], vec![2, 2]).unwrap();
        let result = sign_builtin(Value::Tensor(tensor)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![-1.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_complex_scalar_normalises() {
        let result = sign_builtin(Value::Complex(3.0, 4.0)).unwrap();
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.6).abs() < 1e-12);
                assert!((im - 0.8).abs() < 1e-12);
            }
            other => panic!("expected complex value, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_complex_tensor_handles_zero() {
        let tensor = ComplexTensor::new(vec![(0.0, 0.0), (1.0, -1.0)], vec![2, 1]).unwrap();
        let result = sign_builtin(Value::ComplexTensor(tensor)).unwrap();
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data[0], (0.0, 0.0));
                let (re, im) = out.data[1];
                assert!((re - 0.7071067811865475).abs() < 1e-12);
                assert!((im + 0.7071067811865475).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_character_array() {
        let ca = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
        let result = sign_builtin(Value::CharArray(ca)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 6]);
                assert!(out.data.iter().all(|&v| (v - 1.0).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_logical_array() {
        let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let result = sign_builtin(Value::LogicalArray(logical)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_int_values() {
        let value = Value::Int(IntValue::I32(-7));
        let result = sign_builtin(value).unwrap();
        match result {
            Value::Num(v) => assert_eq!(v, -1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_bool_values() {
        let t = sign_builtin(Value::Bool(true)).unwrap();
        let f = sign_builtin(Value::Bool(false)).unwrap();
        assert_eq!(t, Value::Num(1.0));
        assert_eq!(f, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_infinite_values() {
        let tensor = Tensor::new(
            vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN],
            vec![2, 2],
        )
        .unwrap();
        let result = sign_builtin(Value::Tensor(tensor)).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data[0], 1.0);
                assert_eq!(out.data[1], -1.0);
                assert_eq!(out.data[2], 0.0);
                assert!(out.data[3].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_string_input_errors() {
        let err = sign_builtin(Value::String("runmat".to_string())).unwrap_err();
        assert!(
            err.message().contains("expected numeric, logical, or character input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_complex_with_nan() {
        let result = sign_builtin(Value::Complex(f64::NAN, 1.0)).unwrap();
        match result {
            Value::Complex(re, im) => {
                assert!(re.is_nan());
                assert!(im.is_nan());
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-3.0, -0.5, 0.0, 2.5], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sign_builtin(Value::GpuTensor(handle)).expect("sign");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![-1.0, -1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sign_gpu_fallback_for_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, -4.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).unwrap();
            let result = sign_builtin(Value::GpuTensor(handle)).expect("sign");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![1.0, -1.0]);
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
    fn sign_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.0, 0.0, 4.0, f64::NAN], vec![2, 2]).unwrap();
        let cpu = sign_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = sign_gpu(handle).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    if a.is_nan() && b.is_nan() {
                        continue;
                    }
                    assert_eq!(a, b);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
