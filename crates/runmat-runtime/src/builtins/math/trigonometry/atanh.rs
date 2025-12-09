//! MATLAB-compatible `atanh` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse hyperbolic tangent with full complex promotion and GPU fallbacks
//! mirroring MATLAB behaviour across scalars, tensors, logical inputs, and complex numbers.

use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const ZERO_EPS: f64 = 1.0e-12;
const DOMAIN_EPS: f64 = 1.0e-12;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "atanh")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "atanh"
category: "math/trigonometry"
keywords: ["atanh", "inverse hyperbolic tangent", "artanh", "trigonometry", "gpu", "complex"]
summary: "Element-wise inverse hyperbolic tangent with MATLAB-compatible complex promotion and GPU fallbacks."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses the provider's unary_atanh hook when every element satisfies |x| ≤ 1; gathers to the host when complex promotion is required or hooks are unavailable."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::atanh::tests"
  integration: "builtins::math::trigonometry::atanh::tests::atanh_gpu_provider_roundtrip"
---

# What does the `atanh` function do in MATLAB / RunMat?
`Y = atanh(X)` evaluates the inverse hyperbolic tangent of every element in `X`. Real inputs inside
the open interval `(-1, 1)` remain real. Values with `|x| > 1` (including `±Inf`) automatically
promote to complex results that match MATLAB's principal branch.

## How does the `atanh` function behave in MATLAB / RunMat?
- Accepts scalars, vectors, matrices, and N-D tensors using MATLAB broadcasting semantics.
- Logical values are promoted to double precision (`true → 1`, `false → 0`) before evaluation. The
  results follow MATLAB's rules for infinities at the endpoints (`atanh(1) = Inf`, `atanh(-1) = -Inf`).
- Real values with magnitude greater than `1` return complex numbers whose imaginary part is
  `±π/2`. The runtime automatically promotes the entire tensor to `Value::ComplexTensor` so downstream
  consumers see the MATLAB-compatible complex result.
- Complex inputs are evaluated with MATLAB's definition `atanh(z) = 0.5 * log((1 + z) / (1 - z))`,
  including correct handling of branch cuts, NaNs, and infinities.
- Character arrays are treated as their numeric code points and converted to doubles before the
  inverse hyperbolic tangent is applied.
- Empty arrays return empty outputs with matching shape and type.

## `atanh` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the GPU when:

1. A provider is registered and implements the `unary_atanh` hook.
2. Every element satisfies `|x| ≤ 1` (allowing the endpoints for the ±Inf behaviour).

The runtime first calls the provider's `reduce_min`/`reduce_max` hooks to prove that the entire operand
stays within the real domain `[-1, 1]`. If those hooks are missing, return an error, or detect a value
outside the domain, RunMat gathers the tensor to the host, performs the computation with the CPU
reference implementation, and returns MATLAB-accurate complex results. Users do not need to micromanage
residency with `gpuArray`/`gather`; the runtime transparently falls back only when necessary.

## Examples of using the `atanh` function in MATLAB / RunMat

### Inverse hyperbolic tangent of a real scalar

```matlab
y = atanh(0.5);
```

Expected output:

```matlab
y = 0.5493
```

### Applying atanh element-wise to a vector

```matlab
x = linspace(-0.9, 0.9, 5);
y = atanh(x);
```

Expected output:

```matlab
y = [-1.4722  -0.4847   0    0.4847   1.4722]
```

### Dealing with values near ±1 in atanh

```matlab
A = [0.99  1.0  -1.0; 0.0  0.5  -0.5];
B = atanh(A);
```

Expected output:

```matlab
B =
    2.6467        Inf       -Inf
         0    0.5493    -0.5493
```

### Producing complex outputs for |x| > 1

```matlab
values = [2 -3];
result = atanh(values);
```

Expected output:

```matlab
result =
   0.5493 + 1.5708i  -0.3466 + 1.5708i
```

### Computing atanh for complex numbers

```matlab
Z = [1 + 2i, -0.5 + 0.75i];
W = atanh(Z);
```

Expected output:

```matlab
W =
   0.1733 + 1.1781i  -0.3104 + 0.7232i
```

### Running atanh on GPU arrays

```matlab
G = gpuArray(linspace(-0.8, 0.8, 4));
gpuResult = atanh(G);
hostResult = gather(gpuResult);
```

Expected output:

```matlab
hostResult = [-1.0986  -0.2733   0.2733   1.0986]
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` manually. The auto-offload planner keeps tensors on
the GPU whenever the provider exposes a working `unary_atanh` hook and every element satisfies `|x| ≤ 1`.
If any element requires complex promotion or the provider lacks native support, RunMat automatically
gathers to the host and still returns MATLAB-compatible results. Manual `gpuArray` / `gather` remains
available for workflows that demand explicit residency control.

## FAQ

### When does `atanh` return complex numbers?
Whenever an element has magnitude strictly greater than `1`, including `±Inf`, MATLAB defines the
result as complex with an imaginary component of `±π/2`. RunMat mirrors that behaviour exactly.

### How are the endpoints `±1` handled?
`atanh(1)` returns positive infinity and `atanh(-1)` returns negative infinity, matching MATLAB. These
values still count as real outputs, so the GPU path can handle them without falling back.

### What happens with NaN inputs?
NaNs propagate through the computation. If the GPU provider supports `unary_atanh`, the operation can
remain on the device; otherwise the runtime falls back to the host and returns the same NaN results.

### Does the GPU path differ in precision from the CPU path?
Both paths evaluate the inverse hyperbolic tangent in the provider's precision (`f32` or `f64`). Small
rounding differences may occur but stay within MATLAB's tolerance requirements.

### Can complex inputs stay on the GPU?
Not yet. GPU tensor handles represent real buffers. When complex outputs are required (either because
the input is complex or a real value exceeds the domain), RunMat promotes the result to
`Value::Complex`/`Value::ComplexTensor` on the host so the semantics match MATLAB.

### Will `atanh` participate in fusion and autodiff?
Yes. The builtin registers element-wise fusion metadata so the planner can inline it into fused WGSL
kernels, and the same metadata feeds future autodiff tooling.

### Are logical and integer inputs supported?
Yes. They are promoted to doubles before evaluation. Outputs follow MATLAB's double or complex-double
conventions depending on the values.

### What if my provider does not implement `unary_atanh`?
RunMat automatically gathers the data, evaluates `atanh` on the CPU, and returns correct results.
Consider enabling the in-process provider for development or the WGPU provider for production to keep
the operation on the GPU.

## See Also
[tanh](./tanh), [asinh](./asinh), [acosh](./acosh), [atan](./atan), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `atanh` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/atanh.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/atanh.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "atanh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_atanh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Keeps tensors on the device when the provider exposes unary_atanh and every element satisfies |x| ≤ 1; otherwise gathers to the host for complex promotion.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "atanh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("atanh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `atanh` calls; providers can substitute custom kernels when available.",
};

#[runtime_builtin(
    name = "atanh",
    category = "math/trigonometry",
    summary = "Inverse hyperbolic tangent with MATLAB-compatible complex promotion.",
    keywords = "atanh,inverse hyperbolic tangent,artanh,gpu",
    accel = "unary"
)]
fn atanh_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => atanh_gpu(handle),
        Value::Complex(re, im) => Ok(atanh_complex_scalar(re, im)),
        Value::ComplexTensor(ct) => atanh_complex_tensor(ct),
        Value::CharArray(ca) => atanh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err("atanh: expected numeric input".to_string())
        }
        other => atanh_real(other),
    }
}

fn atanh_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match gpu_domain_is_real(provider, &handle) {
            Ok(true) => {
                if let Ok(out) = provider.unary_atanh(&handle) {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(false) => {
                // fall back to host below
            }
            Err(_) => {
                // fall back to host below
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    atanh_tensor_real(tensor)
}

fn gpu_domain_is_real(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<bool, String> {
    let min_handle = provider
        .reduce_min(handle)
        .map_err(|e| format!("atanh: reduce_min failed: {e}"))?;
    let max_handle = provider.reduce_max(handle).map_err(|e| {
        let _ = provider.free(&min_handle);
        format!("atanh: reduce_max failed: {e}")
    })?;

    let min_host = match provider.download(&min_handle) {
        Ok(values) => values,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(format!("atanh: reduce_min download failed: {err}"));
        }
    };
    let max_host = match provider.download(&max_handle) {
        Ok(values) => values,
        Err(err) => {
            let _ = provider.free(&min_handle);
            let _ = provider.free(&max_handle);
            return Err(format!("atanh: reduce_max download failed: {err}"));
        }
    };

    let _ = provider.free(&min_handle);
    let _ = provider.free(&max_handle);

    if min_host.data.is_empty() || max_host.data.is_empty() {
        return Err("atanh: reduce_min/reduce_max returned empty result".to_string());
    }

    let min_value = min_host.data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_value = max_host
        .data
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if !min_value.is_finite() || !max_value.is_finite() {
        return Ok(false);
    }

    if min_value < -1.0 - DOMAIN_EPS || max_value > 1.0 + DOMAIN_EPS {
        return Ok(false);
    }

    Ok(true)
}

fn atanh_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("atanh", value)?;
    atanh_tensor_real(tensor)
}

fn atanh_tensor_real(tensor: Tensor) -> Result<Value, String> {
    if tensor.data.is_empty() {
        return Ok(tensor::tensor_into_value(tensor));
    }

    let mut requires_complex = false;
    let mut real_values = Vec::with_capacity(tensor.data.len());
    let mut complex_values = Vec::with_capacity(tensor.data.len());

    for &x in &tensor.data {
        if x.is_finite() && x.abs() <= 1.0 {
            let re = zero_small(x.atanh());
            real_values.push(re);
            complex_values.push((re, 0.0));
        } else {
            let result = Complex64::new(x, 0.0).atanh();
            let re = zero_small(result.re);
            let im = zero_small(result.im);
            if im.abs() > ZERO_EPS {
                requires_complex = true;
            }
            real_values.push(re);
            complex_values.push((re, im));
        }
    }

    if requires_complex {
        if complex_values.len() == 1 {
            let (re, im) = complex_values[0];
            Ok(Value::Complex(re, im))
        } else {
            let tensor = ComplexTensor::new(complex_values, tensor.shape.clone())
                .map_err(|e| format!("atanh: {e}"))?;
            Ok(Value::ComplexTensor(tensor))
        }
    } else {
        let tensor =
            Tensor::new(real_values, tensor.shape.clone()).map_err(|e| format!("atanh: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn atanh_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    if ct.data.is_empty() {
        return Ok(Value::ComplexTensor(ct));
    }
    let mut mapped = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let result = Complex64::new(re, im).atanh();
        mapped.push((zero_small(result.re), zero_small(result.im)));
    }
    if mapped.len() == 1 {
        let (re, im) = mapped[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor =
            ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("atanh: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn atanh_complex_scalar(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).atanh();
    Value::Complex(zero_small(result.re), zero_small(result.im))
}

fn atanh_char_array(ca: CharArray) -> Result<Value, String> {
    if ca.data.is_empty() {
        let tensor =
            Tensor::new(Vec::new(), vec![ca.rows, ca.cols]).map_err(|e| format!("atanh: {e}"))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("atanh: {e}"))?;
    atanh_tensor_real(tensor)
}

fn zero_small(value: f64) -> f64 {
    if value.abs() < ZERO_EPS {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use num_complex::Complex64;
    use runmat_builtins::{CharArray, IntValue, LogicalArray};

    #[test]
    fn atanh_scalar_real() {
        let result = atanh_builtin(Value::Num(0.5)).expect("atanh");
        match result {
            Value::Num(v) => assert!((v - 0.5493061443340549).abs() < 1e-12),
            other => panic!("expected scalar real result, got {other:?}"),
        }
    }

    #[test]
    fn atanh_scalar_boundary() {
        let result = atanh_builtin(Value::Num(1.0)).expect("atanh");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_positive()),
            other => panic!("expected +Inf, got {other:?}"),
        }
        let result = atanh_builtin(Value::Num(-1.0)).expect("atanh");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected -Inf, got {other:?}"),
        }
    }

    #[test]
    fn atanh_tensor_real_values() {
        let tensor =
            Tensor::new(vec![0.0, 0.5, -0.5, 0.9], vec![2, 2]).expect("tensor construction");
        let result = atanh_builtin(Value::Tensor(tensor)).expect("atanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    0.0,
                    0.5493061443340549,
                    -0.5493061443340549,
                    1.4722194895832204,
                ];
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn atanh_real_promotes_to_complex() {
        let result = atanh_builtin(Value::Num(2.0)).expect("atanh");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(2.0, 0.0).atanh();
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn atanh_tensor_complex_output() {
        let tensor =
            Tensor::new(vec![2.0, -3.0, 0.5, -0.5], vec![2, 2]).expect("tensor construction");
        let result = atanh_builtin(Value::Tensor(tensor)).expect("atanh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let inputs = [
                    Complex64::new(2.0, 0.0),
                    Complex64::new(-3.0, 0.0),
                    Complex64::new(0.5, 0.0),
                    Complex64::new(-0.5, 0.0),
                ];
                let expected: Vec<Complex64> = inputs.iter().map(|z| z.atanh()).collect();
                for ((re, im), exp) in t.data.iter().zip(expected.iter()) {
                    assert!((re - exp.re).abs() < 1e-12);
                    assert!((im - exp.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn atanh_complex_inputs() {
        let inputs = [Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.75)];
        let complex = ComplexTensor::new(inputs.iter().map(|c| (c.re, c.im)).collect(), vec![1, 2])
            .expect("complex tensor");
        let result = atanh_builtin(Value::ComplexTensor(complex)).expect("atanh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (actual, input) in t.data.iter().zip(inputs.iter()) {
                    let expected = input.atanh();
                    assert!((actual.0 - expected.re).abs() < 1e-12);
                    assert!((actual.1 - expected.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[test]
    fn atanh_char_array_promotes_to_complex() {
        let chars = CharArray::new(vec!['A'], 1, 1).expect("char array");
        let result = atanh_builtin(Value::CharArray(chars)).expect("atanh");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new('A' as u32 as f64, 0.0).atanh();
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("expected complex scalar result, got {other:?}"),
        }
    }

    #[test]
    fn atanh_string_input_errors() {
        let err = atanh_builtin(Value::from("hello")).expect_err("expected error");
        assert!(err.contains("numeric"));
    }

    #[test]
    fn atanh_char_arrays() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).expect("chars");
        let result = atanh_builtin(Value::CharArray(chars)).expect("atanh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                // 'A' = 65, 'B' = 66 -> complex outputs
                for (idx, (re, im)) in t.data.iter().enumerate() {
                    let value = (65 + idx) as f64;
                    let expected = Complex64::new(value, 0.0).atanh();
                    assert!((re - expected.re).abs() < 1e-12);
                    assert!((im - expected.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[test]
    fn atanh_logical_array() {
        let logical =
            LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).expect("logical array creation");
        let result = atanh_builtin(Value::LogicalArray(logical)).expect("atanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t.data[0] == 0.0);
                assert!(t.data[1].is_infinite());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn atanh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![-0.5, -0.25, 0.25, 0.5], vec![2, 2]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&x| x.atanh()).collect();
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - exp).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn atanh_gpu_keeps_residency_for_real_inputs() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-0.75, -0.25, 0.25, 0.75], vec![2, 2])
                .expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
            match result {
                Value::GpuTensor(out_handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(out_handle.clone())).expect("gather");
                    let expected: Vec<f64> = tensor.data.iter().copied().map(f64::atanh).collect();
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for (actual, exp) in gathered.data.iter().zip(expected.iter()) {
                        assert!((actual - exp).abs() < 1e-12);
                    }
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[test]
    fn atanh_gpu_falls_back_for_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.5, 2.0], vec![2, 1]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
            match result {
                Value::ComplexTensor(t) => {
                    assert_eq!(t.shape, vec![2, 1]);
                    let expected: Vec<Complex64> = tensor
                        .data
                        .iter()
                        .map(|&x| Complex64::new(x, 0.0).atanh())
                        .collect();
                    for ((re, im), exp) in t.data.iter().zip(expected.iter()) {
                        assert!((re - exp.re).abs() < 1e-12);
                        assert!((im - exp.im).abs() < 1e-12);
                    }
                }
                Value::Complex(re, im) => {
                    let expected = Complex64::new(2.0, 0.0).atanh();
                    assert!((re - expected.re).abs() < 1e-12);
                    assert!((im - expected.im).abs() < 1e-12);
                }
                other => panic!("expected complex host result, got {other:?}"),
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn atanh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor =
            Tensor::new(vec![-0.8, -0.4, 0.4, 0.8], vec![2, 2]).expect("tensor construction");
        let expected: Vec<f64> = tensor.data.iter().map(|&x| x.atanh()).collect();

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let result = atanh_builtin(Value::GpuTensor(handle)).expect("atanh");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, tensor.shape);

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 5e-5,
        };

        for (actual, exp) in gathered.data.iter().zip(expected.iter()) {
            assert!((actual - exp).abs() < tol, "|{actual} - {exp}| >= {tol}");
        }
    }

    #[test]
    fn atanh_accepts_int_inputs() {
        let value = Value::Int(IntValue::I8(0));
        let result = atanh_builtin(value).expect("atanh");
        match result {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar real result, got {other:?}"),
        }
    }

    #[test]
    fn atanh_doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
