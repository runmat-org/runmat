//! MATLAB-compatible `acosh` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse hyperbolic cosine with full complex promotion and GPU fallbacks
//! that mirror MATLAB behaviour for real, logical, character, and complex inputs.

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

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "acosh",
        wasm_path = "crate::builtins::math::trigonometry::acosh"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "acosh"
category: "math/trigonometry"
keywords: ["acosh", "inverse hyperbolic cosine", "arccosh", "trigonometry", "gpu", "complex"]
summary: "Element-wise inverse hyperbolic cosine with MATLAB-compatible complex promotion and GPU fallbacks."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses the provider's unary_acosh hook when every element stays within the real domain (x ≥ 1); gathers to the host when complex promotion is required or hooks are unavailable."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::acosh::tests"
  integration: "builtins::math::trigonometry::acosh::tests::acosh_gpu_provider_roundtrip"
---

# What does the `acosh` function do in MATLAB / RunMat?
`Y = acosh(X)` evaluates the inverse hyperbolic cosine of each element in `X`. Real inputs greater
than or equal to `1` stay real; values below `1` automatically promote to complex results that match
MATLAB's principal branch.

## How does the `acosh` function behave in MATLAB / RunMat?
- Accepts scalars, vectors, matrices, and N-D tensors with MATLAB broadcasting semantics.
- Logical inputs are promoted to double precision (`true → 1`, `false → 0`) before evaluation.
- Character arrays are interpreted as their numeric code points. If any element is less than `1`,
  the result becomes a complex array, mirroring MATLAB.
- Real values in `[1, ∞)` return real outputs computed by `acosh`.
- Real values in `(-∞, 1)` produce complex results: the runtime returns `Value::Complex` or
  `Value::ComplexTensor` so that downstream code sees the same behaviour as MATLAB.
- Complex inputs follow MATLAB's definition `acosh(z) = \log(z + \sqrt{z-1}\sqrt{z+1})`,
  including NaN/Inf handling on branch cuts.
- Special values propagate exactly like MATLAB: `acosh(NaN) = NaN`, `acosh(Inf) = Inf`, and
  `acosh(-Inf) = Inf + i·π`.

## `acosh` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the GPU when:

1. A provider is registered and implements the `unary_acosh` hook.
2. Every element is provably within the real domain (that is, all elements ≥ 1).
3. All inputs are finite; NaN or `-Inf` values force a host fallback so complex promotion and
   branch-cut semantics remain correct.

The runtime calls `reduce_min` on the active provider to perform this domain check. If the provider
does not expose the hook, the reduction fails, or any element violates the conditions above, RunMat
falls back to the host, evaluates `acosh` with the CPU reference implementation, and returns results
identical to MATLAB. This ensures correctness without forcing users to manually `gather` tensors.

## Examples of using the `acosh` function in MATLAB / RunMat

### Inverse hyperbolic cosine of a scalar greater than one

```matlab
y = acosh(1.5);
```

Expected output:

```matlab
y = 0.9624
```

### Applying `acosh` to each element of a vector

```matlab
x = [1 1.5 2 4];
y = acosh(x);
```

Expected output:

```matlab
y = [0 0.9624 1.31696 2.06344]
```

### Handling elements below one that produce complex results

```matlab
values = [0.5 1 2];
z = acosh(values);
```

Expected output:

```matlab
z =
   0.0000 + 1.0472i   0.0000 + 0.0000i   1.31696 + 0.0000i
```

### Computing `acosh` on GPU-resident data when the domain stays real

```matlab
G = gpuArray(linspace(1, 5, 5));
result_gpu = acosh(G);
result = gather(result_gpu);
```

Expected output:

```matlab
result = [0 1.31696 1.76275 2.06344 2.29243]
```

### Evaluating `acosh` for complex numbers

```matlab
z = [1 + 2i, -2 + 0.5i];
w = acosh(z);
```

Expected output:

```matlab
w =
   1.5286 + 1.1437i
   1.3090 + 2.3910i
```

### Working with character arrays (mix of complex and real results)

```matlab
C = char([0 65]);   % includes a code point below 1
Y = acosh(C);
```

Expected output:

```matlab
Y =
   0.0000 + 1.5708i   4.8675 + 0.0000i
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` manually. The auto-offload planner keeps tensors on
the GPU whenever the provider exposes `unary_acosh` and the input stays within the real domain.
When complex promotion is needed, RunMat gathers automatically and still returns MATLAB-compatible
results. Manual `gpuArray` / `gather` remains available for workflows that require explicit
residency control.

## FAQ

### Why does `acosh` sometimes return complex numbers?
The real-valued inverse hyperbolic cosine is only defined for `x ≥ 1`. Inputs below that range
require complex results, so RunMat (like MATLAB) promotes them automatically.

### Can `acosh` run entirely on the GPU?
Yes—when all elements are ≥ 1 and the provider implements `unary_acosh`, the runtime executes the
operation on the GPU. Otherwise, it falls back to the host transparently.

### How are NaN or Inf values handled?
`acosh(NaN)` returns `NaN`. Positive infinity stays real infinity. Negative infinity produces the
same complex result as MATLAB (`Inf + i·π`).

### Do logical and integer inputs work?
Yes. They are promoted to double precision before evaluation. The output is a dense double (or
complex double) array following MATLAB semantics.

### Can I keep complex results on the GPU?
Currently, GPU tensors represent real data. When complex outputs are required, the runtime gathers
to the host and returns `Value::Complex` / `Value::ComplexTensor` for correctness.

### Does `acosh` participate in fusion?
Yes. The fusion planner treats `acosh` as an element-wise operation and can inline it into fused
WGSL kernels when the provider supports the generated code.

### What tolerance does the runtime use to decide GPU fallback?
Any element below `1` triggers a host fallback so the runtime can return the correct complex result.
This mirrors MATLAB exactly instead of relying on GPU intrinsics that would otherwise yield `NaN`.

### Can `acosh` be differentiated automatically?
Yes. Marking it as an element-wise builtin ensures future autodiff tooling can reuse the same
metadata to generate gradients.

## See Also
[asinh](./asinh), [tanh](./tanh), [cosh](./cosh), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `acosh` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/acosh.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/acosh.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::math::trigonometry::acosh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "acosh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_acosh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute acosh directly on device buffers when inputs stay within the real domain (x ≥ 1); otherwise the runtime gathers to the host for complex promotion.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::math::trigonometry::acosh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "acosh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("acosh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `acosh` calls; providers can substitute custom kernels when available.",
};

#[runtime_builtin(
    name = "acosh",
    category = "math/trigonometry",
    summary = "Inverse hyperbolic cosine with MATLAB-compatible complex promotion.",
    keywords = "acosh,inverse hyperbolic cosine,arccosh,gpu",
    accel = "unary",
    wasm_path = "crate::builtins::math::trigonometry::acosh"
)]
fn acosh_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => acosh_gpu(handle),
        Value::Complex(re, im) => Ok(acosh_complex_scalar(re, im)),
        Value::ComplexTensor(ct) => acosh_complex_tensor(ct),
        Value::CharArray(ca) => acosh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err("acosh: expected numeric input".to_string())
        }
        other => acosh_real(other),
    }
}

fn acosh_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle) {
            Ok(false) => {
                if let Ok(out) = provider.unary_acosh(&handle) {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                return acosh_tensor_real(tensor);
            }
            Err(_) => {
                // Fall through to host path.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    acosh_tensor_real(tensor)
}

fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<bool, String> {
    let min_handle = provider
        .reduce_min(handle)
        .map_err(|e| format!("acosh: reduce_min failed: {e}"))?;
    let min_host = provider.download(&min_handle).map_err(|e| {
        let _ = provider.free(&min_handle);
        format!("acosh: reduce_min download failed: {e}")
    })?;
    let _ = provider.free(&min_handle);
    let min_value = min_host.data.iter().copied().fold(f64::INFINITY, f64::min);
    if !min_value.is_finite() {
        // NaN or -Inf: force host evaluation to preserve MATLAB semantics.
        return Ok(true);
    }
    Ok(min_value < 1.0)
}

fn acosh_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("acosh", value)?;
    acosh_tensor_real(tensor)
}

fn acosh_tensor_real(tensor: Tensor) -> Result<Value, String> {
    if tensor.data.is_empty() {
        return Ok(tensor::tensor_into_value(tensor));
    }

    let mut requires_complex = false;
    let mut real_data = Vec::with_capacity(tensor.data.len());
    let mut complex_data = Vec::with_capacity(tensor.data.len());

    for &x in &tensor.data {
        if x.is_nan() {
            real_data.push(f64::NAN);
            complex_data.push((f64::NAN, 0.0));
            continue;
        }
        if x.is_infinite() && x.is_sign_positive() {
            real_data.push(f64::INFINITY);
            complex_data.push((f64::INFINITY, 0.0));
            continue;
        }
        if x.is_infinite() && x.is_sign_negative() {
            requires_complex = true;
            real_data.push(f64::INFINITY);
            complex_data.push((f64::INFINITY, std::f64::consts::PI));
            continue;
        }
        if x >= 1.0 {
            let val = x.acosh();
            real_data.push(val);
            complex_data.push((val, 0.0));
            continue;
        }

        let result = Complex64::new(x, 0.0).acosh();
        let re = zero_small(result.re);
        let im = zero_small(result.im);
        requires_complex = true;
        real_data.push(re);
        complex_data.push((re, im));
    }

    if requires_complex {
        if complex_data.len() == 1 {
            let (re, im) = complex_data[0];
            Ok(Value::Complex(re, im))
        } else {
            let tensor = ComplexTensor::new(complex_data, tensor.shape.clone())
                .map_err(|e| format!("acosh: {e}"))?;
            Ok(Value::ComplexTensor(tensor))
        }
    } else {
        let tensor =
            Tensor::new(real_data, tensor.shape.clone()).map_err(|e| format!("acosh: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn acosh_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    if ct.data.is_empty() {
        return Ok(Value::ComplexTensor(ct));
    }
    let mut mapped = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let result = Complex64::new(re, im).acosh();
        mapped.push((zero_small(result.re), zero_small(result.im)));
    }
    if mapped.len() == 1 {
        let (re, im) = mapped[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor =
            ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("acosh: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn acosh_complex_scalar(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).acosh();
    Value::Complex(zero_small(result.re), zero_small(result.im))
}

fn acosh_char_array(ca: CharArray) -> Result<Value, String> {
    if ca.data.is_empty() {
        let tensor =
            Tensor::new(Vec::new(), vec![ca.rows, ca.cols]).map_err(|e| format!("acosh: {e}"))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("acosh: {e}"))?;
    acosh_tensor_real(tensor)
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
    use runmat_builtins::{IntValue, LogicalArray};

    #[test]
    fn acosh_scalar_real() {
        let value = Value::Num(1.5);
        let result = acosh_builtin(value).expect("acosh");
        match result {
            Value::Num(v) => assert!((v - 0.9624236501192069).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn acosh_scalar_complex() {
        let result = acosh_builtin(Value::Num(0.5)).expect("acosh");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - 1.0471975511965976).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn acosh_tensor_mixed() {
        let tensor = Tensor::new(vec![0.5, 1.0, 2.0], vec![3, 1]).expect("tensor construction");
        let result = acosh_builtin(Value::Tensor(tensor)).expect("acosh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [
                    (0.0, 1.0471975511965976),
                    (0.0, 0.0),
                    (1.3169578969248166, 0.0),
                ];
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual.0 - exp.0).abs() < 1e-12);
                    assert!((actual.1 - exp.1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn acosh_logical_array_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).expect("logical array");
        let result = acosh_builtin(Value::LogicalArray(logical)).expect("acosh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    (0.0, 0.0),
                    (0.0, std::f64::consts::FRAC_PI_2),
                    (0.0, 0.0),
                    (0.0, std::f64::consts::FRAC_PI_2),
                ];
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual.0 - exp.0).abs() < 1e-12);
                    assert!((actual.1 - exp.1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn acosh_char_array_roundtrip() {
        let chars = CharArray::new("Az".chars().collect(), 1, 2).expect("char array");
        let result = acosh_builtin(Value::CharArray(chars)).expect("acosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> =
                    "Az".chars().map(|ch| (ch as u32 as f64).acosh()).collect();
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<Complex64> = "Az"
                    .chars()
                    .map(|ch| Complex64::new(ch as u32 as f64, 0.0).acosh())
                    .collect();
                for ((re, im), exp) in t.data.iter().zip(expected.iter()) {
                    assert!((re - exp.re).abs() < 1e-12);
                    assert!((im - exp.im).abs() < 1e-12);
                }
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn acosh_char_array_promotes_to_complex() {
        let chars = CharArray::new(vec!['\0'], 1, 1).expect("char array");
        let result = acosh_builtin(Value::CharArray(chars)).expect("acosh");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                let (re, im) = t.data[0];
                assert!(re.abs() < 1e-12);
                assert!((im - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn acosh_complex_inputs() {
        let inputs = [Complex64::new(1.0, 2.0), Complex64::new(-2.0, 0.5)];
        let complex = ComplexTensor::new(inputs.iter().map(|c| (c.re, c.im)).collect(), vec![1, 2])
            .expect("complex tensor");
        let result = acosh_builtin(Value::ComplexTensor(complex)).expect("acosh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (actual, input) in t.data.iter().zip(inputs.iter()) {
                    let expected = input.acosh();
                    assert!((actual.0 - expected.re).abs() < 1e-12);
                    assert!((actual.1 - expected.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn acosh_integer_input() {
        let result = acosh_builtin(Value::Int(IntValue::I32(4))).expect("acosh");
        match result {
            Value::Num(v) => assert!((v - 2.0634370688955608).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[test]
    fn acosh_bool_inputs() {
        let true_result = acosh_builtin(Value::Bool(true)).expect("acosh");
        match true_result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("expected real scalar, got {other:?}"),
        }
        let false_result = acosh_builtin(Value::Bool(false)).expect("acosh");
        match false_result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn acosh_infinity_inputs() {
        let pos = acosh_builtin(Value::Num(f64::INFINITY)).expect("acosh");
        match pos {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_positive()),
            other => panic!("expected positive infinity result, got {other:?}"),
        }

        let neg = acosh_builtin(Value::Num(f64::NEG_INFINITY)).expect("acosh");
        match neg {
            Value::Complex(re, im) => {
                assert!(re.is_infinite() && re.is_sign_positive());
                assert!((im - std::f64::consts::PI).abs() < 1e-12);
            }
            other => panic!("expected complex infinity result, got {other:?}"),
        }
    }

    #[test]
    fn acosh_nan_propagates() {
        let result = acosh_builtin(Value::Num(f64::NAN)).expect("acosh");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN scalar, got {other:?}"),
        }
    }

    #[test]
    fn acosh_string_errors() {
        let err = acosh_builtin(Value::from("oops")).expect_err("expected error");
        assert!(err.contains("expected numeric input"));
    }

    #[test]
    fn acosh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![1.0, 2.0, 5.0, 10.0], vec![4, 1]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = acosh_builtin(Value::GpuTensor(handle)).expect("acosh");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            for (actual, expected) in gathered.data.iter().zip(tensor.data.iter()) {
                let ref_val = expected.acosh();
                assert!((actual - ref_val).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn acosh_gpu_falls_back_for_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.5, 2.0], vec![2, 1]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = acosh_builtin(Value::GpuTensor(handle)).expect("acosh");
            match result {
                Value::ComplexTensor(t) => {
                    assert_eq!(t.shape, vec![2, 1]);
                    let expected = [
                        Complex64::new(0.5, 0.0).acosh(),
                        Complex64::new(2.0, 0.0).acosh(),
                    ];
                    for (actual, exp) in t.data.iter().zip(expected.iter()) {
                        assert!((actual.0 - exp.re).abs() < 1e-12);
                        assert!((actual.1 - exp.im).abs() < 1e-12);
                    }
                }
                other => panic!("expected complex tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn acosh_wgpu_matches_cpu_when_real() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 10.0], vec![3, 1]).unwrap();
        let cpu = acosh_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("provider")
            .upload(&view)
            .expect("upload");
        let gpu = acosh_gpu(handle).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (actual, expected) in gathered.data.iter().zip(ct.data.iter()) {
                    assert!((actual - expected).abs() < tol);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
