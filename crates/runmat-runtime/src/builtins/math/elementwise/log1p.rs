//! MATLAB-compatible `log1p` builtin with GPU-aware semantics for RunMat.
//!
//! Provides an element-wise `log(1 + x)` with improved accuracy for small magnitudes, covering
//! real, logical, character, and complex inputs. GPU execution uses provider hooks when available
//! and falls back to host computation whenever complex results are required or device support is
//! missing, mirroring MATLAB behavior.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
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

const IMAG_EPS: f64 = 1e-12;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "log1p"
category: "math/elementwise"
keywords: ["log1p", "log(1+x)", "natural logarithm", "elementwise", "gpu", "precision"]
summary: "Accurate element-wise computation of log(1 + x) for scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host when the provider lacks unary_log1p or reduce_min, or when complex outputs are required."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::log1p::tests"
  integration: "builtins::math::elementwise::log1p::tests::log1p_gpu_provider_roundtrip"
---

# What does the `log1p` function do in MATLAB / RunMat?
`Y = log1p(X)` evaluates `log(1 + X)` element-wise with high accuracy for values of `X`
close to zero. It mirrors MATLAB semantics across scalars, vectors, matrices, logical
arrays, character arrays, and complex inputs.

## How does the `log1p` function behave in MATLAB / RunMat?
- Logical inputs are promoted to double precision (`true -> 1.0`, `false -> 0.0`) before execution.
- Character arrays are interpreted as their numeric code points and return dense double tensors.
- Values equal to `-1` yield `-Inf`, matching MATLAB's handling of `log(0)`.
- Inputs smaller than `-1` promote to complex outputs: `log1p(-2)` returns `0 + iπ`.
- Complex inputs follow MATLAB's definition by computing the natural logarithm of `1 + z`.
- Existing GPU tensors remain on the device when the registered provider implements `unary_log1p`
  alongside `reduce_min`. RunMat queries the device-side minimum to confirm the data stays within
  the real-valued domain; otherwise it gathers to the host, computes the exact result, and preserves
  residency metadata.

## `log1p` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors resident on the GPU whenever the provider exposes the `unary_log1p`
hook together with `reduce_min`. The runtime uses the device-side minimum to ensure that
`1 + X` stays non-negative; when complex outputs are required or either hook is missing, RunMat
automatically gathers the tensor, computes on the CPU using double precision, and returns the
result with the expected MATLAB semantics.

## Examples of using the `log1p` function in MATLAB / RunMat

### Protecting precision when adding tiny percentages

```matlab
delta = 1e-12;
value = log1p(delta);
```

Expected output:

```matlab
value = 9.999999999995e-13;
```

### Computing log-growth factors from percentage changes

```matlab
rates = [-0.25 -0.10 0 0.10 0.25];
growth = log1p(rates);
```

Expected output:

```matlab
growth = [-0.2877 -0.1054 0 0.0953 0.2231];
```

### Handling the branch cut at x = -1

```matlab
y = log1p(-1);
```

Expected output:

```matlab
y = -Inf;
```

### Obtaining complex results for inputs less than -1

```matlab
data = [-2 -3 -5];
result = log1p(data);
```

Expected output:

```matlab
result = [0.0000 + 3.1416i, 0.6931 + 3.1416i, 1.3863 + 3.1416i];
```

### Executing log1p on GPU arrays with automatic residency

```matlab
G = gpuArray(linspace(-0.5, 0.5, 5));
out = log1p(G);
realResult = gather(out);
```

Expected output:

```matlab
realResult = [-0.6931 -0.2877 0 0.2231 0.4055];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
In most workflows you do **not** need to call `gpuArray` manually. RunMat's auto-offload planner
and fusion engine keep data on the GPU when beneficial. When your expression requires complex
results (e.g., values smaller than `-1`), RunMat gathers data to the host automatically and still
returns the MATLAB-compatible output. You can call `gpuArray` and `gather` explicitly if you wish
to mirror MathWorks MATLAB workflows.

## FAQ

### When should I prefer `log1p` over `log(1 + x)`?
Use `log1p` whenever `x` can be very close to zero. It avoids catastrophic cancellation and matches
MATLAB's high-accuracy results for tiny magnitudes.

### Does `log1p` change my tensor's shape?
No. The output has the same shape as the input, subject to MATLAB broadcasting semantics.

### How are logical arrays handled?
Logical values convert to doubles before applying `log1p`, so `log1p([true false])` yields a double
array `[log(2), 0]`.

### What about inputs smaller than `-1`?
Values less than `-1` promote to complex results (`log(1 + x)` on the complex branch), matching
MATLAB's behavior.

### How does `log1p` interact with complex numbers?
Complex scalars and tensors compute `log(1 + z)` using the principal branch, returning both real and
imaginary parts just like MATLAB.

### What happens when the GPU provider lacks `unary_log1p`?
RunMat gathers the tensor to the host, computes the result in double precision, and returns it. This
ensures users always see MATLAB-compatible behavior without manual residency management.

### Is double precision guaranteed?
Yes. RunMat stores dense numeric tensors in double precision (`f64`). GPU providers may choose
single precision internally but convert back to double when returning data to the runtime.

### Can `log1p` participate in fusion?
Yes. The fusion planner recognizes `log1p` as an element-wise op. Providers that support fused
kernels can materialize `log(1 + x)` directly in generated WGSL.

## See Also
[log](./log), [expm1](./expm1), [sin](../trigonometry/sin), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `log1p` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/log1p.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/log1p.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log1p",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log1p" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers should supply unary_log1p and reduce_min; runtimes gather to host when complex outputs are required or either hook is unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log1p",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.get(0).ok_or(FusionError::MissingInput(0))?;
            let one = match ctx.scalar_ty {
                ScalarType::F32 => "1.0".to_string(),
                ScalarType::F64 => "f64(1.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!("log({input} + {one})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `log(x + 1)` sequences; providers may substitute fused kernels when available.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("log1p", DOC_MD);

#[runtime_builtin(
    name = "log1p",
    category = "math/elementwise",
    summary = "Accurate element-wise computation of log(1 + x).",
    keywords = "log1p,log(1+x),natural logarithm,elementwise,gpu,precision",
    accel = "unary"
)]
fn log1p_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => log1p_gpu(handle),
        Value::Complex(re, im) => {
            let (real, imag) = log1p_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => log1p_complex_tensor(ct),
        Value::CharArray(ca) => log1p_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err("log1p: expected numeric input".to_string())
        }
        other => log1p_real(other),
    }
}

fn log1p_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        // Fast path: try device op first; if unsupported, fall back to complex-domain check
        if let Ok(out) = provider.unary_log1p(&handle) {
            return Ok(Value::GpuTensor(out));
        }
        match detect_gpu_requires_complex(provider, &handle) {
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                return log1p_tensor(tensor);
            }
            Ok(false) => {}
            Err(_) => {}
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    log1p_tensor(tensor)
}

fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<bool, String> {
    let min_handle = provider
        .reduce_min(handle)
        .map_err(|e| format!("log1p: reduce_min failed: {e}"))?;
    let download = provider
        .download(&min_handle)
        .map_err(|e| format!("log1p: reduce_min download failed: {e}"));
    let _ = provider.free(&min_handle);
    let host = download?;
    if host.data.iter().any(|&v| v.is_nan()) {
        return Err("log1p: reduce_min result contained NaN".to_string());
    }
    Ok(host.data.iter().any(|&v| v < -1.0))
}

fn log1p_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("log1p", value)?;
    log1p_tensor(tensor)
}

fn log1p_tensor(tensor: Tensor) -> Result<Value, String> {
    let shape = tensor.shape.clone();
    let mut entries = Vec::with_capacity(tensor.data.len());
    let mut has_imag = false;

    for &v in &tensor.data {
        let sum = 1.0 + v;
        if sum.is_nan() {
            entries.push((f64::NAN, 0.0));
            continue;
        }
        if sum < 0.0 {
            let (mut real_part, mut imag_part) = log1p_complex_parts(v, 0.0);
            if real_part.abs() < IMAG_EPS {
                real_part = 0.0;
            }
            if imag_part.abs() < IMAG_EPS {
                imag_part = 0.0;
            }
            if imag_part != 0.0 {
                has_imag = true;
            }
            entries.push((real_part, imag_part));
        } else {
            entries.push((v.ln_1p(), 0.0));
        }
    }

    if has_imag {
        if entries.len() == 1 {
            let (re, im) = entries[0];
            Ok(Value::Complex(re, im))
        } else {
            let tensor = ComplexTensor::new(entries, shape).map_err(|e| format!("log1p: {e}"))?;
            Ok(Value::ComplexTensor(tensor))
        }
    } else {
        let data: Vec<f64> = entries.into_iter().map(|(re, _)| re).collect();
        let tensor = Tensor::new(data, shape).map_err(|e| format!("log1p: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log1p_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let (mut real_part, mut imag_part) = log1p_complex_parts(re, im);
        if real_part.abs() < IMAG_EPS {
            real_part = 0.0;
        }
        if imag_part.abs() < IMAG_EPS {
            imag_part = 0.0;
        }
        data.push((real_part, imag_part));
    }
    if data.len() == 1 {
        let (re, im) = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor =
            ComplexTensor::new(data, ct.shape.clone()).map_err(|e| format!("log1p: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn log1p_char_array(ca: CharArray) -> Result<Value, String> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("log1p: {e}"))?;
    log1p_tensor(tensor)
}

fn log1p_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let shifted_re = re + 1.0;
    let magnitude = shifted_re.hypot(im);
    if magnitude == 0.0 {
        (f64::NEG_INFINITY, 0.0)
    } else {
        let real_part = magnitude.ln();
        let imag_part = im.atan2(shifted_re);
        (real_part, imag_part)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{LogicalArray, Tensor};
    use std::f64::consts::PI;

    #[test]
    fn log1p_scalar_zero() {
        let result = log1p_builtin(Value::Num(0.0)).expect("log1p");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log1p_scalar_negative_one() {
        let result = log1p_builtin(Value::Num(-1.0)).expect("log1p");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log1p_scalar_less_than_negative_one_complex() {
        let result = log1p_builtin(Value::Num(-2.0)).expect("log1p");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - PI).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn log1p_tensor_mixed_values() {
        let tensor = Tensor::new(vec![0.0, -0.5, -2.0, 3.0], vec![2, 2]).unwrap();
        let result = log1p_builtin(Value::Tensor(tensor)).expect("log1p");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
                let expected = vec![
                    (0.0, 0.0),
                    ((0.5f64).ln(), 0.0),
                    (0.0, PI),
                    ((4.0f64).ln(), 0.0),
                ];
                for ((re, im), (er, ei)) in ct.data.iter().zip(expected.iter()) {
                    assert!((re - er).abs() < 1e-12);
                    assert!((im - ei).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn log1p_complex_input() {
        let result = log1p_builtin(Value::Complex(0.5, 1.0)).expect("log1p");
        match result {
            Value::Complex(re, im) => {
                let expected = (1.5f64.hypot(1.0).ln(), 1.0f64.atan2(1.5));
                assert!((re - expected.0).abs() < 1e-12);
                assert!((im - expected.1).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn log1p_char_array_roundtrip() {
        let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
        let result = log1p_builtin(Value::CharArray(chars)).expect("log1p");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['A', 'B', 'C'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).ln_1p();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn log1p_string_rejects() {
        let err = log1p_builtin(Value::from("not numeric")).expect_err("should fail");
        assert!(
            err.contains("expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn log1p_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, -0.25, 0.5, 2.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log1p_builtin(Value::GpuTensor(handle)).expect("log1p");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.ln_1p()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (out, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((out - exp).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn log1p_bool_promotes() {
        let result = log1p_builtin(Value::Bool(true)).expect("log1p");
        match result {
            Value::Num(v) => assert!((v - 2.0f64.ln()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log1p_logical_array_converts() {
        let logical = LogicalArray::new(vec![0, 1], vec![2, 1]).unwrap();
        let result = log1p_builtin(Value::LogicalArray(logical)).expect("log1p");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 0.0).abs() < 1e-12);
                assert!((t.data[1] - 2.0f64.ln()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn log1p_gpu_complex_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, -3.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log1p_builtin(Value::GpuTensor(handle)).expect("log1p");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![2, 1]);
                    let expected = vec![(0.0, PI), ((2.0f64).ln(), PI)];
                    for ((re, im), (er, ei)) in ct.data.iter().zip(expected.iter()) {
                        assert!((re - er).abs() < 1e-12);
                        assert!((im - ei).abs() < 1e-12);
                    }
                }
                other => panic!("expected complex tensor result, got {other:?}"),
            }
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
    fn log1p_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, -0.25, 0.25, 1.0], vec![4, 1]).unwrap();
        let cpu = log1p_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = log1p_gpu(handle).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected value kinds"),
        }
    }
}
