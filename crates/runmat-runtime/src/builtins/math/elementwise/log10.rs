//! MATLAB-compatible base-10 logarithm (`log10`) builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise base-10 logarithms for real, logical, character, and complex inputs while
//! preserving MATLAB semantics. Negative real values promote to complex outputs and GPU execution
//! falls back to the host whenever complex numbers are required or the provider lacks a dedicated
//! kernel.

use runmat_accelerate_api::GpuTensorHandle;
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

use super::log::{detect_gpu_requires_complex, log_complex_parts};

const IMAG_EPS: f64 = 1e-12;
const LOG10_E: f64 = std::f64::consts::LOG10_E;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "log10"
category: "math/elementwise"
keywords: ["log10", "base-10 logarithm", "elementwise", "magnitude", "gpu"]
summary: "Base-10 logarithm of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the provider lacks unary_log10 or when the result requires complex values."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::log10::tests"
  integration: "builtins::math::elementwise::log10::tests::log10_gpu_provider_roundtrip"
  gpu: "builtins::math::elementwise::log10::tests::log10_wgpu_matches_cpu_elementwise"
---

# What does the `log10` function do in MATLAB / RunMat?
`Y = log10(X)` computes the base-10 logarithm of every element in `X`, following MATLAB's
numeric semantics for real, logical, character, and complex inputs. Negative real values promote to
complex results so that you can analyze magnitudes without losing phase information.

## How does the `log10` function behave in MATLAB / RunMat?
- `log10` operates element-wise with MATLAB broadcasting rules.
- Logical inputs convert to doubles (`true → 1.0`, `false → 0.0`) before the logarithm is applied.
- Character arrays are interpreted as their numeric code points and return dense double tensors.
- `log10(0)` returns `-Inf`, matching MATLAB's handling of the logarithm singularity at zero.
- Negative real values promote to complex results: `log10(-10)` returns `1 + i·π/ln(10)`.
- Complex inputs follow MATLAB's definition: `log10(z) = log(z) / ln(10)`.

## `log10` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the GPU when the active provider implements `unary_log10` and
the data is known to stay in the real domain. If complex outputs are required (for example, negative
inputs) or the provider lacks the hook, RunMat gathers the tensor to the host, computes the exact
MATLAB-compatible result, updates residency metadata, and returns the host-resident value.

## Examples of using the `log10` function in MATLAB / RunMat

### Finding the order of magnitude of a number

```matlab
value = log10(1000);
```

Expected output:

```matlab
value = 3;
```

### Computing base-10 logarithms of a matrix

```matlab
A = [1 10 100; 0.1 0.01 0.001];
B = log10(A);
```

Expected output:

```matlab
B = [0 1 2; -1 -2 -3];
```

### Understanding how `log10` handles zero

```matlab
z = log10(0);
```

Expected output:

```matlab
z = -Inf;
```

### Working with negative inputs using complex results

```matlab
neg = [-10 -100];
out = log10(neg);
```

Expected output:

```matlab
out = [1.0000 + 1.3644i, 2.0000 + 1.3644i];
```

### Running `log10` on GPU-resident data

```matlab
G = gpuArray([1 10 1000]);
result = log10(G);
host = gather(result);
```

Expected output:

```matlab
host = [0 1 3];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do **not** need to call `gpuArray` yourself. The auto-offload planner keeps tensors on
the GPU when profitable and the result stays real. When complex results are required, RunMat
automatically gathers the data to the host to produce the precise MATLAB-compatible answer. Use
`gpuArray`/`gather` only when you want to mirror MathWorks MATLAB workflows explicitly.

## FAQ

### When should I use `log10` instead of `log`?
Use `log10` when you want base-10 magnitudes, such as for decibel calculations or scientific
notation. Use `log` (natural logarithm) for exponential growth/decay or calculus contexts.

### What happens if an element is zero?
`log10(0)` returns negative infinity (`-Inf`), matching MATLAB behavior.

### How does `log10` handle negative real numbers?
Negative values promote to complex numbers with an imaginary component of `π/ln(10)`. This preserves
phase information instead of producing `NaN`.

### Can I pass complex inputs to `log10`?
Yes. Complex scalars and tensors are handled as `log(z) / ln(10)`, matching MATLAB exactly.

### Does the GPU implementation support complex outputs?
Current providers operate on real buffers. When complex outputs are required, RunMat gathers the
tensor to the host while keeping fusion metadata consistent.

### Is `log10` numerically stable for very small or large values?
Yes. The implementation promotes to 64-bit doubles throughout and clamps tiny imaginary parts to
zero, mirroring MATLAB's behavior for well-conditioned inputs.

## See Also
[log](./log), [log1p](./log1p), [exp](./exp), [sqrt](./sqrt), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `log10` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/log10.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/log10.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log10",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log10" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute log10 directly on device buffers; runtimes fall back to the host when complex outputs are required or the hook is unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log10",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.get(0).ok_or(FusionError::MissingInput(0))?;
            let expr = match ctx.scalar_ty {
                ScalarType::F64 => {
                    format!("log({input}) * f64({})", std::f64::consts::LOG10_E)
                }
                ScalarType::F32 => format!(
                    "log({input}) * {:.10}",
                    std::f32::consts::LOG10_E
                ),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(expr)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `log` multiplied by log10(e); providers can override with fused kernels when available.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("log10", DOC_MD);

#[runtime_builtin(
    name = "log10",
    category = "math/elementwise",
    summary = "Base-10 logarithm of scalars, vectors, matrices, or N-D tensors.",
    keywords = "log10,base-10 logarithm,elementwise,magnitude,gpu",
    accel = "unary"
)]
fn log10_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => log10_gpu(handle),
        Value::Complex(re, im) => {
            let (r, i) = log10_complex_parts(re, im);
            Ok(Value::Complex(r, i))
        }
        Value::ComplexTensor(ct) => log10_complex_tensor(ct),
        Value::CharArray(ca) => log10_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err("log10: expected numeric input".to_string())
        }
        other => log10_real(other),
    }
}

fn log10_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle) {
            Ok(false) => {
                if let Ok(out) = provider.unary_log10(&handle) {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                return log10_tensor(tensor);
            }
            Err(_) => {
                // Fall through and gather below if detection fails.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    log10_tensor(tensor)
}

fn log10_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("log10", value)?;
    log10_tensor(tensor)
}

fn log10_tensor(tensor: Tensor) -> Result<Value, String> {
    let shape = tensor.shape.clone();
    let len = tensor.data.len();
    let mut complex_values = Vec::with_capacity(len);
    let mut has_imag = false;

    for &v in &tensor.data {
        let (re_part, im_part) = log10_complex_parts(v, 0.0);
        if im_part != 0.0 {
            has_imag = true;
        }
        complex_values.push((re_part, im_part));
    }

    if has_imag {
        if len == 1 {
            let (re, im) = complex_values[0];
            Ok(Value::Complex(re, im))
        } else {
            let tensor =
                ComplexTensor::new(complex_values, shape).map_err(|e| format!("log10: {e}"))?;
            Ok(Value::ComplexTensor(tensor))
        }
    } else {
        let data: Vec<f64> = complex_values
            .into_iter()
            .map(|(mut re, _)| {
                if re.is_finite() && re.abs() < IMAG_EPS {
                    re = 0.0;
                }
                re
            })
            .collect();
        let tensor = Tensor::new(data, shape).map_err(|e| format!("log10: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log10_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        data.push(log10_complex_parts(re, im));
    }
    if data.len() == 1 {
        let (re, im) = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor =
            ComplexTensor::new(data, ct.shape.clone()).map_err(|e| format!("log10: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn log10_char_array(ca: CharArray) -> Result<Value, String> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("log10: {e}"))?;
    log10_tensor(tensor)
}

fn log10_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let (real_ln, imag_ln) = log_complex_parts(re, im);
    let mut real_part = real_ln * LOG10_E;
    let mut imag_part = imag_ln * LOG10_E;

    if real_part.is_finite() && real_part.abs() < IMAG_EPS {
        real_part = 0.0;
    }
    if imag_part.abs() < IMAG_EPS {
        imag_part = 0.0;
    }

    (real_part, imag_part)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray, StringArray, Tensor, Value};

    #[test]
    fn log10_scalar_one() {
        let result = log10_builtin(Value::Num(1.0)).expect("log10");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log10_scalar_ten() {
        let result = log10_builtin(Value::Num(10.0)).expect("log10");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log10_scalar_zero() {
        let result = log10_builtin(Value::Num(0.0)).expect("log10");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log10_scalar_negative() {
        let result = log10_builtin(Value::Num(-10.0)).expect("log10");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 1.0).abs() < 1e-12);
                let expected_im = std::f64::consts::PI * LOG10_E;
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn log10_bool_true() {
        let result = log10_builtin(Value::Bool(true)).expect("log10");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log10_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-10.0, 10.0], vec![1, 2]).unwrap();
        let result = log10_builtin(Value::Tensor(tensor)).expect("log10");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!((ct.data[0].0 - 1.0).abs() < 1e-12);
                let expected_im = std::f64::consts::PI * LOG10_E;
                assert!((ct.data[0].1 - expected_im).abs() < 1e-12);
                assert!((ct.data[1].0 - 1.0).abs() < 1e-12);
                assert!((ct.data[1].1).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn log10_complex_scalar() {
        let result = log10_builtin(Value::Complex(1.0, 2.0)).expect("log10");
        match result {
            Value::Complex(re, im) => {
                let (ln_re, ln_im) = log_complex_parts(1.0, 2.0);
                assert!((re - ln_re * LOG10_E).abs() < 1e-12);
                assert!((im - ln_im * LOG10_E).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn log10_logical_array_inputs() {
        let logical = LogicalArray::new(vec![1u8, 0u8], vec![2, 1]).expect("logical");
        let result = log10_builtin(Value::LogicalArray(logical)).expect("log10");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 0.0).abs() < 1e-12);
                assert!(t.data[1].is_infinite() && t.data[1].is_sign_negative());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn log10_char_array_inputs() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = log10_builtin(Value::CharArray(chars)).expect("log10");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - (65.0f64).log10()).abs() < 1e-12);
                assert!((t.data[1] - (90.0f64).log10()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn log10_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 10.0, 1000.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log10_builtin(Value::GpuTensor(handle)).expect("log10");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.log10()).collect();
            for (a, b) in gathered.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn log10_string_input_errors() {
        let err = log10_builtin(Value::from("hello"));
        assert!(matches!(err, Err(msg) if msg.contains("expected numeric input")));
    }

    #[test]
    fn log10_string_array_errors() {
        let array = StringArray::new(vec!["hello".to_string()], vec![1, 1]).unwrap();
        let err = log10_builtin(Value::StringArray(array));
        assert!(matches!(err, Err(msg) if msg.contains("expected numeric input")));
    }

    #[test]
    fn log10_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-10.0, 10.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log10_builtin(Value::GpuTensor(handle)).expect("log10");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![1, 2]);
                    let expected_im = std::f64::consts::PI * LOG10_E;
                    assert!((ct.data[0].1 - expected_im).abs() < 1e-12);
                }
                other => panic!("expected complex tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn log10_with_integer_argument() {
        let result = log10_builtin(Value::Int(IntValue::I32(100))).expect("log10");
        match result {
            Value::Num(v) => assert!((v - 2.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn log10_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 10.0, 1000.0, 0.1], vec![4, 1]).unwrap();
        let cpu = log10_real(Value::Tensor(tensor.clone())).expect("cpu log10");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_value = log10_gpu(handle).expect("gpu log10");
        let gathered = test_support::gather(gpu_value).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                for (gpu, cpu) in gathered.data.iter().zip(ct.data.iter()) {
                    let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                    };
                    assert!((gpu - cpu).abs() < tol, "|{gpu} - {cpu}| >= {tol}");
                }
            }
            _ => panic!("unexpected cpu result"),
        }
    }
}
