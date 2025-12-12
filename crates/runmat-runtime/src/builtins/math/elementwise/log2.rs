//! MATLAB-compatible base-2 logarithm (`log2`) builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise base-2 logarithms for real, logical, character, and complex inputs while
//! preserving MATLAB semantics. Negative real values promote to complex outputs and GPU execution
//! falls back to the host whenever complex numbers are required or the provider lacks a dedicated
//! kernel.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use super::log::{detect_gpu_requires_complex, log_complex_parts};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const IMAG_EPS: f64 = 1e-12;
const LOG2_E: f64 = std::f64::consts::LOG2_E;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "log2",
        builtin_path = "crate::builtins::math::elementwise::log2"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "log2"
category: "math/elementwise"
keywords: ["log2", "base-2 logarithm", "elementwise", "gpu", "complex"]
summary: "Base-2 logarithm of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the provider lacks unary_log2 or when complex results are required."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::log2::tests"
  integration: "builtins::math::elementwise::log2::tests::log2_gpu_provider_roundtrip"
  gpu: "builtins::math::elementwise::log2::tests::log2_wgpu_matches_cpu_elementwise"
---

# What does the `log2` function do in MATLAB / RunMat?
`Y = log2(X)` computes the base-2 logarithm of every element in `X`, following MATLAB's semantics
for real, logical, character, and complex inputs. Negative real values promote to complex results
so that `log2(-1)` yields `0 + i·π/ln(2)`.

## How does the `log2` function behave in MATLAB / RunMat?
- `log2` operates element-wise with MATLAB broadcasting rules.
- Logical inputs convert to doubles (`true → 1.0`, `false → 0.0`) before the logarithm is applied.
- Character arrays are interpreted as their numeric code points and return dense double tensors of the same shape.
- `log2(0)` returns `-Inf`; positive infinity stays `Inf`; `NaN` propagates unchanged.
- Negative real values promote to complex results: `log2([-1 1])` returns `[0 + i·π/ln(2), 0]`.
- Complex inputs follow MATLAB's definition `log2(z) = log(z) / ln(2)` and clamp subnormal imaginary parts to zero for readability.

## `log2` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the GPU when the active provider implements `unary_log2` and
the data is known to remain in the real domain. If complex outputs are required (for example,
negative inputs) or the provider lacks the hook, RunMat gathers the tensor to the host, computes the
exact MATLAB-compatible result, updates residency metadata, and returns the host-resident value.

## Examples of using the `log2` function in MATLAB / RunMat

### Computing base-2 logarithms of powers of two

```matlab
values = [1 2 4 8];
powers = log2(values);
```

Expected output:

```matlab
powers = [0 1 2 3];
```

### Understanding how `log2` handles zero

```matlab
z = log2(0);
```

Expected output:

```matlab
z = -Inf;
```

### Working with negative inputs using complex results

```matlab
neg = [-1 -2];
out = log2(neg);
```

Expected output:

```matlab
out = [0.0000 + 4.5324i  1.0000 + 4.5324i];
```

### Checking power-of-two exponents for matrix sizes

```matlab
A = [64 128; 256 512];
exponents = log2(A);
```

Expected output:

```matlab
exponents = [6 7; 8 9];
```

### Running `log2` on GPU-resident data

```matlab
G = gpuArray([1 4 16 64]);
result = log2(G);
host = gather(result);
```

Expected output:

```matlab
host = [0 2 4 6];
```

### Applying `log2` to character codes

```matlab
C = 'ABC';
values = log2(C);
```

Expected output:

```matlab
values = [6.0224 6.1293 6.1990];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do **not** need to call `gpuArray` yourself. The auto-offload planner keeps tensors on
the GPU when profitable and the result stays real. When complex results are required, RunMat
automatically gathers the data to the host to produce the precise MATLAB-compatible answer. Use
`gpuArray`/`gather` only when you want to mirror MathWorks MATLAB workflows explicitly.

## FAQ

### When should I use `log2` instead of `log` or `log10`?
Use `log2` when you care about binary scalings—such as signal-processing bit widths, FFT sizes, or
exponent analysis. Use `log` (natural logarithm) for exponential growth/decay and `log10` for
decimal magnitudes.

### What happens if an element is zero?
`log2(0)` returns negative infinity (`-Inf`), matching MATLAB behavior.

### How does `log2` handle negative real numbers?
Negative values promote to complex numbers with an imaginary component of `π/ln(2)`. This preserves
phase information instead of producing `NaN`.

### Can I pass complex inputs to `log2`?
Yes. Complex scalars and tensors are handled as `log(z) / ln(2)`, matching MATLAB exactly.

### Does the GPU implementation support complex outputs?
Current providers operate on real buffers. When complex outputs are required, RunMat gathers the
tensor to the host while keeping fusion metadata consistent.

### Is `log2` numerically stable for very small or large values?
Yes. The implementation promotes to 64-bit doubles throughout and clamps tiny imaginary parts to
zero, mirroring MATLAB's behavior for well-conditioned inputs.

## See Also
[log](./log), [log10](./log10), [log1p](./log1p), [exp](./exp), [sqrt](./sqrt), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `log2` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/log2.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/log2.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::log2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log2" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute log2 directly on device buffers; runtimes fall back to the host when complex outputs are required or the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::log2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("log2({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `log2` calls; providers can override with fused kernels when available.",
};

#[runtime_builtin(
    name = "log2",
    category = "math/elementwise",
    summary = "Base-2 logarithm of scalars, vectors, matrices, or N-D tensors.",
    keywords = "log2,base-2 logarithm,elementwise,gpu,complex",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::log2"
)]
fn log2_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => log2_gpu(handle),
        Value::Complex(re, im) => {
            let (r, i) = log2_complex_parts(re, im);
            Ok(Value::Complex(r, i))
        }
        Value::ComplexTensor(ct) => log2_complex_tensor(ct),
        Value::CharArray(ca) => log2_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("log2: expected numeric input".to_string()),
        other => log2_real(other),
    }
}

fn log2_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle) {
            Ok(false) => {
                if let Ok(out) = provider.unary_log2(&handle) {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                return log2_tensor(tensor);
            }
            Err(_) => {
                // Fall through and gather below if detection fails.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    log2_tensor(tensor)
}

fn log2_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("log2", value)?;
    log2_tensor(tensor)
}

fn log2_tensor(tensor: Tensor) -> Result<Value, String> {
    let shape = tensor.shape.clone();
    let len = tensor.data.len();
    let mut complex_values = Vec::with_capacity(len);
    let mut has_imag = false;

    for &v in &tensor.data {
        let (re_part, im_part) = log2_complex_parts(v, 0.0);
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
                ComplexTensor::new(complex_values, shape).map_err(|e| format!("log2: {e}"))?;
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
        let tensor = Tensor::new(data, shape).map_err(|e| format!("log2: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log2_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        data.push(log2_complex_parts(re, im));
    }
    if data.len() == 1 {
        let (re, im) = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor =
            ComplexTensor::new(data, ct.shape.clone()).map_err(|e| format!("log2: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn log2_char_array(ca: CharArray) -> Result<Value, String> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("log2: {e}"))?;
    log2_tensor(tensor)
}

fn log2_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let (real_ln, imag_ln) = log_complex_parts(re, im);
    let mut real_part = real_ln * LOG2_E;
    let mut imag_part = imag_ln * LOG2_E;

    if real_part.is_finite() && real_part.abs() < IMAG_EPS {
        real_part = 0.0;
    }
    if imag_part.abs() < IMAG_EPS {
        imag_part = 0.0;
    }

    (real_part, imag_part)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{LogicalArray, StringArray, Tensor, Value};

    #[test]
    fn log2_scalar_one() {
        let result = log2_builtin(Value::Num(1.0)).expect("log2");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log2_scalar_two() {
        let result = log2_builtin(Value::Num(2.0)).expect("log2");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log2_scalar_zero() {
        let result = log2_builtin(Value::Num(0.0)).expect("log2");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log2_scalar_negative() {
        let result = log2_builtin(Value::Num(-1.0)).expect("log2");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - (std::f64::consts::PI * LOG2_E)).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn log2_bool_true() {
        let result = log2_builtin(Value::Bool(true)).expect("log2");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn log2_logical_array_inputs() {
        let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
        let result = log2_builtin(Value::LogicalArray(logical)).expect("log2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!((t.data[0] - 0.0).abs() < 1e-12);
                assert!(t.data[1].is_infinite() && t.data[1].is_sign_negative());
                assert!((t.data[2] - 0.0).abs() < 1e-12);
                assert!(t.data[3].is_infinite() && t.data[3].is_sign_negative());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn log2_string_input_errors() {
        let err = log2_builtin(Value::from("hello")).unwrap_err();
        assert!(
            err.contains("log2: expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn log2_string_array_errors() {
        let array = StringArray::new(vec!["hello".to_string()], vec![1, 1]).unwrap();
        let err = log2_builtin(Value::StringArray(array)).unwrap_err();
        assert!(
            err.contains("log2: expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn log2_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let result = log2_builtin(Value::Tensor(tensor)).expect("log2");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!((ct.data[0].0 - 0.0).abs() < 1e-12);
                assert!((ct.data[0].1 - (std::f64::consts::PI * LOG2_E)).abs() < 1e-12);
                assert!((ct.data[1].0 - 0.0).abs() < 1e-12);
                assert!((ct.data[1].1).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn log2_complex_scalar() {
        let result = log2_builtin(Value::Complex(1.0, 2.0)).expect("log2");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = log2_complex_parts(1.0, 2.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn log2_char_array_inputs() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = log2_builtin(Value::CharArray(chars)).expect("log2");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - (65.0f64).log2()).abs() < 1e-12);
                assert!((t.data[1] - (90.0f64).log2()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn log2_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log2_builtin(Value::GpuTensor(handle)).expect("log2");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.log2()).collect();
            for (a, b) in gathered.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn log2_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, 2.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log2_builtin(Value::GpuTensor(handle)).expect("log2");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![1, 2]);
                    assert!((ct.data[0].0 - 1.0).abs() < 1e-12);
                    assert!((ct.data[0].1 - (std::f64::consts::PI * LOG2_E)).abs() < 1e-12);
                    assert!((ct.data[1].0 - 1.0).abs() < 1e-12);
                    assert!((ct.data[1].1).abs() < 1e-12);
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
    fn log2_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = log2_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = log2_gpu(handle).expect("log2 gpu");
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
