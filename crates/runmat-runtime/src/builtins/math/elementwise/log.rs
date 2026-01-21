//! MATLAB-compatible natural logarithm (`log`) builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise natural logarithms for real, logical, character, and complex inputs while
//! preserving MATLAB semantics, including promotion of negative real values to complex outputs.
//! GPU execution uses provider hooks when available and falls back to host computation whenever
//! complex results are required or the provider lacks a dedicated kernel.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const IMAG_EPS: f64 = 1e-12;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "log",
        builtin_path = "crate::builtins::math::elementwise::log"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "log"
category: "math/elementwise"
keywords: ["log", "natural logarithm", "elementwise", "gpu", "complex"]
summary: "Natural logarithm of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the provider lacks unary_log or when the result requires complex values."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::log::tests"
  integration: "builtins::math::elementwise::log::tests::log_gpu_provider_roundtrip"
  gpu: "builtins::math::elementwise::log::tests::log_wgpu_matches_cpu_elementwise"
---

# What does the `log` function do in MATLAB / RunMat?
`Y = log(X)` computes the natural logarithm of every element in `X`, extending MATLAB semantics
to real, logical, character, and complex inputs. Negative real values are promoted to complex
results so that `log(-1)` yields `0 + iπ`.

## How does the `log` function behave in MATLAB / RunMat?
- `log(X)` applies the operation element-wise with MATLAB broadcasting rules.
- Logical values convert to doubles (`true → 1.0`, `false → 0.0`) before the logarithm is taken.
- Character arrays are interpreted as their numeric code points and return dense double tensors.
- Negative real values produce complex results: `log([-1 1])` returns `[0 + iπ, 0]`.
- Complex inputs follow MATLAB's definition: `log(a + bi) = log(|a + bi|) + i·atan2(b, a)`.
- `log(0)` returns `-Inf`, matching MATLAB's handling of the logarithm singularity at zero.

## `log` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the GPU when the active provider implements `unary_log`
*and* `reduce_min`. The runtime queries the device-side minimum to confirm that all values are
non-negative, so positive datasets stay resident and can participate in fused elementwise kernels.
When complex outputs are required or the provider cannot supply these hooks, RunMat gathers the
tensor to the host, computes the exact MATLAB-compatible result (including complex promotion),
updates residency metadata, and returns the host-resident value.

## Examples of using the `log` function in MATLAB / RunMat

### Natural log of a positive scalar

```matlab
y = log(exp(3));
```

Expected output:

```matlab
y = 3;
```

### Understanding log of zero

```matlab
value = log(0);
```

Expected output:

```matlab
value = -Inf;
```

### Taking the logarithm of negative values

```matlab
data = [-1 -2 -4];
result = log(data);
```

Expected output:

```matlab
result = [0.0000 + 3.1416i, 0.6931 + 3.1416i, 1.3863 + 3.1416i];
```

### Applying log to complex numbers

```matlab
z = [1+2i, -1+pi*i];
w = log(z);
```

Expected output:

```matlab
w = [0.8047 + 1.1071i, 1.1447 + 1.2626i];
```

### Element-wise log on a matrix living on the GPU

```matlab
G = gpuArray([1 2; 4 8]);
out = log(G);
result = gather(out);
```

Expected output:

```matlab
result = [0.0000 0.6931; 1.3863 2.0794];
```

### Logging character codes from a string

```matlab
C = 'ABC';
values = log(C);
```

Expected output:

```matlab
values = [4.1744 4.1897 4.2047];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do **not** need to call `gpuArray` yourself. The auto-offload planner tracks
residency and keeps tensors on the GPU when profitable. When your expression produces complex
results (e.g., negative inputs to `log`), RunMat will gather the data automatically and still
return exactly the MATLAB-compatible output. You can force explicit residency with `gpuArray`
and `gather` if you want to mirror MathWorks MATLAB workflows.

## FAQ

### When should I use the `log` function?
Use `log` whenever you need the natural logarithm of your data—for example, to linearize
exponential growth, compute likelihoods, or transform multiplicative relationships into additive
ones.

### What happens if the input contains zeros?
`log(0)` returns negative infinity (`-Inf`). Entire tensors follow the same rule element-wise.

### How are negative real numbers handled?
Negative values automatically promote to complex results: `log(-x)` returns `log(x) + iπ`.
This matches MATLAB behaviour and avoids losing information compared with returning `NaN`.

### What about tiny floating-point noise producing small negative numbers?
Values that are numerically negative (e.g., `-1e-15`) are treated just like other negatives and
promote to complex outputs. Use `abs` or `max` to clip values if you require a purely real result.

### Does the GPU implementation support complex outputs?
Providers currently operate on real buffers. When complex results are required, RunMat gathers
data to the host to compute the exact result and keeps residency metadata consistent.

### Does `log` accept complex inputs directly?
Yes. Complex scalars and tensors follow the MATLAB definition using magnitude and phase
(`log(|z|) + i·angle(z)`).

## See Also
[exp](./exp), [expm1](./expm1), [abs](./abs), [angle](./angle), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `log` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/log.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/log.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::log")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "log",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_log" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute log directly on device buffers; runtimes gather to host when complex outputs are required or the hook is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::log")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "log",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("log({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `log` calls; providers can override with fused kernels when available.",
};

const BUILTIN_NAME: &str = "log";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "log",
    category = "math/elementwise",
    summary = "Natural logarithm of scalars, vectors, matrices, or N-D tensors.",
    keywords = "log,natural logarithm,elementwise,gpu,complex",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::log"
)]
async fn log_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => log_gpu(handle).await,
        Value::Complex(re, im) => {
            let (r, i) = log_complex_parts(re, im);
            Ok(Value::Complex(r, i))
        }
        Value::ComplexTensor(ct) => log_complex_tensor(ct),
        Value::CharArray(ca) => log_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("log: expected numeric input"))
        }
        other => log_real(other),
    }
}

async fn log_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle) {
            Ok(false) => {
                if let Ok(out) = provider.unary_log(&handle) {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
                return log_tensor_real(tensor);
            }
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(err);
                }
                // Fall through to host fallback below.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    log_tensor_real(tensor)
}

pub(super) fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<bool> {
    let min_handle = provider
        .reduce_min(handle)
        .map_err(|e| builtin_error(format!("log: reduce_min failed: {e}")))?;
    let download = provider
        .download(&min_handle)
        .map_err(|e| builtin_error(format!("log: reduce_min download failed: {e}")));
    let _ = provider.free(&min_handle);
    let host = download?;
    if host.data.iter().any(|&v| v.is_nan()) {
        return Err(builtin_error("log: reduce_min result contained NaN"));
    }
    Ok(host.data.iter().any(|&v| v.is_sign_negative()))
}

fn log_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("log", value)
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    log_tensor_real(tensor)
}

fn log_tensor_real(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let len = tensor.data.len();
    let mut complex_values = Vec::with_capacity(len);
    let mut has_imag = false;

    for &v in &tensor.data {
        let (mut real_part, mut imag_part) = log_complex_parts(v, 0.0);
        if real_part.is_finite() && real_part.abs() < IMAG_EPS {
            real_part = 0.0;
        }
        if !imag_part.is_finite() || imag_part.abs() < IMAG_EPS {
            imag_part = 0.0;
        }
        if imag_part != 0.0 {
            has_imag = true;
        }
        complex_values.push((real_part, imag_part));
    }

    if has_imag {
        if len == 1 {
            let (re, im) = complex_values[0];
            Ok(Value::Complex(re, im))
        } else {
            let tensor = ComplexTensor::new(complex_values, shape)
                .map_err(|e| builtin_error(format!("log: {e}")))?;
            Ok(Value::ComplexTensor(tensor))
        }
    } else {
        let data: Vec<f64> = complex_values.into_iter().map(|(re, _)| re).collect();
        let tensor = Tensor::new(data, shape).map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

fn log_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let (mut real_part, mut imag_part) = log_complex_parts(re, im);
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
        let tensor = ComplexTensor::new(data, ct.shape.clone())
            .map_err(|e| builtin_error(format!("log: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn log_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("log: {e}")))?;
    log_tensor_real(tensor)
}

pub(super) fn log_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let magnitude = re.hypot(im);
    if magnitude == 0.0 {
        (f64::NEG_INFINITY, 0.0)
    } else {
        let real_part = magnitude.ln();
        let imag_part = im.atan2(re);
        (real_part, imag_part)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, Tensor, Value};

    fn log_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::log_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_one() {
        let result = log_builtin(Value::Num(1.0)).expect("log");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_zero() {
        let result = log_builtin(Value::Num(0.0)).expect("log");
        match result {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_negative()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_negative() {
        let result = log_builtin(Value::Num(-1.0)).expect("log");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 0.0).abs() < 1e-12);
                assert!((im - std::f64::consts::PI).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_scalar_nan_remains_real() {
        let result = log_builtin(Value::Num(f64::NAN)).expect("log");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected real NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_bool_true() {
        let result = log_builtin(Value::Bool(true)).expect("log");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_logical_array_inputs() {
        let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
        let result = log_builtin(Value::LogicalArray(logical)).expect("log");
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_string_input_errors() {
        let err = log_builtin(Value::from("hello")).unwrap_err();
        assert!(
            err.message().contains("log: expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
        let result = log_builtin(Value::Tensor(tensor)).expect("log");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!((ct.data[0].0 - 0.0).abs() < 1e-12);
                assert!((ct.data[0].1 - std::f64::consts::PI).abs() < 1e-12);
                assert!((ct.data[1].0 - 0.0).abs() < 1e-12);
                assert!((ct.data[1].1).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_complex_scalar() {
        let result = log_builtin(Value::Complex(1.0, 2.0)).expect("log");
        match result {
            Value::Complex(re, im) => {
                let expected_re = (1.0_f64.hypot(2.0)).ln();
                let expected_im = 2.0_f64.atan2(1.0);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_char_array_inputs() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = log_builtin(Value::CharArray(chars)).expect("log");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - (65.0f64).ln()).abs() < 1e-12);
                assert!((t.data[1] - (90.0f64).ln()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log_builtin(Value::GpuTensor(handle)).expect("log");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.ln()).collect();
            for (a, b) in gathered.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 1.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = log_builtin(Value::GpuTensor(handle)).expect("log");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![1, 2]);
                    assert!((ct.data[0].0 - 0.0).abs() < 1e-12);
                    assert!((ct.data[0].1 - std::f64::consts::PI).abs() < 1e-12);
                }
                other => panic!("expected complex tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn log_with_integer_argument() {
        let result = log_builtin(Value::Int(IntValue::I32(4))).expect("log");
        match result {
            Value::Num(v) => assert!((v - (4.0f64).ln()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
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
    fn log_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = log_real(Value::Tensor(tensor.clone())).expect("cpu log");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_value = log_gpu(handle).expect("gpu log");
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
            Value::Num(_) => panic!("expected tensor result from cpu path"),
            other => panic!("unexpected cpu result {other:?}"),
        }
    }
}
