//! MATLAB-compatible `expm1` builtin with GPU-aware semantics for RunMat.
//!
//! Provides an element-wise `exp(x) - 1` with improved accuracy for tiny magnitudes, covering
//! real, logical, character, and complex inputs. GPU execution uses provider hooks when available
//! and falls back to host computation otherwise, matching MATLAB behaviour.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "expm1",
        builtin_path = "crate::builtins::math::elementwise::expm1"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "expm1"
category: "math/elementwise"
keywords: ["expm1", "exp(x)-1", "exponential", "elementwise", "gpu", "precision"]
summary: "Accurate element-wise computation of exp(x) - 1 for scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the active provider lacks unary_expm1."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::expm1::tests"
  integration: "builtins::math::elementwise::expm1::tests::expm1_gpu_provider_roundtrip"
---

# What does the `expm1` function do in MATLAB / RunMat?
`Y = expm1(X)` evaluates `exp(X) - 1` element-wise while maintaining high accuracy for values of
`X` that are close to zero. It mirrors MATLAB semantics across scalars, vectors, matrices, logical
arrays, character arrays, and complex inputs.

## How does the `expm1` function behave in MATLAB / RunMat?
- Logical inputs are promoted to double precision (`true -> 1.0`, `false -> 0.0`) before execution.
- Character arrays are interpreted as their numeric code points and return dense double tensors.
- Complex values follow MATLAB's definition by computing `exp(z) - 1` using complex arithmetic.
- Existing GPU tensors remain on the device when the registered provider implements
  `unary_expm1`; otherwise RunMat gathers the data, computes on the CPU, and returns the result.

## `expm1` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors resident on the GPU whenever the provider exposes the
`unary_expm1` hook. When the hook is missing or errors, RunMat automatically gathers the tensor,
performs the computation on the host using `f64::expm1` for the real components, and returns the
result while preserving residency metadata. This ensures users always observe MATLAB-compatible
behaviour without manual residency management.

## Examples of using the `expm1` function in MATLAB / RunMat

### Maintaining precision for tiny growth rates

```matlab
x = 1e-12;
y = expm1(x);
```

Expected output:

```matlab
y = 1.0000000000005e-12;
```

### Applying expm1 to model percentage growth

```matlab
rates = [-0.10 -0.05 0 0.05 0.10];
factors = expm1(rates);
```

Expected output:

```matlab
factors = [-0.0952 -0.0488 0 0.0513 0.1052];
```

### Running expm1 on GPU arrays

```matlab
G = gpuArray(linspace(-1, 1, 5));
result = expm1(G);
out = gather(result);
```

Expected behaviour:

```matlab
out = [-0.6321 -0.3935 0 0.6487 1.7183];
```

### Using expm1 with complex numbers

```matlab
z = [1+1i, -1+pi*1i];
w = expm1(z);
```

Expected output:

```matlab
w = [0.4687 + 2.2874i, -1.3679 + 0.0000i];
```

### Applying expm1 to character data

```matlab
C = 'ABC';
Y = expm1(C);
```

Expected output:

```matlab
Y = [1.6949e+28 4.6072e+28 1.2524e+29];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
In most workflows you do **not** need to call `gpuArray` manually. RunMat's auto-offload planner
and fusion engine keep data on the GPU when beneficial. You can still call `gpuArray` to mirror
MathWorks MATLAB workflows or to pin data on the device explicitly.

## FAQ

### When should I prefer `expm1` over `exp(x) - 1`?
Use `expm1` whenever `x` can be very close to zero. It avoids catastrophic cancellation and matches
MATLAB's high-accuracy results for tiny magnitudes.

### Does `expm1` change my tensor's shape?
No. The output has the same shape as the input, subject to MATLAB broadcasting semantics.

### How are logical arrays handled?
Logical values convert to doubles before applying `expm1`, so `expm1([true false])` yields a
double array `[e - 1, 0]`.

### What about complex inputs?
Complex scalars and tensors use MATLAB's complex exponential formula and subtract one from the
result, keeping both real and imaginary parts accurate.

### What happens if the GPU provider lacks `unary_expm1`?
RunMat gathers the tensor to the host, computes with double precision, and returns the correct
result. Subsequent fused kernels still see accurate residency metadata.

### Can I expect double precision?
Yes. RunMat stores dense numeric tensors in double precision (`f64`). GPU providers may choose
single precision internally but convert back to double when returning data to the runtime.

### How does `expm1` interact with fusion?
The fusion planner recognises `expm1` as an element-wise op. Providers that support fused kernels
can materialise `expm1` directly in generated WGSL.

## See Also
[exp](./exp), [log1p](./log1p), [sin](./sin), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `expm1` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/expm1.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/expm1.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::expm1")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "expm1",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_expm1" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may implement expm1 directly; runtimes gather to host when unary_expm1 is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::expm1")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "expm1",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let one = match ctx.scalar_ty {
                ScalarType::F32 => "1.0".to_string(),
                ScalarType::F64 => "f64(1.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!("(exp({input}) - {one})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL exp(x) - 1 sequences; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "expm1",
    category = "math/elementwise",
    summary = "Accurate element-wise computation of exp(x) - 1.",
    keywords = "expm1,exp(x)-1,exponential,elementwise,gpu,precision",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::expm1"
)]
fn expm1_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => expm1_gpu(handle),
        Value::Complex(re, im) => {
            let (real, imag) = expm1_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => expm1_complex_tensor(ct),
        Value::CharArray(ca) => expm1_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err("expm1: expected numeric input".to_string())
        }
        other => expm1_real(other),
    }
}

fn expm1_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_expm1(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    expm1_tensor(tensor).map(tensor::tensor_into_value)
}

fn expm1_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("expm1", value)?;
    expm1_tensor(tensor).map(tensor::tensor_into_value)
}

fn expm1_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.exp_m1()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("expm1: {e}"))
}

fn expm1_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| expm1_complex_parts(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("expm1: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn expm1_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).exp_m1())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("expm1: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn expm1_complex_parts(re: f64, im: f64) -> (f64, f64) {
    let half = 0.5 * im;
    let sin_half = half.sin();
    let cos_half = half.cos();
    let cos_b_minus_one = -2.0 * sin_half * sin_half;
    let sin_b = 2.0 * sin_half * cos_half;
    let expm1_a = re.exp_m1();
    let exp_a = expm1_a + 1.0;
    let real = expm1_a + exp_a * cos_b_minus_one;
    let imag = exp_a * sin_b;
    (real, imag)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[test]
    fn expm1_scalar_zero() {
        let result = expm1_builtin(Value::Num(0.0)).expect("expm1");
        match result {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn expm1_scalar_small_matches_high_precision() {
        let input = 1.0e-16;
        let result = expm1_builtin(Value::Num(input)).expect("expm1");
        match result {
            Value::Num(v) => {
                let naive = input.exp() - 1.0;
                let delta_precise = v - input;
                let delta_naive = naive - input;
                assert!(delta_precise.abs() <= delta_naive.abs());
                assert!(delta_precise.abs() < 1e-28);
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn expm1_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, 1.0, -1.0], vec![3, 1]).unwrap();
        let result = expm1_builtin(Value::Tensor(tensor)).expect("expm1");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [0.0, 1.0f64.exp_m1(), (-1.0f64).exp_m1()];
                for (out, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((out - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn expm1_int_promotes() {
        let result = expm1_builtin(Value::Int(IntValue::I32(1))).expect("expm1");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.exp_m1()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn expm1_complex_scalar() {
        let result = expm1_builtin(Value::Complex(1.0, 1.0)).expect("expm1");
        match result {
            Value::Complex(re, im) => {
                let exp_a = 1.0f64.exp();
                let expected_re = exp_a * 1.0f64.cos() - 1.0;
                let expected_im = exp_a * 1.0f64.sin();
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn expm1_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = expm1_builtin(Value::CharArray(chars)).expect("expm1");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).exp_m1();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn expm1_string_rejects() {
        let err = expm1_builtin(Value::from("not numeric")).expect_err("should fail");
        assert!(
            err.contains("expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn expm1_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = expm1_builtin(Value::GpuTensor(handle)).expect("expm1");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.exp_m1()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (out, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((out - exp).abs() < 1e-12);
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
    fn expm1_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, -0.5, 0.5, 1.0], vec![4, 1]).unwrap();
        let cpu = expm1_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = expm1_gpu(h).unwrap();
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
