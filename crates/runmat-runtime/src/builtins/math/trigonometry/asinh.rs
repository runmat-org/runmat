//! MATLAB-compatible `asinh` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise inverse hyperbolic sine for scalars, tensors, and complex inputs while
//! matching MATLAB's promotion and residency rules.

use num_complex::Complex64;
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

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "asinh"
category: "math/trigonometry"
keywords: ["asinh", "inverse hyperbolic sine", "arcsinh", "trigonometry", "gpu"]
summary: "Element-wise inverse hyperbolic sine with MATLAB-compatible complex promotion and GPU fallbacks."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses the provider's unary_asinh hook when available; gathers to host otherwise."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::asinh::tests"
  integration: "builtins::math::trigonometry::asinh::tests::asinh_gpu_provider_roundtrip"
---

# What does the `asinh` function do in MATLAB / RunMat?
`Y = asinh(X)` evaluates the inverse hyperbolic sine of every element in `X`. Results match MATLAB
semantics for real, logical, character, and complex inputs, including branch cut handling for complex
numbers.

## How does the `asinh` function behave in MATLAB / RunMat?
- Accepts scalars, vectors, matrices, and N-D tensors and applies the operation element-wise using MATLAB's broadcasting rules.
- Logical inputs are promoted to doubles (`true → 1.0`, `false → 0.0`) before the inverse hyperbolic sine is applied.
- Character arrays are interpreted as numeric code points and return dense double arrays with the same shape.
- Real inputs produce real outputs for all finite values. Complex inputs follow MATLAB's principal branch definition `asinh(z) = log(z + sqrt(z^2 + 1))`.
- NaNs and infinities propagate according to IEEE arithmetic and match MATLAB's special-case tables.

## `asinh` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the GPU when:

1. A provider is registered and implements the `unary_asinh` hook.
2. The inputs do not require complex promotion that the provider cannot represent.

If either condition is not met, RunMat gathers the data to the host, evaluates `asinh` with the CPU
reference implementation, and returns the correct MATLAB result. This guarantees correctness without
forcing users to manually call `gpuArray` or `gather`.

## Examples of using the `asinh` function in MATLAB / RunMat

### Inverse hyperbolic sine of a scalar

```matlab
y = asinh(0.5);
```

Expected output:

```matlab
y = 0.4812
```

### Applying `asinh` to every element of a vector

```matlab
x = linspace(-2, 2, 5);
y = asinh(x);
```

Expected output:

```matlab
y = [-1.4436 -0.9624 0 0.9624 1.4436]
```

### Evaluating `asinh` on a matrix

```matlab
A = [0 -0.5 1.0; 1.5 -2.0 3.0];
B = asinh(A);
```

Expected output:

```matlab
B =
         0   -0.4812    0.8814
    1.1948   -1.4436    1.8184
```

### Computing `asinh` on GPU data

```matlab
G = gpuArray([0.25 0.5; 0.75 1.0]);
result_gpu = asinh(G);
result = gather(result_gpu);
```

Expected output:

```matlab
result = [0.2470 0.4812; 0.6931 0.8814]
```

### Handling complex arguments with `asinh`

```matlab
z = [1 + 2i, -0.5 + 0.75i];
w = asinh(z);
```

Expected output:

```matlab
w =
   1.4694 + 1.0634i
  -0.4306 + 0.5770i
```

### Using `asinh` inside fused GPU expressions

```matlab
G = gpuArray(rand(1024, 1));
Y = sinh(G) + asinh(G);
```

Expected behavior: The fusion planner keeps the entire expression on the GPU when the provider
implements `unary_sinh` and `unary_asinh`.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. The fusion planner keeps tensors on the
GPU whenever the active provider exposes the necessary hooks (`unary_asinh`) and the runtime can keep
the data in real form. When the GPU path is incomplete, RunMat gathers to the host automatically and
still returns MATLAB-compatible results. Manual `gpuArray` / `gather` remains available for workflows
that require explicit residency control.

## FAQ

### Does `asinh` ever return complex numbers for real inputs?
No. Real arguments always produce real results. Complex outputs arise only when the inputs themselves
are complex.

### How are logical or integer inputs handled?
They are promoted to double precision before evaluation, mirroring MATLAB's behavior. The result is
returned as a dense double tensor (or complex tensor for complex inputs).

### What happens if the GPU provider lacks `unary_asinh`?
RunMat gathers the data to the host, evaluates `asinh` on the CPU, and keeps the result on the host
unless a downstream operation requests GPU residency.

### Is there a precision difference between CPU and GPU outputs?
Both paths operate in the provider's precision (`f32` or `f64`). Small rounding differences may appear
near very large magnitudes, but the results stay within MATLAB's tolerance specs.

### Can `asinh` participate in fusion?
Yes. Element-wise fusion templates include `asinh`, allowing the fusion planner to inline it alongside
other unary operations when generating WGSL kernels.

### How are character arrays processed?
They are interpreted as their Unicode code points, converted to doubles, and then passed through
`asinh`, producing a numeric array that matches MATLAB's output.

### What happens with NaN or Inf values?
NaNs propagate unchanged. Positive and negative infinities map to infinities with the same sign, matching
MATLAB's special-case rules.

### Can I keep complex results on the GPU?
Currently GPU tensor handles represent real data. When complex values arise, RunMat returns host-resident
`Value::Complex` or `Value::ComplexTensor` results to remain MATLAB-compatible.

### Does `asinh` support automatic differentiation plans?
Yes. The fusion and acceleration metadata mark `asinh` as an element-wise operation, so future autodiff
infrastructure can reuse the same kernel hooks.

## See Also
[sinh](./sinh), [tanh](./tanh), [sin](./sin), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `asinh` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/asinh.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/asinh.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "asinh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_asinh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute asinh directly on device buffers; runtimes gather to host when unary_asinh is unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "asinh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("asinh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `asinh` calls; providers may override via fused elementwise kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("asinh", DOC_MD);

#[runtime_builtin(
    name = "asinh",
    category = "math/trigonometry",
    summary = "Inverse hyperbolic sine of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "asinh,arcsinh,inverse hyperbolic sine,trigonometry,gpu",
    accel = "unary"
)]
fn asinh_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => asinh_gpu(handle),
        Value::Complex(re, im) => Ok(complex_asinh_scalar(re, im)),
        Value::ComplexTensor(ct) => asinh_complex_tensor(ct),
        Value::CharArray(ca) => asinh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err("asinh: expected numeric input".to_string())
        }
        other => asinh_real(other),
    }
}

fn asinh_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_asinh(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    asinh_tensor(tensor).map(tensor::tensor_into_value)
}

fn asinh_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("asinh", value)?;
    asinh_tensor(tensor).map(tensor::tensor_into_value)
}

fn asinh_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.asinh()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("asinh: {e}"))
}

fn asinh_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| {
            let res = Complex64::new(re, im).asinh();
            (res.re, res.im)
        })
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("asinh: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn asinh_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).asinh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("asinh: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn complex_asinh_scalar(re: f64, im: f64) -> Value {
    let result = Complex64::new(re, im).asinh();
    Value::Complex(result.re, result.im)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use num_complex::Complex64;
    use runmat_builtins::LogicalArray;

    #[test]
    fn asinh_scalar() {
        let value = Value::Num(0.5);
        let result = asinh_builtin(value).expect("asinh");
        match result {
            Value::Num(v) => assert!((v - 0.48121182505960347).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn asinh_tensor_values() {
        let tensor =
            Tensor::new(vec![0.0, -0.5, 1.0, 3.0], vec![2, 2]).expect("tensor construction");
        let result = asinh_builtin(Value::Tensor(tensor)).expect("asinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    0.0,
                    -0.48121182505960347,
                    0.881373587019543,
                    1.8184464592320668,
                ];
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn asinh_complex_inputs() {
        let inputs = [Complex64::new(1.0, 2.0), Complex64::new(-0.5, 0.75)];
        let complex = ComplexTensor::new(inputs.iter().map(|c| (c.re, c.im)).collect(), vec![1, 2])
            .expect("complex tensor");
        let result = asinh_builtin(Value::ComplexTensor(complex)).expect("asinh");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (actual, input) in t.data.iter().zip(inputs.iter()) {
                    let expected = input.asinh();
                    assert!((actual.0 - expected.re).abs() < 1e-12);
                    assert!((actual.1 - expected.im).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn asinh_char_array_roundtrip() {
        let chars = CharArray::new("az".chars().collect(), 1, 2).expect("char array");
        let result = asinh_builtin(Value::CharArray(chars)).expect("asinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [(('a' as u32) as f64).asinh(), (('z' as u32) as f64).asinh()];
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn asinh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new(vec![0.0, 0.5, 1.0, 2.0], vec![2, 2]).expect("tensor construction");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = asinh_builtin(Value::GpuTensor(handle)).expect("asinh");
            let gathered = test_support::gather(result).expect("gather");
            let expected = tensor.data.iter().map(|&v| v.asinh()).collect::<Vec<_>>();
            assert_eq!(gathered.shape, vec![2, 2]);
            for (actual, exp) in gathered.data.iter().zip(expected.iter()) {
                assert!((actual - exp).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn asinh_logical_array_promotes() {
        let logical =
            LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).expect("logical array construction");
        let result = asinh_builtin(Value::LogicalArray(logical)).expect("asinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [
                    1.0f64.asinh(),
                    0.0f64.asinh(),
                    1.0f64.asinh(),
                    1.0f64.asinh(),
                ];
                for (actual, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn asinh_string_errors() {
        let err = asinh_builtin(Value::from("not numeric")).expect_err("expected error");
        assert!(
            err.contains("expected numeric input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn asinh_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5, 1]).unwrap();
        let cpu = asinh_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("provider")
            .upload(&view)
            .expect("upload");
        let gpu = asinh_gpu(handle).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (actual, expected) in gathered.data.iter().zip(ct.data.iter()) {
                    assert!(
                        (actual - expected).abs() < tol,
                        "|{} - {}| >= {}",
                        actual,
                        expected,
                        tol
                    );
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }
}
