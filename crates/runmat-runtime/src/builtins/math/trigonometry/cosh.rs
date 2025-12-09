//! MATLAB-compatible `cosh` builtin with GPU-aware semantics for RunMat.

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
#[runmat_macros::register_doc_text(name = "cosh")]
pub const DOC_MD: &str = r#"---
title: "cosh"
category: "math/trigonometry"
keywords: ["cosh", "hyperbolic cosine", "trigonometry", "elementwise", "gpu"]
summary: "Hyperbolic cosine of scalars, vectors, matrices, complex numbers, or character arrays with MATLAB broadcasting and GPU acceleration."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers provider unary_cosh hooks; falls back to the host path when a provider is unavailable or cannot service the operand type."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::cosh::tests"
  integration: "builtins::math::trigonometry::cosh::tests::cosh_gpu_provider_roundtrip"
---

# What does the `cosh` function do in MATLAB / RunMat?
`Y = cosh(X)` computes the hyperbolic cosine of every element in `X`, extending naturally to complex values.

## How does the `cosh` function behave in MATLAB / RunMat?
- Works on scalars, vectors, matrices, and N-D tensors with MATLAB broadcasting semantics for scalar expansion.
- Logical inputs are converted to double precision (`true → 1.0`, `false → 0.0`) before applying `cosh`.
- Complex inputs follow the analytic rule `cosh(a + bi) = cosh(a)cos(b) + i·sinh(a)sin(b)`, propagating `NaN`/`Inf` components independently.
- Character arrays are converted to their numeric code points prior to evaluation, with double-precision outputs that preserve the input shape.
- Empty arrays return empty results that respect MATLAB’s shape semantics.

## `cosh` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that already reside on the GPU stay there. Providers implementing the optional `unary_cosh` hook execute the operation entirely on the device (and fused elementwise kernels can inline `cosh` alongside other operations). If the active provider lacks this hook, RunMat gathers the data back to the host, computes the reference result, and only re-uploads when downstream operations demand GPU residency.

## Examples of using the `cosh` function in MATLAB / RunMat

### Hyperbolic cosine of a scalar

```matlab
y = cosh(2);
```

Expected output:

```matlab
y = 3.7622;
```

### Applying `cosh` elementwise to a vector

```matlab
x = linspace(-2, 2, 5);
y = cosh(x);
```

Expected output:

```matlab
y = [3.7622 1.5431 1.0000 1.5431 3.7622];
```

### Evaluating `cosh` on a matrix

```matlab
A = [0 0.5; 1.0 1.5];
B = cosh(A);
```

Expected output:

```matlab
B = [1.0000 1.1276; 1.5431 2.3524];
```

### Executing `cosh` on a GPU tensor

```matlab
G = gpuArray([0.25 0.75; 1.25 1.75]);
result_gpu = cosh(G);
result = gather(result_gpu);
```

Expected output:

```matlab
result = [1.0310 1.2947; 1.8890 2.9510];
```

### Working with complex inputs

```matlab
z = 1 + 2i;
w = cosh(z);
```

Expected output:

```matlab
w = -0.6421 + 1.0686i;
```

### Hyperbolic cosine for character codes

```matlab
chars = 'AZ';
codes = cosh(chars);
```

Expected output:

```matlab
codes = [4.3771e18 9.1487e18];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. The fusion planner keeps tensors on the GPU whenever the active provider exposes the required kernels (such as `unary_cosh`). Manual `gpuArray` / `gather` calls remain available for MATLAB compatibility or when you must control residency before interoperating with external code.

## FAQ

### When should I use `cosh`?
Use `cosh` for hyperbolic modelling, solving differential equations, or transforming signals where the hyperbolic cosine naturally appears.

### Does `cosh` work with complex inputs?
Yes. Real and imaginary components are evaluated using the analytic continuation, matching MATLAB’s `cosh` semantics.

### What happens if the GPU provider lacks `unary_cosh`?
RunMat falls back to the host implementation. Tensors are gathered to the CPU, evaluated, and left on the host unless later operations request GPU residency (`'like'`, planner decisions, etc.).

### Can `cosh` participate in fusion?
Yes. The fusion planner can inline `cosh` inside elementwise groups, generating WGSL kernels that execute directly on the GPU when supported.

### Are integers preserved?
Inputs are promoted to double precision before evaluation, matching MATLAB behaviour. Cast back explicitly if you need integer outputs.

### How does `cosh` handle NaN or Inf values?
`cosh` propagates `NaN` and `Inf` in the same way as MATLAB: each component is treated independently, and results mirror IEEE-754 expectations.

### Is there a warmup penalty on first GPU use?
Providers may compile elementwise pipelines during initialization. If `unary_cosh` is unavailable, the CPU fallback avoids the warmup altogether.

## See Also
[cos](./cos), [sinh](./sinh), [tanh](./tanh), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `cosh` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/cosh.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cosh.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cosh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_cosh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute cosh directly on the device; runtimes gather to the host when unary_cosh is unavailable.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cosh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("cosh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `cosh` calls; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "cosh",
    category = "math/trigonometry",
    summary = "Hyperbolic cosine of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "cosh,hyperbolic cosine,trigonometry,gpu",
    accel = "unary"
)]
fn cosh_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => cosh_gpu(handle),
        Value::Complex(re, im) => Ok(Value::Complex(
            cosh_complex_re(re, im),
            cosh_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => cosh_complex_tensor(ct),
        Value::CharArray(ca) => cosh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("cosh: expected numeric input".to_string()),
        other => cosh_real(other),
    }
}

fn cosh_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_cosh(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    cosh_tensor(tensor).map(tensor::tensor_into_value)
}

fn cosh_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("cosh", value)?;
    cosh_tensor(tensor).map(tensor::tensor_into_value)
}

fn cosh_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.cosh()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("cosh: {e}"))
}

fn cosh_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| (cosh_complex_re(re, im), cosh_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("cosh: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn cosh_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).cosh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("cosh: {e}"))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn cosh_complex_re(re: f64, im: f64) -> f64 {
    re.cosh() * im.cos()
}

#[inline]
fn cosh_complex_im(re: f64, im: f64) -> f64 {
    re.sinh() * im.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntValue, LogicalArray, Tensor};

    use crate::builtins::common::test_support;

    #[test]
    fn cosh_scalar() {
        let value = Value::Num(2.0);
        let result = cosh_builtin(value).expect("cosh");
        match result {
            Value::Num(v) => assert!((v - 2.0f64.cosh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn cosh_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = cosh_builtin(Value::Tensor(tensor)).expect("cosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [(-1.0f64).cosh(), 1.0, 1.0f64.cosh()];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cosh_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = cosh_builtin(value).expect("cosh");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.cosh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn cosh_complex_scalar() {
        let result = cosh_builtin(Value::Complex(1.0, 2.0)).expect("cosh");
        match result {
            Value::Complex(re, im) => {
                assert!((re - cosh_complex_re(1.0, 2.0)).abs() < 1e-12);
                assert!((im - cosh_complex_im(1.0, 2.0)).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn cosh_char_array_roundtrip() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = cosh_builtin(Value::CharArray(chars)).expect("cosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                for (idx, ch) in ['A', 'Z'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).cosh();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cosh_logical_array_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let result = cosh_builtin(Value::LogicalArray(logical)).expect("cosh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [1.0f64.cosh(), 0.0f64.cosh(), 1.0f64.cosh()];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cosh_string_errors() {
        let err = cosh_builtin(Value::String("runmat".to_string())).expect_err("expected error");
        assert!(err.contains("numeric"));
    }

    #[test]
    fn cosh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cosh_builtin(Value::GpuTensor(handle)).expect("cosh");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.cosh()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, expected);
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn cosh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 0.25, 0.5, 0.75], vec![4, 1]).unwrap();
        let cpu = cosh_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = cosh_gpu(h).unwrap();
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
