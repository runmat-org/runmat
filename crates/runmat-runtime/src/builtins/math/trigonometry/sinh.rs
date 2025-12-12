//! MATLAB-compatible `sinh` builtin with GPU-aware semantics for RunMat.

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
        name = "sinh",
        builtin_path = "crate::builtins::math::trigonometry::sinh"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "sinh"
category: "math/trigonometry"
keywords: ["sinh", "hyperbolic", "trigonometry", "gpu"]
summary: "Hyperbolic sine of scalars, vectors, matrices, or N-D tensors (element-wise)."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers provider unary_sinh hooks; falls back to the host path when the active provider cannot service the request."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::sinh::tests"
  integration: "builtins::math::trigonometry::sinh::tests::sinh_gpu_provider_roundtrip"
---

# What does the `sinh` function do in MATLAB / RunMat?
`Y = sinh(X)` computes the hyperbolic sine of every element in `X`, treating angles as real numbers and extending naturally to complex values.

## How does the `sinh` function behave in MATLAB / RunMat?
- Works on scalars, vectors, matrices, and N-D tensors with MATLAB broadcasting semantics.
- Logical inputs are converted to double precision (`true → 1.0`, `false → 0.0`) before evaluation.
- Complex inputs follow the analytic definition `sinh(a + bi) = sinh(a)cos(b) + i·cosh(a)sin(b)`, propagating `NaN`/`Inf` components independently.
- Character arrays are converted to their numeric code points prior to evaluation, returning double arrays of the same size.
- Empty arrays return empty results that respect MATLAB’s shape semantics.

## `sinh` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that already reside on the GPU stay there. Providers that implement `unary_sinh` execute the operation entirely on the device (and fused elementwise kernels can inline `sinh` as part of a larger expression). If the active provider lacks this hook, RunMat gathers the data back to the host, computes the result, and re-uploads only when later operations require GPU residency.

## Examples of using the `sinh` function in MATLAB / RunMat

### Compute the hyperbolic sine of a scalar value

```matlab
y = sinh(1);
```

Expected output:

```matlab
y = 1.1752;
```

### Apply `sinh` to each element of a vector

```matlab
x = linspace(-1, 1, 5);
y = sinh(x);
```

Expected output:

```matlab
y = [-1.1752 -0.5211 0 0.5211 1.1752];
```

### Evaluate `sinh` on a matrix

```matlab
A = [0 1; 2 3];
B = sinh(A);
```

Expected output:

```matlab
B = [0 1.1752; 3.6269 10.0179];
```

### Compute the hyperbolic sine on the GPU

```matlab
G = gpuArray([0.25 0.5; 0.75 1.0]);
result_gpu = sinh(G);
result = gather(result_gpu);
```

Expected output:

```matlab
result = [0.2526 0.5211; 0.8223 1.1752];
```

### Work with complex inputs

```matlab
z = 1 + 2i;
w = sinh(z);
```

Expected output:

```matlab
w = -0.4891 + 1.4031i;
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. The fusion planner keeps tensors on the GPU whenever the active provider exposes the necessary kernels (such as `unary_sinh`). Manual `gpuArray` / `gather` calls remain supported for MATLAB compatibility or when you need to pin residency before interacting with external code.

## FAQ

### When should I use `sinh`?
Use `sinh` for hyperbolic sine transformations, signal-processing workflows, and analytic continuations where the hyperbolic trigonometric family is required.

### Does `sinh` work with complex numbers?
Yes. The real and imaginary parts are evaluated using the analytic definition `sinh(a + bi) = sinh(a)cos(b) + i·cosh(a)sin(b)`.

### What happens if the provider does not implement `unary_sinh`?
RunMat gathers the GPU tensor to host memory, performs the computation on the CPU, and leaves the result on the host unless later operations require GPU residency.

### Can `sinh` participate in fusion?
Yes. The fusion planner can inline `sinh` inside elementwise groups, generating WGSL kernels that execute directly on the GPU.

### Does `sinh` respect MATLAB broadcasting rules?
It operates elementwise and therefore respects the same scalar-expansion (broadcasting) semantics as other MATLAB math functions in RunMat.

### Are integers preserved?
Inputs are promoted to double precision before evaluation. To keep data in integer form, perform downstream casting explicitly.

### How does `sinh` handle NaN or Inf values?
Hyperbolic functions propagate `NaN` and `Inf` components consistently with MATLAB. For complex inputs, each component is treated independently.

### Can I combine `sinh` with other hyperbolic functions?
Yes. `sinh`, `cosh`, and `tanh` will share the same residency rules. Fused expressions can include any combination of elementwise builtins.

### Is there a GPU warmup penalty?
Providers may warm up pipelines during initialization. If `unary_sinh` is unavailable, the CPU fallback ensures correctness without warmup.

## See Also
[sin](./sin), [cos](./cos), [tan](./tan), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `sinh` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/sinh.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sinh.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::sinh")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sinh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sinh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute sinh directly on the device; runtimes gather to the host when unary_sinh is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::sinh")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sinh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sinh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `sinh` calls; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "sinh",
    category = "math/trigonometry",
    summary = "Hyperbolic sine of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "sinh,hyperbolic,trigonometry,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::trigonometry::sinh"
)]
fn sinh_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => sinh_gpu(handle),
        Value::Complex(re, im) => Ok(Value::Complex(
            sinh_complex_re(re, im),
            sinh_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => sinh_complex_tensor(ct),
        Value::CharArray(ca) => sinh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("sinh: expected numeric input".to_string()),
        other => sinh_real(other),
    }
}

fn sinh_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_sinh(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    sinh_tensor(tensor).map(tensor::tensor_into_value)
}

fn sinh_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("sinh", value)?;
    sinh_tensor(tensor).map(tensor::tensor_into_value)
}

fn sinh_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.sinh()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("sinh: {e}"))
}

fn sinh_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| (sinh_complex_re(re, im), sinh_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("sinh: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn sinh_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).sinh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("sinh: {e}"))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn sinh_complex_re(re: f64, im: f64) -> f64 {
    re.sinh() * im.cos()
}

#[inline]
fn sinh_complex_im(re: f64, im: f64) -> f64 {
    re.cosh() * im.sin()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{IntValue, Tensor};

    use crate::builtins::common::test_support;

    #[test]
    fn sinh_scalar() {
        let value = Value::Num(1.0);
        let result = sinh_builtin(value).expect("sinh");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.sinh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn sinh_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = sinh_builtin(Value::Tensor(tensor)).expect("sinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected = [-1.0f64.sinh(), 0.0, 1.0f64.sinh()];
                for (got, exp) in t.data.iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sinh_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = sinh_builtin(value).expect("sinh");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.sinh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn sinh_complex_scalar() {
        let result = sinh_builtin(Value::Complex(1.0, 2.0)).expect("sinh");
        match result {
            Value::Complex(re, im) => {
                assert!((re - sinh_complex_re(1.0, 2.0)).abs() < 1e-12);
                assert!((im - sinh_complex_im(1.0, 2.0)).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn sinh_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = sinh_builtin(Value::CharArray(chars)).expect("sinh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).sinh();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sinh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sinh_builtin(Value::GpuTensor(handle)).expect("sinh");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.sinh()).collect();
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
    fn sinh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 0.25, 0.5, 0.75], vec![4, 1]).unwrap();
        let cpu = sinh_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = sinh_gpu(h).unwrap();
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
