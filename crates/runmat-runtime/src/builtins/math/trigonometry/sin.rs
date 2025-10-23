//! MATLAB-compatible `sin` builtin with GPU-aware semantics for RunMat.

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
title: "sin"
category: "math/trigonometry"
keywords: ["sin", "sine", "trigonometry", "gpu"]
summary: "Sine of scalars, vectors, matrices, or N-D tensors (element-wise)."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the active provider lacks unary_sin."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::sin::tests"
  integration: "builtins::math::trigonometry::sin::tests::sin_gpu_provider_roundtrip"
---

# What does the `sin` function do in MATLAB / RunMat?
`y = sin(x)` computes the sine of every element in `x`, with angles specified in radians.

## How does the `sin` function behave in MATLAB / RunMat?
- Works on scalars, vectors, matrices, and N-D tensors with MATLAB broadcasting semantics.
- Logical inputs are converted to double precision (`true → 1.0`, `false → 0.0`) before applying sine.
- Complex inputs follow MATLAB's definition: `sin(a + bi) = sin(a)cosh(b) + i·cos(a)sinh(b)`.
- Character arrays are treated as their numeric code points and return a double array of the same size.

## `sin` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that already reside on the GPU remain on the device.
If the selected provider implements `unary_sin`, the operation executes directly on the GPU. Otherwise,
RunMat gathers the data from the GPU to the host and falls back to the host implementation.

## Examples of using the `sin` function in MATLAB / RunMat

### Computing the sine of a scalar

```matlab
y = sin(pi/2);
```

Expected output:

```matlab
y = 1;
```

### Computing the sine of a vector

```matlab
x = linspace(0, 2*pi, 5);
y = sin(x);
```

Expected output:

```matlab
y = [0 0.5 1 0.5 0];
```

### Computing the sine of a complex number

```matlab
z = sin(1 + 2i);
```

Expected output:

```matlab
z = 3.1657 + 1.9596i;
```

### Computing the sine of a matrix on a GPU

In RunMat:

```matlab
G = rand(1024, 1024);
result = sin(G);
```

In MathWorks MATLAB (supported in RunMat as well):

```matlab
G = gpuArray(rand(1024, 1024));
result_gpu = sin(G);
result = gather(result_gpu);
```

In both cases, the expected output is:

```matlab
result = [0.8415 0.9093 0.1411 -0.7568 -0.9589];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, in the above example, the result of the `sin` call will already be on the GPU when the fusion planner has detected a net benefit to operating the fused expression it is part of on the GPU.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the core language, as their toolbox is a separate commercial product, MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.

## FAQ

### When should I use the `sin` function?
Use `sin` whenever you need to compute the sine of a scalar, vector, matrix, or N-D tensor. This is useful for computing the sine of angles, waves, or other periodic functions.

### Does `sin` produce double arrays by default?
Yes, by default, `sin` creates dense double-precision arrays unless you explicitly specify a type such as `'single'` or use the `'like'` argument to match a prototype array.

### What does `sin(A)` return?
If you call `sin(A)`, where `A` is an array, the result is a new array of the same shape as `A` with the sine of each element. For example, if `A` is a 2x3 matrix, `sin(A)` will return a 2x3 matrix with the sine of each element.

## See Also
[cos](./cos), [tan](./tan), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `sin` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sin.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sin",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sin" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute sin in-place on the device; runtimes gather to host when unary_sin is unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sin",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.get(0).ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sin({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `sin` calls; providers may override via fused elementwise kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("sin", DOC_MD);

#[runtime_builtin(
    name = "sin",
    category = "math/trigonometry",
    summary = "Sine of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "sin,sine,trigonometry,gpu",
    accel = "unary"
)]
fn sin_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => sin_gpu(handle),
        Value::Complex(re, im) => Ok(Value::Complex(
            sin_complex_re(re, im),
            sin_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => sin_complex_tensor(ct),
        Value::CharArray(ca) => sin_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("sin: expected numeric input".to_string()),
        other => sin_real(other),
    }
}

fn sin_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(out) = provider.unary_sin(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    sin_tensor(tensor).map(tensor::tensor_into_value)
}

fn sin_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("sin", value)?;
    sin_tensor(tensor).map(tensor::tensor_into_value)
}

fn sin_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.sin()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("sin: {e}"))
}

fn sin_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| (sin_complex_re(re, im), sin_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("sin: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn sin_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).sin())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("sin: {e}"))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn sin_complex_re(re: f64, im: f64) -> f64 {
    re.sin() * im.cosh()
}

#[inline]
fn sin_complex_im(re: f64, im: f64) -> f64 {
    re.cos() * im.sinh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntValue, Tensor};

    use crate::builtins::common::test_support;

    #[test]
    fn sin_scalar() {
        let value = Value::Num(std::f64::consts::PI / 2.0);
        let result = sin_builtin(value).expect("sin");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn sin_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let result = sin_builtin(Value::Tensor(tensor)).expect("sin");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 0.0).abs() < 1e-12);
                assert!((t.data[1] - 0.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sin_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = sin_builtin(value).expect("sin");
        match result {
            Value::Num(v) => assert!((v - 1.0_f64.sin()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn sin_complex_scalar() {
        let result = sin_builtin(Value::Complex(1.0, 2.0)).expect("sin");
        match result {
            Value::Complex(re, im) => {
                assert!((re - (1.0f64.sin() * 2.0f64.cosh())).abs() < 1e-12);
                assert!((im - (1.0f64.cos() * 2.0f64.sinh())).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn sin_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = sin_builtin(Value::CharArray(chars)).expect("sin");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).sin();
                    assert!((t.data[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sin_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sin_builtin(Value::GpuTensor(handle)).expect("sin");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.sin()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, expected);
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
    fn sin_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = sin_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = sin_gpu(h).unwrap();
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
