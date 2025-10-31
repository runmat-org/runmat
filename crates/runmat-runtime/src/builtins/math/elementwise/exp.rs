//! MATLAB-compatible `exp` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise exponential for real, logical, character, and complex inputs while
//! preserving MATLAB broadcasting semantics. GPU execution uses provider hooks when available
//! and falls back to host computation otherwise.

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
title: "exp"
category: "math/elementwise"
keywords: ["exp", "exponential", "elementwise", "gpu", "complex"]
summary: "Element-wise exponential of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the active provider lacks unary_exp."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::exp::tests"
  integration: "builtins::math::elementwise::exp::tests::exp_gpu_provider_roundtrip"
---

# What does the `exp` function do in MATLAB / RunMat?
`Y = exp(X)` raises *e* to the power of every element in `X`. Results follow MATLAB semantics
for scalars, vectors, matrices, logical arrays, character arrays, and complex inputs.

## How does the `exp` function behave in MATLAB / RunMat?
- `exp(X)` applies the exponential element-by-element with MATLAB broadcasting rules.
- Logical inputs convert to double (`true → 1.0`, `false → 0.0`) before exponentiation.
- Character arrays are treated as their numeric code points and return dense double tensors.
- Complex values follow MATLAB's definition: `exp(a + bi) = exp(a) * (cos(b) + i·sin(b))`.
- Inputs already living on the GPU stay there when the provider implements `unary_exp`; otherwise
  RunMat gathers the data, computes on the host, and returns the correct residency.

## `exp` Function GPU Execution Behaviour
RunMat Accelerate keeps GPU tensors resident whenever the selected provider implements
`unary_exp`. When the hook is missing or returns an error, RunMat automatically gathers the tensor,
computes on the CPU, and re-wraps the result, ensuring correctness without surprising users.

## Examples of using the `exp` function in MATLAB / RunMat

### Calculate the exponential of a scalar value

```matlab
y = exp(1);
```

Expected output:

```matlab
y = 2.7183;
```

### Apply the exponential function to a vector of growth rates

```matlab
rates = [-1 -0.5 0 0.5 1];
factor = exp(rates);
```

Expected output:

```matlab
factor = [0.3679 0.6065 1 1.6487 2.7183];
```

### Exponentiate every element of a matrix

```matlab
A = [0 1 2; 3 4 5];
B = exp(A);
```

Expected output:

```matlab
B = [1.0000 2.7183 7.3891; 20.0855 54.5982 148.4132];
```

### Compute the exponential of complex numbers

```matlab
z = [1+2i, -1+pi*i];
w = exp(z);
```

Expected output:

```matlab
w = [-1.1312 + 2.4717i, -0.3679 + 0.0000i];
```

### Run element-wise exponential on GPU data

```matlab
G = gpuArray([0 1; 2 3]);
out = exp(G);
result = gather(out);
```

Expected behavior:

```matlab
result = [1.0000 2.7183; 7.3891 20.0855];
```

### Convert character codes to exponentials

```matlab
C = 'ABC';
Y = exp(C);
```

Expected output:

```matlab
Y = [5.9874e+41 1.2946e+42 2.7992e+42];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do **not** need to call `gpuArray` manually in RunMat. The acceleration planner and
fusion engine keep tensors on the GPU automatically when profitable. Users can still call
`gpuArray` for explicit residency or to mirror MathWorks MATLAB workflows.

## FAQ

### When should I use the `exp` function?
Use `exp` whenever you need the natural exponential of a value or array, such as modelling growth,
discounting continuous compounding, or preparing inputs for activation functions.

### Does `exp` preserve tensor shapes?
Yes. `exp` returns a tensor with the same shape as the input, applying broadcasting rules where
applicable.

### How are logical arrays handled?
Logical arrays convert to doubles before exponentiation, matching MATLAB behavior:
`exp([true false])` returns `[e 1]`.

### What about complex inputs?
Complex scalars and tensors use MATLAB's complex exponential formula, producing complex outputs.

### What happens when the GPU provider lacks `unary_exp`?
RunMat automatically gathers the data to the host, computes the result, and returns a dense tensor.
Future GPU operations can still fuse because residency metadata is updated accordingly.

### Can I expect double precision?
Yes. RunMat stores dense numeric tensors as double precision (`f64`). Providers may internally use
single precision when configured, but results are converted back to double.

## See Also
[logspace](../../array/creation/logspace), [abs](./abs), [sin](../trigonometry/sin), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `exp` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/exp.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/exp.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "exp",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_exp" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may evaluate exp directly on device buffers; runtimes gather to host when unary_exp is unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "exp",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.get(0).ok_or(FusionError::MissingInput(0))?;
            Ok(format!("exp({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `exp` calls; providers can override with fused elementwise kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("exp", DOC_MD);

#[runtime_builtin(
    name = "exp",
    category = "math/elementwise",
    summary = "Element-wise exponential of scalars, vectors, matrices, or N-D tensors.",
    keywords = "exp,exponential,elementwise,gpu",
    accel = "unary"
)]
fn exp_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => exp_gpu(handle),
        Value::Complex(re, im) => Ok(Value::Complex(
            exp_complex_re(re, im),
            exp_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => exp_complex_tensor(ct),
        Value::CharArray(ca) => exp_char_array(ca),
        Value::String(_) | Value::StringArray(_) => {
            Err("exp: expected numeric input, got string".to_string())
        }
        other => exp_real(other),
    }
}

fn exp_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(out) = provider.unary_exp(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    exp_tensor(tensor).map(tensor::tensor_into_value)
}

fn exp_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("exp", value)?;
    exp_tensor(tensor).map(tensor::tensor_into_value)
}

fn exp_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data: Vec<f64> = tensor.data.iter().map(|&v| v.exp()).collect();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("exp: {e}"))
}

fn exp_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| (exp_complex_re(re, im), exp_complex_im(re, im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("exp: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn exp_char_array(ca: CharArray) -> Result<Value, String> {
    let data: Vec<f64> = ca.data.iter().map(|&ch| (ch as u32 as f64).exp()).collect();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("exp: {e}"))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn exp_complex_re(re: f64, im: f64) -> f64 {
    let exp_re = re.exp();
    exp_re * im.cos()
}

#[inline]
fn exp_complex_im(re: f64, im: f64) -> f64 {
    let exp_re = re.exp();
    exp_re * im.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray, Tensor};

    #[test]
    fn exp_scalar() {
        let result = exp_builtin(Value::Num(1.0)).expect("exp");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::E).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn exp_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let result = exp_builtin(Value::Tensor(tensor)).expect("exp");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                let expected: Vec<f64> = vec![0.0_f64, 1.0_f64, 2.0_f64]
                    .into_iter()
                    .map(|v| v.exp())
                    .collect();
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            Value::Num(_) => panic!("expected tensor result"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn exp_int_value_promotes() {
        let value = Value::Int(IntValue::I32(2));
        let result = exp_builtin(value).expect("exp");
        match result {
            Value::Num(v) => assert!((v - 2.0_f64.exp()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn exp_bool_scalar() {
        let result = exp_builtin(Value::Bool(true)).expect("exp");
        match result {
            Value::Num(v) => assert!((v - std::f64::consts::E).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn exp_complex_scalar() {
        let result = exp_builtin(Value::Complex(1.0, 2.0)).expect("exp");
        match result {
            Value::Complex(re, im) => {
                let expected = (1.0f64.exp() * 2.0f64.cos(), 1.0f64.exp() * 2.0f64.sin());
                assert!((re - expected.0).abs() < 1e-12);
                assert!((im - expected.1).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn exp_complex_tensor_elements() {
        let tensor = ComplexTensor::new(vec![(0.0, 0.0), (1.0, 1.0)], vec![2, 1]).unwrap();
        let result = exp_builtin(Value::ComplexTensor(tensor)).expect("exp");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let expected: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 1.0)]
                    .into_iter()
                    .map(|(re, im)| (exp_complex_re(re, im), exp_complex_im(re, im)))
                    .collect();
                for (idx, (re, im)) in t.data.iter().enumerate() {
                    assert!((re - expected[idx].0).abs() < 1e-12);
                    assert!((im - expected[idx].1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn exp_char_array_roundtrip() {
        let chars = CharArray::new("Hi".chars().collect(), 1, 2).unwrap();
        let result = exp_builtin(Value::CharArray(chars)).expect("exp");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = "Hi".chars().map(|c| (c as u32 as f64).exp()).collect();
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn exp_logical_array_promotes_to_double() {
        let logical =
            LogicalArray::new(vec![1u8, 0u8, 1u8, 0u8], vec![2, 2]).expect("logical array");
        let result = exp_builtin(Value::LogicalArray(logical)).expect("exp");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = vec![std::f64::consts::E, 1.0, std::f64::consts::E, 1.0];
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn exp_string_rejected() {
        let err = exp_builtin(Value::from("runmat")).unwrap_err();
        assert!(
            err.contains("expected numeric input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn exp_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = exp_builtin(Value::GpuTensor(handle)).expect("exp");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.exp()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (a, b) in gathered.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
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
    fn exp_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let cpu = exp_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = exp_gpu(handle).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu value {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered.data.iter().zip(cpu_tensor.data.iter()) {
            assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
        }
    }
}
