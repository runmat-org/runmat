//! MATLAB-compatible `sqrt` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise square roots for real, logical, character, and complex inputs while
//! preserving MATLAB semantics. Negative real values promote to complex outputs. GPU execution
//! utilises provider hooks when available and falls back to host computation whenever complex
//! results are required or the provider lacks the dedicated kernel.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const ZERO_EPS: f64 = 1e-12;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "sqrt")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "sqrt"
category: "math/elementwise"
keywords: ["sqrt", "square root", "elementwise", "gpu", "complex"]
summary: "Element-wise square root of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the provider lacks unary_sqrt or when the result requires complex values."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::sqrt::tests"
  integration: "builtins::math::elementwise::sqrt::tests::sqrt_gpu_provider_roundtrip"
  gpu: "builtins::math::elementwise::sqrt::tests::sqrt_wgpu_matches_cpu_elementwise"
---

# What does the `sqrt` function do in MATLAB / RunMat?
`Y = sqrt(X)` computes the principal square root of every element in `X`. Results follow MATLAB
semantics for scalars, vectors, matrices, logical arrays, character arrays, and complex inputs.
Negative real values automatically promote to complex results so that the output remains exact.

## How does the `sqrt` function behave in MATLAB / RunMat?
- `sqrt(X)` applies the operation element-wise with MATLAB broadcasting rules.
- Logical values convert to doubles (`true → 1.0`, `false → 0.0`) before the square root is taken.
- Character arrays are interpreted as their numeric code points and return dense double tensors.
- Negative real inputs yield purely imaginary outputs: `sqrt([-1 4])` returns `[0 + 1i, 2]`.
- Complex inputs follow MATLAB's definition and stay on the principal branch of the complex square
  root, ensuring continuity across the negative real axis.
- Zeros (including `-0`) remain zero in the output; infinities and NaNs propagate according to IEEE
  arithmetic.

## `sqrt` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the GPU when the active provider implements both `unary_sqrt`
and `reduce_min`, allowing the runtime to prove that every element is non-negative before launching
the kernel. When the dataset contains negative values (and therefore requires complex promotion) or
when either hook is missing, RunMat gathers the data to the host, computes the MATLAB-compatible
result, and updates residency metadata so subsequent operations see the correct value.

## Examples of using the `sqrt` function in MATLAB / RunMat

### Taking the square root of a positive scalar

```matlab
y = sqrt(9);
```

Expected output:

```matlab
y = 3;
```

### Square roots of a matrix

```matlab
A = [1 4 9; 16 25 36];
R = sqrt(A);
```

Expected output:

```matlab
R = [1 2 3; 4 5 6];
```

### Square roots of negative inputs produce complex results

```matlab
values = [-1 -4 9];
roots = sqrt(values);
```

Expected output:

```matlab
roots = [0.0000 + 1.0000i, 0.0000 + 2.0000i, 3.0000 + 0.0000i];
```

### Element-wise square root on GPU data

```matlab
G = gpuArray([0 1; 4 9]);
out = sqrt(G);
result = gather(out);
```

Expected output:

```matlab
result = [0 1; 2 3];
```

### Square root of complex numbers

```matlab
z = [3 + 4i, -1 + 2i];
w = sqrt(z);
```

Expected output:

```matlab
w = [2 + 1i, 0.7862 + 1.2720i];
```

### Square root of character codes

```matlab
C = 'AB';
numericRoots = sqrt(C);
```

Expected output:

```matlab
numericRoots = [8.0623 8.2462];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do **not** need to call `gpuArray` manually. The acceleration planner and fusion
engine keep tensors on the GPU when profitable. If your data contains negative values, RunMat
automatically gathers the tensor, produces the complex result on the host, and keeps residency
metadata in sync. You can still use `gpuArray`/`gather` to mirror MathWorks MATLAB workflows.

## FAQ

### Does `sqrt` return complex results for negative inputs?
Yes. Negative real numbers produce purely imaginary outputs that match MATLAB. Mixed-sign tensors
automatically promote to complex tensors.

### How are logical inputs handled?
Logical arrays convert to doubles before the square root is applied, so `sqrt([true false])`
returns `[1 0]`.

### Can the GPU handle negative inputs?
Providers currently operate on real buffers. When negative values are present, RunMat gathers the
tensor to the host to compute the correct complex result before continuing execution.

### Does `sqrt` preserve the input shape?
Yes. The output has the same shape as the input, with element-wise results.

### How are NaNs and infinities treated?
They follow IEEE rules: `sqrt(NaN)` is `NaN`, `sqrt(Inf)` is `Inf`, and `sqrt(-Inf)` is `0 + Inf·i`.

### What about complex inputs with small imaginary parts?
The implementation uses the principal square root branch and rounds very small real or imaginary
components to zero to avoid `-0` artefacts.

### Will future providers support complex tensors directly?
Yes. The current design promotes to host computation when complex results are needed. Future
providers may expose complex buffers, and the builtin will automatically benefit from those hooks.

## See Also
[abs](./abs), [exp](./exp), [log](./log), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `sqrt` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/sqrt.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/sqrt.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sqrt",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sqrt" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers execute sqrt directly on device buffers when inputs are non-negative; runtime gathers to host when complex promotion is required.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sqrt",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sqrt({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL sqrt calls; providers may replace them with fused elementwise kernels.",
};

#[runtime_builtin(
    name = "sqrt",
    category = "math/elementwise",
    summary = "Element-wise square root of scalars, vectors, matrices, or N-D tensors.",
    keywords = "sqrt,square root,elementwise,gpu,complex",
    accel = "unary"
)]
fn sqrt_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => sqrt_gpu(handle),
        Value::Complex(re, im) => Ok(sqrt_complex_value(re, im)),
        Value::ComplexTensor(ct) => sqrt_complex_tensor(ct),
        Value::CharArray(ca) => sqrt_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("sqrt: expected numeric input".to_string()),
        other => sqrt_real(other),
    }
}

fn sqrt_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        match detect_gpu_requires_complex(provider, &handle) {
            Ok(false) => {
                if let Ok(out) = provider.unary_sqrt(&handle) {
                    return Ok(Value::GpuTensor(out));
                }
            }
            Ok(true) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                return sqrt_tensor_real(tensor);
            }
            Err(_) => {
                // Fall through to host fallback.
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    sqrt_tensor_real(tensor)
}

fn detect_gpu_requires_complex(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<bool, String> {
    let min_handle = provider
        .reduce_min(handle)
        .map_err(|e| format!("sqrt: reduce_min failed: {e}"))?;
    let download = provider
        .download(&min_handle)
        .map_err(|e| format!("sqrt: reduce_min download failed: {e}"));
    let _ = provider.free(&min_handle);
    let host = download?;
    if host.data.iter().any(|&v| v.is_nan()) {
        return Err("sqrt: reduce_min result contained NaN".to_string());
    }
    Ok(host.data.iter().any(|&v| v < 0.0))
}

fn sqrt_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("sqrt", value)?;
    sqrt_tensor_real(tensor)
}

fn sqrt_tensor_real(tensor: Tensor) -> Result<Value, String> {
    let len = tensor.data.len();
    let mut requires_complex = false;
    for &v in &tensor.data {
        if v < 0.0 {
            requires_complex = true;
            break;
        }
    }

    if !requires_complex {
        let mut data = Vec::with_capacity(len);
        for &v in &tensor.data {
            let root = zero_small(v.sqrt());
            data.push(root);
        }
        let tensor = Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("sqrt: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let mut data = Vec::with_capacity(len);
        for &v in &tensor.data {
            if v < 0.0 {
                let imag = zero_small((-v).sqrt());
                data.push((0.0, imag));
            } else {
                let real = zero_small(v.sqrt());
                data.push((real, 0.0));
            }
        }
        if len == 1 {
            let (re, im) = data[0];
            if im == 0.0 {
                Ok(Value::Num(re))
            } else {
                Ok(Value::Complex(re, im))
            }
        } else {
            let tensor =
                ComplexTensor::new(data, tensor.shape.clone()).map_err(|e| format!("sqrt: {e}"))?;
            Ok(Value::ComplexTensor(tensor))
        }
    }
}

fn sqrt_complex_value(re: f64, im: f64) -> Value {
    let (mut real_part, mut imag_part) = sqrt_complex_parts(re, im);
    real_part = zero_small(real_part);
    imag_part = zero_small(imag_part);
    Value::Complex(real_part, imag_part)
}

fn sqrt_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mut data = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let (mut real_part, mut imag_part) = sqrt_complex_parts(re, im);
        real_part = zero_small(real_part);
        imag_part = zero_small(imag_part);
        data.push((real_part, imag_part));
    }
    if data.len() == 1 {
        let (re, im) = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let tensor =
            ComplexTensor::new(data, ct.shape.clone()).map_err(|e| format!("sqrt: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn sqrt_char_array(ca: CharArray) -> Result<Value, String> {
    let mut data = Vec::with_capacity(ca.data.len());
    for &ch in &ca.data {
        let code = ch as u32 as f64;
        data.push(zero_small(code.sqrt()));
    }
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("sqrt: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn sqrt_complex_parts(re: f64, im: f64) -> (f64, f64) {
    if im == 0.0 {
        if re < 0.0 {
            (0.0, (-re).sqrt())
        } else {
            (re.sqrt(), 0.0)
        }
    } else {
        let magnitude = re.hypot(im);
        if magnitude == 0.0 {
            (0.0, 0.0)
        } else {
            let real_part = ((magnitude + re) / 2.0).sqrt();
            let imag_part_raw = ((magnitude - re) / 2.0).sqrt();
            let imag_part = if im >= 0.0 {
                imag_part_raw
            } else {
                -imag_part_raw
            };
            (real_part, imag_part)
        }
    }
}

fn zero_small(value: f64) -> f64 {
    if value.abs() < ZERO_EPS {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, Tensor};

    #[test]
    fn sqrt_scalar_positive() {
        let result = sqrt_builtin(Value::Num(9.0)).expect("sqrt");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn sqrt_scalar_negative() {
        let result = sqrt_builtin(Value::Num(-4.0)).expect("sqrt");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - 2.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn sqrt_bool_true() {
        let result = sqrt_builtin(Value::Bool(true)).expect("sqrt");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn sqrt_logical_array_inputs() {
        let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
        let result = sqrt_builtin(Value::LogicalArray(logical)).expect("sqrt");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!((t.data[0] - 1.0).abs() < 1e-12);
                assert!(t.data[1].abs() < 1e-12);
                assert!((t.data[2] - 1.0).abs() < 1e-12);
                assert!(t.data[3].abs() < 1e-12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn sqrt_tensor_with_negatives() {
        let tensor = Tensor::new(vec![-1.0, 4.0], vec![1, 2]).unwrap();
        let result = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!(ct.data[0].0.abs() < 1e-12);
                assert!((ct.data[0].1 - 1.0).abs() < 1e-12);
                assert!((ct.data[1].0 - 2.0).abs() < 1e-12);
                assert!(ct.data[1].1.abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn sqrt_char_array_inputs() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
        let result = sqrt_builtin(Value::CharArray(chars)).expect("sqrt");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - (65.0f64).sqrt()).abs() < 1e-12);
                assert!((t.data[1] - (90.0f64).sqrt()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sqrt_string_input_errors() {
        let err = sqrt_builtin(Value::from("hello")).unwrap_err();
        assert!(
            err.contains("sqrt: expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn sqrt_complex_scalar() {
        let result = sqrt_builtin(Value::Complex(3.0, 4.0)).expect("sqrt");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!((im - 1.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn sqrt_integer_argument() {
        let result = sqrt_builtin(Value::Int(IntValue::I32(9))).expect("sqrt");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn sqrt_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.data.iter().map(|&v| v.sqrt()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            for (gpu, cpu) in gathered.data.iter().zip(expected.iter()) {
                assert!((gpu - cpu).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn sqrt_gpu_negative_falls_back_to_complex() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.0, 9.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![1, 2]);
                    assert!(ct.data[0].0.abs() < 1e-12);
                    assert!((ct.data[0].1 - 1.0).abs() < 1e-12);
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
    fn sqrt_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
        let cpu = sqrt_real(Value::Tensor(tensor.clone())).expect("cpu sqrt");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_value = sqrt_gpu(handle).expect("gpu sqrt");
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
