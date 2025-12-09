//! MATLAB-compatible `tanh` builtin with GPU-aware semantics for RunMat.

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
#[runmat_macros::register_doc_text(name = "tanh")]
pub const DOC_MD: &str = r#"---
title: "tanh"
category: "math/trigonometry"
keywords: ["tanh", "hyperbolic tangent", "trigonometry", "gpu", "elementwise"]
summary: "Hyperbolic tangent of scalars, vectors, matrices, complex numbers, or character arrays with MATLAB broadcasting and GPU acceleration."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers provider unary_tanh hooks; falls back to the host path when a provider is unavailable or cannot service the operand type."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::trigonometry::tanh::tests"
  integration: "builtins::math::trigonometry::tanh::tests::tanh_gpu_provider_roundtrip"
---

# What does the `tanh` function do in MATLAB / RunMat?
`y = tanh(x)` evaluates the hyperbolic tangent of each element in `x`, preserving MATLAB's column-major layout and broadcasting rules across scalars, arrays, and tensors.

## How does the `tanh` function behave in MATLAB / RunMat?
- Operates on scalars, vectors, matrices, and N-D tensors with MATLAB-compatible implicit expansion.
- Logical and integer inputs are promoted to double precision before evaluation so downstream arithmetic keeps MATLAB's numeric semantics.
- Complex values follow the analytic extension `tanh(a + bi) = sinh(a + bi) / cosh(a + bi)`, propagating `NaN`/`Inf` components component-wise.
- Character arrays are interpreted through their Unicode code points and return dense double arrays that mirror MATLAB's behavior.
- Inputs that already live on the GPU stay resident when the provider implements `unary_tanh`; otherwise RunMat gathers to the host, computes, and reapplies residency hints for later operations.
- Empty inputs and singleton dimensions are preserved without introducing extraneous allocations.
- String and string-array arguments raise descriptive errors to match MATLAB's numeric-only contract for the hyperbolic family.

## `tanh` Function GPU Execution Behaviour
- With RunMat Accelerate active, tensors remain on the device and execute through the provider's `unary_tanh` hook (or a fused elementwise kernel) without leaving GPU memory.
- If the provider declines the operation—for example, when it lacks the hook for the active precision—RunMat transparently gathers to the host, computes the result, and reapplies the requested residency rules.
- Fusion planning keeps neighbouring elementwise operators grouped, reducing host↔device transfers even when an intermediate fallback occurs.

## Examples of using the `tanh` function in MATLAB / RunMat

### Hyperbolic tangent of a real scalar

```matlab
y = tanh(1);
```

Expected output:

```matlab
y = 0.7616
```

### Applying `tanh` to a symmetric vector

```matlab
x = linspace(-2, 2, 5);
y = tanh(x);
```

Expected output:

```matlab
y = [-0.9640  -0.7616         0   0.7616   0.9640]
```

### Evaluating `tanh` on a matrix in GPU memory

```matlab
G = gpuArray([0    0.5; 1.0  1.5]);
result_gpu = tanh(G);
result = gather(result_gpu);
```

Expected output:

```matlab
result =
         0    0.4621
    0.7616    0.9051
```

### Computing `tanh` for complex angles

```matlab
z = 0.5 + 1.0i;
w = tanh(z);
```

Expected output:

```matlab
w = 1.0428 + 0.8069i
```

### Converting character codes via `tanh`

```matlab
c = tanh('ABC');
```

Expected output:

```matlab
c = [1.0000  1.0000  1.0000]
```

### Preserving empty array shapes

```matlab
E = zeros(0, 3);
out = tanh(E);
```

Expected output:

```matlab
out = zeros(0, 3)
```

### Stabilising activation functions

```matlab
inputs = [-3 -1 0 1 3];
activations = tanh(inputs / 2);
```

Expected output:

```matlab
activations = [-0.9051  -0.4621         0   0.4621   0.9051]
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. The fusion planner keeps tensors on the GPU whenever the active provider exposes the necessary kernels (such as `unary_tanh`). Manual `gpuArray` / `gather` calls remain supported for MATLAB compatibility or when you need to pin residency before interacting with external code.

## FAQ

### When should I reach for `tanh`?
Use `tanh` for hyperbolic tangent evaluations—common in signal processing, numerical solvers, and neural-network activations thanks to its bounded output.

### Does `tanh` support complex numbers?
Yes. RunMat mirrors MATLAB by evaluating `tanh(z) = sinh(z) / cosh(z)` for complex `z`, producing correct real and imaginary parts while propagating `NaN`/`Inf` values.

### How does the GPU fallback work?
If the provider lacks `unary_tanh`, RunMat gathers the tensor to host memory, computes the result, and reapplies residency choices so downstream GPU consumers still see device-backed tensors when appropriate.

### Can `tanh` appear in fused GPU kernels?
Absolutely. The fusion planner emits WGSL kernels that inline `tanh`, and providers can supply custom fused pipelines for even higher performance.

### How does `tanh` treat logical arrays?
Logical arrays are promoted to `0.0` or `1.0` doubles before evaluation, matching MATLAB's behavior for the hyperbolic family.

### What happens with empty or singleton dimensions?
Shapes are preserved. Empty inputs return empty outputs, and singleton dimensions remain intact so downstream broadcasting behaves as expected.

### Do I need to worry about numerical overflow?
`tanh` saturates towards ±1 for large-magnitude real inputs, providing stable results. Complex poles can still yield infinities, mirroring MATLAB.

### Can I differentiate `tanh` in RunMat?
Yes. The autograd infrastructure recognises `tanh` as a primitive and records it on the reverse-mode tape for native gradients once acceleration is enabled.

## See Also
[sinh](./sinh), [cosh](./cosh), [atanh](./atanh), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `tanh` function is available at: [`crates/runmat-runtime/src/builtins/math/trigonometry/tanh.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/tanh.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tanh",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_tanh" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute tanh directly on the device; runtimes gather to the host when unary_tanh is unavailable.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tanh",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("tanh({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion planner emits WGSL `tanh` calls; providers may override with specialised kernels.",
};

#[runtime_builtin(
    name = "tanh",
    category = "math/trigonometry",
    summary = "Hyperbolic tangent of scalars, vectors, matrices, or N-D tensors (element-wise).",
    keywords = "tanh,hyperbolic tangent,trigonometry,gpu",
    accel = "unary"
)]
fn tanh_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => tanh_gpu(handle),
        Value::Complex(re, im) => {
            let (real, imag) = tanh_complex_parts(re, im);
            Ok(Value::Complex(real, imag))
        }
        Value::ComplexTensor(ct) => tanh_complex_tensor(ct),
        Value::CharArray(ca) => tanh_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("tanh: expected numeric input".to_string()),
        other => tanh_real(other),
    }
}

fn tanh_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_tanh(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    tanh_tensor(tensor).map(tensor::tensor_into_value)
}

fn tanh_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("tanh", value)?;
    tanh_tensor(tensor).map(tensor::tensor_into_value)
}

fn tanh_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.tanh()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("tanh: {e}"))
}

fn tanh_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mapped = ct
        .data
        .iter()
        .map(|&(re, im)| tanh_complex_parts(re, im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(mapped, ct.shape.clone()).map_err(|e| format!("tanh: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn tanh_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).tanh())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("tanh: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn tanh_complex_parts(re: f64, im: f64) -> (f64, f64) {
    // Use tanh(z) = sinh(z) / cosh(z) with explicit real/imag components.
    let sinh_re = re.sinh() * im.cos();
    let sinh_im = re.cosh() * im.sin();
    let cosh_re = re.cosh() * im.cos();
    let cosh_im = re.sinh() * im.sin();
    let denom = cosh_re * cosh_re + cosh_im * cosh_im;
    // Division by zero yields the expected IEEE infinities/NaNs, matching MATLAB's behaviour at poles.
    let real = (sinh_re * cosh_re + sinh_im * cosh_im) / denom;
    let imag = (sinh_im * cosh_re - sinh_re * cosh_im) / denom;
    (real, imag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use num_complex::Complex64;
    use runmat_builtins::{CharArray, Tensor};

    #[test]
    fn tanh_scalar_num() {
        let result = tanh_builtin(Value::Num(1.0)).expect("tanh");
        match result {
            Value::Num(v) => assert!((v - 1.0_f64.tanh()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn tanh_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = tanh_builtin(Value::Tensor(tensor)).expect("tanh");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                for (value, expected) in out
                    .data
                    .iter()
                    .zip([-1.0_f64.tanh(), 0.0, 1.0_f64.tanh()].iter())
                {
                    assert!((*value - *expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn tanh_complex_scalar() {
        let result = tanh_builtin(Value::Complex(0.5, 1.0)).expect("tanh");
        match result {
            Value::Complex(re, im) => {
                let target = Complex64::new(0.5, 1.0).tanh();
                assert!((re - target.re).abs() < 1e-12);
                assert!((im - target.im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn tanh_char_array_roundtrip() {
        let chars = CharArray::new("Az".chars().collect(), 1, 2).unwrap();
        let result = tanh_builtin(Value::CharArray(chars)).expect("tanh");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = "Az".chars().map(|c| (c as u32 as f64).tanh()).collect();
                for (value, expect) in t.data.iter().zip(expected.iter()) {
                    assert!((*value - *expect).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn tanh_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5, 1.0, 1.5], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tanh_builtin(Value::GpuTensor(handle)).expect("tanh");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            for (value, expect) in gathered.data.iter().zip(tensor.data.iter()) {
                assert!((*value - expect.tanh()).abs() < 1e-12);
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn tanh_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let tensor = Tensor::new(vec![-1.25, -0.5, 0.0, 0.75, 1.5], vec![5, 1]).unwrap();
        let cpu_value = tanh_real(Value::Tensor(tensor.clone())).expect("cpu tanh");
        let cpu_tensor = test_support::gather(cpu_value).expect("gather cpu");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .upload(&view)
            .expect("upload");
        let gpu_value = tanh_gpu(handle).expect("gpu tanh");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu");

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        let tol = match runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .precision()
        {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (got, expect) in gpu_tensor.data.iter().zip(cpu_tensor.data.iter()) {
            assert!(
                (*got - *expect).abs() < tol,
                "tanh mismatch: got {got}, expect {expect}, tol {tol}"
            );
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
