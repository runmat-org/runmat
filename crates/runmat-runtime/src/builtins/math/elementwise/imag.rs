//! MATLAB-compatible `imag` builtin with GPU-aware semantics for RunMat.

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
#[runmat_macros::register_doc_text(name = "imag")]
pub const DOC_MD: &str = r#"---
title: "imag"
category: "math/elementwise"
keywords: ["imag", "imaginary", "complex", "elementwise", "gpu"]
summary: "Extract the imaginary component of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host when the active provider lacks unary_imag or when inputs require host-only conversions (complex tensors, characters)."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::imag::tests"
  integration: "builtins::math::elementwise::imag::tests::imag_gpu_provider_roundtrip"
---

# What does the `imag` function do in MATLAB / RunMat?
`imag(x)` returns the imaginary component of every element in `x`. Complex inputs yield their imaginary part, while real inputs produce zero-valued results of matching shape.

## How does the `imag` function behave in MATLAB / RunMat?
- Complex scalars, vectors, matrices, and higher-dimensional tensors return only their imaginary components.
- Purely real inputs (double, single, logical) produce zeros of type double that match the input size.
- Character arrays are converted to double and therefore produce zero-filled numeric arrays of the same size.
- String arrays are unsupported and raise an error (`imag` expects numeric, logical, or character data).
- Sparse arrays are currently densified; native sparse support is planned.

## `imag` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that already reside on the GPU stay on the device. The runtime checks whether the active provider implements the `unary_imag` hook:

- **Hook available:** The provider materialises a zero-filled tensor directly on the GPU without any host transfers.
- **Hook missing or unsupported dtype:** RunMat gathers the tensor to host memory, applies the CPU semantics (including the conversions for logical and character inputs), and resumes execution. Downstream fusion can still re-upload the result when profitable.

Complex GPU tensors are currently gathered to the host because GPU-side complex storage is not yet available; providers can add fused support later without changing this builtin.

## Examples of using the `imag` function in MATLAB / RunMat

### Extracting the imaginary part of a complex scalar

```matlab
z = 3 + 4i;
b = imag(z);
```

Expected output:

```matlab
b = 4;
```

### Retrieving imaginary components of a complex matrix

```matlab
Z = [1+2i 4-3i; -5+0i 7+8i];
Y = imag(Z);
```

Expected output:

```matlab
Y =
     2    -3
     0     8
```

### Verifying that real inputs yield zero

```matlab
data = [-2.5 0 9.75];
values = imag(data);
```

Expected output:

```matlab
values = [0 0 0];
```

### Imaginary part of logical masks

```matlab
mask = logical([0 1 0; 1 0 1]);
zerosOnly = imag(mask);
```

Expected output:

```matlab
zerosOnly =
     0     0     0
     0     0     0
```

### Working with GPU-resident tensors

```matlab
G = rand(2048, 256, "gpuArray");
res = imag(G);
```

When the provider implements `unary_imag`, `res` stays on the GPU. Otherwise RunMat transparently gathers the data, computes the result on the host, and continues execution.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. RunMat's fusion planner and Accelerate layer track residency automatically, keeping tensors on the GPU whenever device execution is profitable. Explicit `gpuArray` / `gather` calls remain available for MATLAB compatibility or when you need deterministic residency control (for example, when integrating with custom CUDA or OpenCL kernels).

## FAQ

### Does `imag` modify purely real inputs?
No. Purely real, logical, and character inputs become zero-valued doubles of the same size.

### How does `imag` handle complex zeros?
`imag(0 + 0i)` returns exactly `0`. Imaginary zeros are preserved.

### Can I call `imag` on string arrays?
No. Like MATLAB, `imag` only accepts numeric, logical, or character arrays. Convert strings with `double(string)` first if you require numeric codes.

### Does `imag` allocate a new array?
Yes, in line with MATLAB semantics. Fusion may eliminate the allocation when the surrounding expression can be fused safely.

### What happens on the GPU without `unary_imag`?
RunMat gathers the tensor to host memory, applies the CPU semantics (producing zeros or extracting complex components), and allows subsequent operations to re-upload the data if doing so is worthwhile.

### Is GPU execution numerically identical to CPU?
Yes. For real tensors the result is exactly zero; for complex tensors the CPU path matches MATLAB's behaviour.

### Does `imag` participate in fusion?
Yes. The fusion planner can fold `imag` into neighbouring elementwise kernels, letting providers keep tensors on the GPU whenever possible.

## See Also
[real](./real), [abs](./abs), [sign](./sign), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `imag` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/imag.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/imag.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "imag",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_imag" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    two_pass_threshold: None,
    workgroup_size: None,
    nan_mode: ReductionNaN::Include,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_imag to materialise zero tensors in-place; the runtime gathers to the host whenever complex storage or string conversions are required.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "imag",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let literal = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(literal)
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion kernels treat imag as a zero-producing transform for real tensors; providers can override via fused pipelines to keep tensors resident on the GPU.",
};

#[runtime_builtin(
    name = "imag",
    category = "math/elementwise",
    summary = "Extract the imaginary component of scalars, vectors, matrices, or N-D tensors.",
    keywords = "imag,imaginary,complex,elementwise,gpu",
    accel = "unary"
)]
fn imag_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => imag_gpu(handle),
        Value::Complex(_, im) => Ok(Value::Num(im)),
        Value::ComplexTensor(ct) => imag_complex_tensor(ct),
        Value::CharArray(ca) => imag_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("imag: expected numeric input".to_string()),
        x @ (Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)) => imag_real(x),
        other => Err(format!(
            "imag: unsupported input type {:?}; expected numeric, logical, or char input",
            other
        )),
    }
}

fn imag_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_imag(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    imag_tensor(tensor).map(tensor::tensor_into_value)
}

fn imag_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("imag", value)?;
    imag_tensor(tensor).map(tensor::tensor_into_value)
}

fn imag_tensor(tensor: Tensor) -> Result<Tensor, String> {
    Tensor::new(vec![0.0; tensor.data.len()], tensor.shape.clone())
        .map_err(|e| format!("imag: {e}"))
}

fn imag_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let data = ct.data.iter().map(|&(_, im)| im).collect::<Vec<_>>();
    let tensor = Tensor::new(data, ct.shape.clone()).map_err(|e| format!("imag: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn imag_char_array(ca: CharArray) -> Result<Value, String> {
    let zeros = vec![0.0; ca.rows * ca.cols];
    let tensor = Tensor::new(zeros, vec![ca.rows, ca.cols]).map_err(|e| format!("imag: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray, StringArray};

    #[test]
    fn imag_scalar_real_zero() {
        let result = imag_builtin(Value::Num(-2.5)).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn imag_complex_scalar() {
        let result = imag_builtin(Value::Complex(3.0, 4.0)).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 4.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn imag_bool_scalar_zero() {
        let result = imag_builtin(Value::Bool(true)).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn imag_int_scalar_zero() {
        let result = imag_builtin(Value::Int(IntValue::I32(-42))).expect("imag");
        match result {
            Value::Num(n) => assert_eq!(n, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn imag_tensor_real_is_zero() {
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5, 4.25], vec![4, 1]).unwrap();
        let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 1]);
                assert!(t.data.iter().all(|v| *v == 0.0));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn imag_empty_tensor_zero_length() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = imag_builtin(Value::Tensor(tensor)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 3]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn imag_complex_tensor_to_tensor_of_imag_parts() {
        let complex =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).expect("complex tensor");
        let result = imag_builtin(Value::ComplexTensor(complex)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![2.0, 4.5]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn imag_logical_array_zero() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array");
        let result = imag_builtin(Value::LogicalArray(logical)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0.0; 4]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn imag_char_array_zeroes() {
        let chars = CharArray::new("Az".chars().collect(), 1, 2).expect("char array");
        let result = imag_builtin(Value::CharArray(chars)).expect("imag");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn imag_string_error() {
        let err = imag_builtin(Value::from("hello")).expect_err("imag should error");
        assert!(err.contains("expected numeric"));
    }

    #[test]
    fn imag_string_array_error() {
        let arr =
            StringArray::new(vec!["a".to_string(), "b".to_string()], vec![2, 1]).expect("array");
        let err = imag_builtin(Value::StringArray(arr)).expect_err("imag should error");
        assert!(err.contains("expected numeric"));
    }

    #[test]
    fn imag_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = imag_builtin(Value::GpuTensor(handle)).expect("imag");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert!(gathered.data.iter().all(|v| *v == 0.0));
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn imag_wgpu_matches_cpu_zero() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, -2.5, 4.0], vec![4, 1]).unwrap();
        let cpu = imag_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = imag_gpu(h).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu value {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.data.len(), cpu_tensor.data.len());
        for (g, c) in gathered.data.iter().zip(cpu_tensor.data.iter()) {
            assert!((g - c).abs() < 1e-12, "imag mismatch {} vs {}", g, c);
        }
    }
}
