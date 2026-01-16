//! MATLAB-compatible `real` builtin with GPU-aware semantics for RunMat.
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "real",
        builtin_path = "crate::builtins::math::elementwise::real"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "real"
category: "math/elementwise"
keywords: ["real", "real part", "complex", "elementwise", "gpu"]
summary: "Extract the real part of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host when the active provider lacks unary_real or when inputs require host-side conversions (complex tensors, characters)."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::real::tests"
  integration: "builtins::math::elementwise::real::tests::real_gpu_provider_roundtrip"
---

# What does the `real` function do in MATLAB / RunMat?
`real(x)` returns the real component of each element in `x`. Real inputs are passed through unchanged, while complex inputs discard their imaginary component.

## How does the `real` function behave in MATLAB / RunMat?
- Complex scalars, vectors, matrices, and higher-dimensional tensors return only their real parts.
- Real inputs (double, single, logical) are preserved exactly, but results are materialised as double precision when necessary (e.g., logical masks become doubles).
- Character arrays are converted to their numeric code points and returned as doubles, matching MATLAB semantics.
- String arrays are not supported and raise an error (`real` expects numeric, logical, or character data).
- Sparse arrays are currently densified (sparse support is planned).

## `real` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that already reside on the GPU stay on the device. The runtime checks whether the active provider implements the `unary_real` hook:

- **Hook available:** The real part is taken directly on the device with no host transfers (the WGPU provider and the in-process provider both implement this hook).
- **Hook missing or unsupported dtype:** RunMat gathers the tensor to host memory, applies the CPU semantics (including logical and character promotion), and hands control back to the caller. Downstream fusion can still re-upload if the planner decides it is profitable.

Complex GPU tensors are currently gathered because device-side complex storage is not yet available; providers can add fused support later without changing this builtin.

## Examples of using the `real` function in MATLAB / RunMat

### Extract the real part of a complex scalar

```matlab
z = 3 + 4i;
r = real(z);
```

Expected output:

```matlab
r = 3;
```

### Take the real part of a complex matrix

```matlab
Z = [1+2i, 4-3i; -5+0i, 7+8i];
R = real(Z);
```

Expected output:

```matlab
R =
     1     4
    -5     7
```

### Confirm that real inputs remain unchanged

```matlab
data = [-2.5 0 9.75];
result = real(data);
```

Expected output:

```matlab
result = [-2.5 0 9.75];
```

### Convert logical masks to doubles

```matlab
mask = logical([0 1 0; 1 1 0]);
numeric = real(mask);
```

Expected output:

```matlab
numeric =
     0     1     0
     1     1     0
```

### Convert characters to numeric codes

```matlab
chars = 'RunMat';
codes = real(chars);
```

Expected output:

```matlab
codes = [82 117 110 77 97 116];
```

### Keep GPU tensors device-resident when possible

```matlab
G = rand(4096, 512, "gpuArray");
R = real(G);
```

`R` stays on the GPU when the provider implements `unary_real`; otherwise RunMat transparently gathers to the host, applies the CPU path, and continues execution.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. RunMat's fusion planner and Accelerate layer track residency automatically, keeping tensors on the GPU whenever device execution is beneficial. Explicit `gpuArray` / `gather` calls remain available for MATLAB compatibility or when you need deterministic residency control (e.g., integrating with third-party GPU kernels).

## FAQ

### Does `real` change purely real inputs?
No. Real, logical, and character inputs are preserved numerically, though logical and character values are returned as doubles to match MATLAB.

### How does `real` handle complex zeros?
`real(0 + 0i)` returns exactly `0`. Imaginary zeros are simply discarded.

### Can I call `real` on string arrays?
No. Like MATLAB, `real` only accepts numeric, logical, or character arrays. Converting strings with `double(string)` first will produce numeric codes if required.

### Does `real` allocate a new array?
Yes. The builtin returns a new tensor (or scalar). Fusion may combine kernels to avoid materialising intermediates when safe.

### What happens on the GPU without `unary_real`?
RunMat gathers the tensor to host memory, applies the CPU semantics, and allows subsequent operations to re-upload if profitable. Providers are encouraged to implement `unary_real` for zero-copy behaviour.

### Is GPU execution numerically identical to CPU?
Yes—`real` is an identity for real tensors. Results match within the provider's precision (`single` or `double`).

### Does `real` participate in fusion?
Yes. The fusion planner can fold `real` into neighbouring elementwise kernels, keeping data on the GPU when possible.

## See Also
[imag](./imag), [abs](./abs), [sign](./sign), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `real` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/real.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/real.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::real")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "real",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_real" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute real in-place via unary_real; the runtime gathers to the host when the hook is absent or when host-only conversions (e.g. complex tensors) are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::real")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "real",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            Ok(format!("({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion kernels treat real as an identity transform for real tensors; providers can override via fused pipelines when advantageous.",
};

const BUILTIN_NAME: &str = "real";

fn builtin_error(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message).with_builtin(BUILTIN_NAME).build().into()
}

#[runtime_builtin(
    name = "real",
    category = "math/elementwise",
    summary = "Extract the real part of scalars, vectors, matrices, or N-D tensors.",
    keywords = "real,real part,complex,elementwise,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::real"
)]
fn real_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => real_gpu(handle),
        Value::Complex(re, _) => Ok(Value::Num(re)),
        Value::ComplexTensor(ct) => real_complex_tensor(ct),
        Value::CharArray(ca) => real_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err(builtin_error("real: expected numeric input")),
        x @ (Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)) => real_real(x),
        other => Err(builtin_error(format!(
            "real: unsupported input type {:?}; expected numeric, logical, or char input",
            other
        ))),
    }
}

fn real_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_real(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(real_tensor(tensor)?))
}

fn real_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("real", value)
        .map_err(|e| builtin_error(format!("real: {e}")))?;
    Ok(tensor::tensor_into_value(real_tensor(tensor)?))
}

fn real_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    Ok(tensor)
}

fn real_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let data = ct.data.iter().map(|&(re, _)| re).collect::<Vec<_>>();
    let tensor = Tensor::new(data, ct.shape.clone())
        .map_err(|e| builtin_error(format!("real: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn real_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("real: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_scalar_num() {
        let result = real_builtin(Value::Num(-2.5)).expect("real");
        match result {
            Value::Num(n) => assert!((n + 2.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_complex_scalar() {
        let result = real_builtin(Value::Complex(3.0, 4.0)).expect("real");
        match result {
            Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_int_promotes_to_double() {
        let result = real_builtin(Value::Int(IntValue::I32(7))).expect("real");
        match result {
            Value::Num(n) => assert!((n - 7.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_complex_tensor_to_real_tensor() {
        let complex =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.0)], vec![2, 1]).expect("complex tensor");
        let result = real_builtin(Value::ComplexTensor(complex)).expect("real");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 1.0).abs() < 1e-12);
                assert!((t.data[1] + 3.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_logical_array_to_numeric() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array");
        let result = real_builtin(Value::LogicalArray(logical)).expect("real");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0.0, 1.0, 1.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_char_array_codes() {
        let chars = CharArray::new("AZ".chars().collect(), 1, 2).expect("char array");
        let result = real_builtin(Value::CharArray(chars)).expect("real");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![65.0, 90.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_string_error() {
        let err = real_builtin(Value::from("hello")).expect_err("real should error");
        match err {
            RuntimeControlFlow::Error(err) => {
                assert!(err.message().contains("expected numeric"));
            }
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspend"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn real_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -2.0, 3.5, -4.25], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = real_builtin(Value::GpuTensor(handle)).expect("real");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, tensor.data);
        });
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
    fn real_wgpu_matches_cpu_identity() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, -2.5, 4.0], vec![4, 1]).unwrap();
        let cpu = real_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = real_gpu(h).unwrap();
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
