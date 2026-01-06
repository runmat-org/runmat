//! MATLAB-compatible `conj` builtin with GPU-aware semantics for RunMat.

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
        name = "conj",
        builtin_path = "crate::builtins::math::elementwise::conj"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "conj"
category: "math/elementwise"
keywords: ["conj", "complex conjugate", "complex", "elementwise", "gpu"]
summary: "Compute the complex conjugate of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses the provider's unary_conj hook for real-valued tensors; complex tensors gather to the host today while native complex kernels are still in flight. Fusion treats conj as a pass-through for real data so pipelines stay resident on GPU."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::conj::tests"
  integration: "builtins::math::elementwise::conj::tests::conj_gpu_provider_roundtrip"
---

# What does the `conj` function do in MATLAB / RunMat?
`conj(x)` negates the imaginary component of every element in `x`; real values are returned unchanged. The builtin mirrors MathWorks MATLAB semantics for scalars, vectors, matrices, and N-D tensors.

## How does the `conj` function behave in MATLAB / RunMat?
- Complex scalars and arrays have their imaginary components multiplied by `-1`; when a result has no imaginary part it collapses to a real scalar or tensor just like MATLAB.
- Purely real numeric inputs (double, single, integer) are returned unchanged.
- Logical arrays are promoted to double precision with `true → 1.0` and `false → 0.0`.
- Character arrays are promoted to double precision containing their Unicode code points; the Conjugate operation does not alter the codes because they are real-valued.
- String arrays are not supported and raise an error.

## `conj` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the device whenever the active provider implements the `unary_conj` hook:

- **Hook available:** Real tensors stay on the GPU and are processed in-place (both the in-process provider used for tests and the WGPU provider expose this path).
- **Fusion and auto-offload:** Because `conj` is tagged as an elementwise unary builtin, the fusion planner treats it as a pass-through for real-valued kernels. Native auto-offload therefore keeps fused expressions resident on the GPU whenever the surrounding ops are profitable.
- **Hook missing or complex input:** RunMat gathers the data to host memory, applies the CPU semantics (including complex negation and type promotion), and returns the result. The planner may re-upload tensors later if profitable.

Complex tensors are currently materialised on the host because GPU-side complex layouts are still under development. Providers can add native complex kernels later without changing this builtin.

## Examples of using the `conj` function in MATLAB / RunMat

### Complex conjugate of a scalar value in MATLAB

```matlab
z = 3 + 4i;
result = conj(z);
```

Expected output:

```matlab
result = 3 - 4i;
```

### Apply conj to every element of a complex matrix

```matlab
Z = [1+2i, 4-3i; -5+0i, 7+8i];
C = conj(Z);
```

Expected output:

```matlab
C =
   1 - 2i   4 + 3i
  -5 + 0i   7 - 8i
```

### Ensure conj leaves real inputs unchanged

```matlab
data = [-2.5 0 9.75];
unchanged = conj(data);
```

Expected output:

```matlab
unchanged = [-2.5 0 9.75];
```

### Use conj on logical masks converted to doubles

```matlab
mask = logical([0 1 0; 1 1 0]);
numeric = conj(mask);
```

Expected output:

```matlab
numeric =
     0     1     0
     1     1     0
```

### Convert MATLAB char arrays to numeric codes with conj

```matlab
chars = 'RunMat';
codes = conj(chars);
outputClass = class(codes);
```

Expected output:

```matlab
codes = [82 117 110 77 97 116];
outputClass = 'double'
```

### Compute conjugate on GPU-resident arrays

```matlab
G = rand(4096, 256, "gpuArray");
H = conj(G);
```

`H` stays on the GPU when the provider implements `unary_conj`. Otherwise, RunMat gathers `G`, computes the result on the CPU, and continues execution transparently.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. RunMat's fusion planner and Accelerate layer manage residency and offload decisions automatically, keeping tensors on the GPU whenever device execution is beneficial. Explicit `gpuArray` and `gather` remain available for MATLAB compatibility or fine-grained residency control.

## FAQ

### Does `conj` change purely real inputs?
No. Real values (including logical and character data) are returned unchanged, although logical and character inputs become double precision arrays just like in MATLAB.

### How does `conj` handle complex zeros?
`conj(0 + 0i)` returns `0`. Imaginary zeros remain zero after negation.

### Can I call `conj` on string arrays?
No. The builtin only accepts numeric, logical, or character inputs. Convert strings with `double(string)` if you need numeric codes.

### Does `conj` allocate a new array?
Yes. The builtin materialises a new tensor (or scalar). Fusion may eliminate the allocation when the surrounding expression can be fused safely, especially when the data stays on the GPU.

### What happens on the GPU without `unary_conj`?
RunMat gathers the tensor to host memory, applies the CPU semantics (including complex negation), and allows later operations to re-upload data if advantageous.

### Is GPU execution numerically identical to CPU?
Yes. For real tensors the result is an exact copy; the conjugate matches CPU results bit-for-bit for supported precisions.

### Does `conj` participate in fusion?
Yes. The fusion planner can fold `conj` into neighbouring elementwise kernels, letting providers keep tensors on the GPU whenever possible.

## See Also
[real](./real), [imag](./imag), [abs](./abs), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `conj` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/conj.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/conj.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::conj")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "conj",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_conj" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute conj in-place for real tensors via unary_conj; complex tensors currently gather to the host for conjugation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::conj")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "conj",
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
    notes:
        "Fusion kernels treat conj as an identity for real tensors; complex tensors fall back to the CPU path until native complex fusion is available.",
};

#[runtime_builtin(
    name = "conj",
    category = "math/elementwise",
    summary = "Compute the complex conjugate of scalars, vectors, matrices, or N-D tensors.",
    keywords = "conj,complex conjugate,complex,elementwise,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::conj"
)]
fn conj_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => conj_gpu(handle),
        Value::Complex(re, im) => conj_complex_scalar(re, im),
        Value::ComplexTensor(ct) => conj_complex_tensor(ct),
        Value::CharArray(ca) => conj_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("conj: expected numeric input".to_string()),
        x @ (Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)) => conj_real(x),
        other => Err(format!(
            "conj: unsupported input type {:?}; expected numeric, logical, or char data",
            other
        )),
    }
}

fn conj_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_conj(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    conj_tensor(tensor).map(tensor::tensor_into_value)
}

fn conj_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("conj", value)?;
    conj_tensor(tensor).map(tensor::tensor_into_value)
}

fn conj_tensor(tensor: Tensor) -> Result<Tensor, String> {
    Ok(tensor)
}

fn conj_complex_scalar(re: f64, im: f64) -> Result<Value, String> {
    let imag = -im;
    if imag == 0.0 && !imag.is_nan() {
        Ok(Value::Num(re))
    } else {
        Ok(Value::Complex(re, imag))
    }
}

fn conj_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let ComplexTensor {
        data: ct_data,
        shape,
        ..
    } = ct;

    let mut all_real = true;
    let mut data = Vec::with_capacity(ct_data.len());
    for (re, im) in ct_data {
        let imag = -im;
        if imag != 0.0 || imag.is_nan() {
            all_real = false;
        }
        data.push((re, imag));
    }
    if all_real {
        let real: Vec<f64> = data.into_iter().map(|(re, _)| re).collect();
        let tensor = Tensor::new(real, shape.clone()).map_err(|e| format!("conj: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let tensor = ComplexTensor::new(data, shape).map_err(|e| format!("conj: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn conj_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("conj: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_scalar_real() {
        let result = conj_builtin(Value::Num(-2.5)).expect("conj");
        match result {
            Value::Num(n) => assert!((n + 2.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_scalar() {
        let result = conj_builtin(Value::Complex(3.0, 4.0)).expect("conj");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 3.0).abs() < 1e-12);
                assert!((im + 4.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_scalar_zero_imag_returns_real() {
        let result = conj_builtin(Value::Complex(5.0, 0.0)).expect("conj");
        match result {
            Value::Num(n) => assert!((n - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_promotes_logical_to_double() {
        let logical =
            LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array construction");
        let result = conj_builtin(Value::LogicalArray(logical)).expect("conj");
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
    fn conj_int_promotes_to_double() {
        let result = conj_builtin(Value::Int(IntValue::I32(7))).expect("conj");
        match result {
            Value::Num(n) => assert!((n - 7.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_tensor_to_complex_tensor() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).expect("complex tensor");
        let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                assert_eq!(ct.data[0], (1.0, -2.0));
                assert_eq!(ct.data[1], (-3.0, 4.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_complex_tensor_realises_real_when_imag_zero() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 0.0), (2.0, -0.0)], vec![2, 1]).expect("complex tensor");
        let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![1.0, 2.0]);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_char_array_returns_double_codes() {
        let chars = CharArray::new("Hi".chars().collect(), 1, 2).expect("char array");
        let result = conj_builtin(Value::CharArray(chars)).expect("conj");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![72.0, 105.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_errors_on_string_input() {
        let err = conj_builtin(Value::from("hello")).unwrap_err();
        assert!(
            err.contains("expected numeric input"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conj_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, -3.0, 4.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
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
    fn conj_wgpu_matches_cpu_for_real() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5, 0.0], vec![4, 1]).unwrap();
        let cpu = conj_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = conj_gpu(handle).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                assert_eq!(ct.data, gt.data);
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
