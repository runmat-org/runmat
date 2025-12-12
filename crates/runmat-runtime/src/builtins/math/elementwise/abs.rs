//! MATLAB-compatible `abs` builtin with GPU-aware semantics for RunMat.

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
        name = "abs",
        builtin_path = "crate::builtins::math::elementwise::abs"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "abs"
category: "math/elementwise"
keywords: ["abs", "absolute value", "magnitude", "complex", "gpu"]
summary: "Absolute value or magnitude of scalars, vectors, matrices, or N-D tensors."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host when the active provider lacks unary_abs or when complex tensors must be gathered."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::abs::tests"
  integration: "builtins::math::elementwise::abs::tests::abs_gpu_provider_roundtrip"
---

# What does the `abs` function do in MATLAB / RunMat?
`y = abs(x)` returns the absolute value of real inputs and the magnitude (modulus) of complex inputs.
For tensors, the operation is applied element-wise following MATLAB's broadcasting rules.

## How does the `abs` function behave in MATLAB / RunMat?
- Real scalars, vectors, matrices, and N-D tensors are mapped to their element-wise absolute values.
- Complex inputs return the magnitude `sqrt(real(x).^2 + imag(x).^2)`, matching MATLAB semantics.
- Logical inputs are promoted to doubles (`true → 1.0`, `false → 0.0`) before taking the absolute value.
- Character arrays are converted to double arrays of code point magnitudes, just like MATLAB.
- String arrays are not supported and raise an error (`abs` only accepts numeric, logical, or char inputs).
- NaN values remain NaN; the function does not change IEEE NaN propagation rules.

## `abs` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that reside on the GPU stay on the device. The runtime
checks whether the active provider implements the `unary_abs` hook:

- **Hook available:** The absolute value is computed directly on the device with no host transfers.
- **Hook missing or unsupported dtype:** RunMat gathers the tensor to host memory, performs the
  CPU absolute value logic (including complex magnitudes), and optionally re-uploads downstream.

Complex tensors are currently handled on the host because the in-process and WGPU providers emit
real-valued magnitudes. Device providers are encouraged to add fused complex support.

## Examples of using the `abs` function in MATLAB / RunMat

### Getting the absolute value of a scalar

```matlab
y = abs(-42);
```

Expected output:

```matlab
y = 42;
```

### Taking the absolute value of a vector

```matlab
v = [-2 -1 0 1 2];
result = abs(v);
```

Expected output:

```matlab
result = [2 1 0 1 2];
```

### Measuring complex magnitudes

```matlab
z = [3+4i, 1-1i];
magnitudes = abs(z);
```

Expected output:

```matlab
magnitudes = [5 1.4142];
```

### Working with matrices on the GPU

```matlab
G = randn(2048, 2048, "gpuArray");
positive = abs(G);
```

`positive` stays on the GPU when the provider implements `unary_abs`; otherwise RunMat gathers the data,
applies the CPU path, and continues execution transparently.

### Using `abs` with logical arrays

```matlab
mask = logical([0 1 0; 1 0 1]);
numeric = abs(mask);
```

Expected output:

```matlab
numeric = [0 1 0; 1 0 1];
```

### Converting characters to numeric codes

```matlab
c = 'ABC';
codes = abs(c);
```

Expected output:

```matlab
codes = [65 66 67];
```

### Chaining `abs` inside fused expressions

```matlab
x = linspace(-2, 2, 5);
y = abs(x) + x.^2;
```

RunMat's fusion planner keeps tensors on the GPU when profitable, so this expression can
stay device-resident without manual `gpuArray` calls.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. RunMat's fusion planner and
Accelerate layer track residency automatically, keeping tensors on the GPU whenever device
execution is beneficial. Explicit `gpuArray`/`gather` calls remain available for MATLAB
compatibility or when you need deterministic residency control (e.g., integrating with
third-party GPU kernels).

## FAQ

### Does `abs` change NaN values?
No. `abs(NaN)` returns `NaN`, consistent with IEEE arithmetic and MATLAB behaviour.

### What happens to complex numbers?
RunMat returns the magnitude `sqrt(real(x).^2 + imag(x).^2)`, identical to MATLAB.

### Can I call `abs` on string arrays?
No. Like MATLAB, `abs` only accepts numeric, logical, or character inputs. Use `double(string)` if you need code points.

### Does `abs` work with sparse arrays?
Sparse support is planned but not yet implemented; inputs are densified today.

### Is GPU execution exact?
Device execution follows IEEE semantics for the provider's precision (`single` or `double`).
F32 backends may incur small rounding differences compared to CPU double.

### How do I keep results on the GPU?
Avoid calling `gather` unless you need host data. The planner keeps device tensors resident whenever possible.

### Does `abs` allocate new memory?
Yes. The builtin returns a new tensor; fusion may in-place combine kernels to reduce allocations when safe.

### Can I use `abs` with logical masks?
Yes. Logical inputs are promoted to doubles (0 or 1) before applying `abs`, just like MATLAB.

## See Also
[sin](../trigonometry/sin), [sum](../reduction/sum), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `abs` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/abs.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/abs.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::abs")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "abs",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_abs" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute abs in-place; the runtime gathers to host when unary_abs is unavailable or when complex magnitudes are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::abs")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "abs",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("abs({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL abs; providers can swap in specialised kernels.",
};

#[runtime_builtin(
    name = "abs",
    category = "math/elementwise",
    summary = "Absolute value or magnitude of scalars, vectors, matrices, or N-D tensors.",
    keywords = "abs,absolute value,magnitude,complex,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::abs"
)]
fn abs_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => abs_gpu(handle),
        Value::Complex(re, im) => Ok(Value::Num(complex_magnitude(re, im))),
        Value::ComplexTensor(ct) => abs_complex_tensor(ct),
        Value::CharArray(ca) => abs_char_array(ca),
        Value::String(_) | Value::StringArray(_) => Err("abs: expected numeric input".to_string()),
        other => abs_real(other),
    }
}

fn abs_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_abs(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    abs_tensor(tensor).map(tensor::tensor_into_value)
}

fn abs_real(value: Value) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("abs", value)?;
    abs_tensor(tensor).map(tensor::tensor_into_value)
}

fn abs_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let data = tensor.data.iter().map(|&v| v.abs()).collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("abs: {e}"))
}

fn abs_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let data = ct
        .data
        .iter()
        .map(|&(re, im)| complex_magnitude(re, im))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, ct.shape.clone()).map_err(|e| format!("abs: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn abs_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| ch as u32 as f64)
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("abs: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn complex_magnitude(re: f64, im: f64) -> f64 {
    re.hypot(im)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_scalar_negative() {
        let result = abs_builtin(Value::Num(-3.5)).expect("abs");
        match result {
            Value::Num(n) => assert!((n - 3.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_int_promotes() {
        let result = abs_builtin(Value::Int(IntValue::I32(-8))).expect("abs");
        match result {
            Value::Num(n) => assert!((n - 8.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_tensor_elements() {
        let tensor = Tensor::new(vec![-1.0, -2.0, 3.0, -4.0], vec![2, 2]).unwrap();
        let result = abs_builtin(Value::Tensor(tensor)).expect("abs");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_complex_scalar() {
        let result = abs_builtin(Value::Complex(3.0, 4.0)).expect("abs");
        match result {
            Value::Num(n) => assert!((n - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_complex_tensor_to_real_tensor() {
        let complex = ComplexTensor::new(vec![(3.0, 4.0), (1.0, -1.0)], vec![2, 1]).unwrap();
        let result = abs_builtin(Value::ComplexTensor(complex)).expect("abs");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.data[0] - 5.0).abs() < 1e-12);
                assert!((t.data[1] - (2f64).sqrt()).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_char_array_codes() {
        let char_array = CharArray::new("Az".chars().collect(), 1, 2).unwrap();
        let result = abs_builtin(Value::CharArray(char_array)).expect("abs");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![65.0, 122.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_string_rejected() {
        let err = abs_builtin(Value::from("hello")).expect_err("should error");
        assert!(err.contains("expected numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-2.0, -1.0, 0.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = abs_builtin(Value::GpuTensor(handle)).expect("abs");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![2.0, 1.0, 0.0, 3.0]);
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
    fn abs_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.0, -1.0, 0.5, -0.25], vec![4, 1]).unwrap();
        let cpu = abs_real(Value::Tensor(tensor.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = abs_gpu(h).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(ct.data.iter()) {
                    assert!((*a - *b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected result shape"),
        }
    }
}
