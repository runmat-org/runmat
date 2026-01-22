//! MATLAB-compatible `fix` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "fix",
        builtin_path = "crate::builtins::math::rounding::fix"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "fix"
category: "math/rounding"
keywords: ["fix", "truncate", "rounding", "toward zero", "gpu"]
summary: "Round scalars, vectors, matrices, or N-D tensors toward zero."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to the host implementation when the active provider lacks unary_fix."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::rounding::fix::tests"
  integration: "builtins::math::rounding::fix::tests::fix_gpu_provider_roundtrip"
---

# What does the `fix` function do in MATLAB / RunMat?
`fix(X)` removes the fractional part of each element in `X` by rounding toward zero. Positive numbers behave like `floor`, negatives behave like `ceil`, and zeros stay zero. The operation works element-wise on scalars, vectors, matrices, N-D tensors, and complex values.

## How does the `fix` function behave in MATLAB / RunMat?
- `fix` rounds positive inputs down toward zero and negative inputs up toward zero.
- Integer-valued elements are returned unchanged. Logical inputs are promoted to double before rounding.
- Complex numbers are rounded component-wise (`fix(a + bi) = fix(a) + i·fix(b)`), matching MATLAB.
- `NaN`, `Inf`, and `-Inf` propagate unchanged.
- Character arrays are treated as their numeric code points and return dense double tensors.
- Empty arrays return empty arrays with identical shape.

## `fix` Function GPU Execution Behaviour
When a tensor already resides on the GPU, RunMat Accelerate checks whether the active provider implements the optional `unary_fix` hook. If available, the truncation is executed directly on the device and the result stays on the GPU. If the hook is missing, RunMat gathers the values to the host, applies the CPU implementation, and returns a host-resident result—ensuring MATLAB-compatible semantics everywhere.

## Examples of using the `fix` function in MATLAB / RunMat

### Truncating positive and negative values toward zero

```matlab
values = [-3.7 -2.4 -0.6 0 0.6 2.4 3.7];
truncated = fix(values);
```

Expected output:

```matlab
truncated = [-3 -2 0 0 0 2 3];
```

### Removing fractional parts from a matrix

```matlab
A = [1.9  4.1; -2.8  0.5];
B = fix(A);
```

Expected output:

```matlab
B = [1 4; -2 0];
```

### Keeping GPU residency when kernels are available

```matlab
G = gpuArray(linspace(-2.4, 2.4, 6));
truncGpu = fix(G);
hostValues = gather(truncGpu);
```

Expected output:

```matlab
hostValues = [-2 -1 0 0 1 2];
```

### Dropping fractional parts of complex numbers

```matlab
z = [1.9 + 2.6i, -3.4 - 0.2i];
fixed = fix(z);
```

Expected output:

```matlab
fixed = [1 + 2i, -3 + 0i];
```

### Converting character arrays to truncated numeric codes

```matlab
letters = ['A' 'B' 'C'];
codes = fix(letters);
```

Expected output:

```matlab
codes = [65 66 67];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do **not** need to call `gpuArray` manually. RunMat's planner keeps tensors on the GPU when the provider exposes the needed kernels and it is profitable to do so. If the provider does not yet implement `unary_fix`, RunMat automatically gathers tensors to the host, applies `fix`, and returns the result. You can still use `gpuArray` explicitly for compatibility with MATLAB scripts or to force GPU residency.

## FAQ

### How is `fix` different from `floor` and `ceil`?
`fix` always rounds toward zero. Positive numbers behave like `floor`, negatives behave like `ceil`. Use `floor` or `ceil` when you specifically want to round toward negative or positive infinity.

### What happens to existing integers?
Integer-valued inputs (including logical values) are returned unchanged; the output is still reported as a double tensor, matching MATLAB's default numeric type.

### Does `fix` change NaN or Inf values?
No. `NaN`, `Inf`, and `-Inf` propagate unchanged.

### How are complex numbers handled?
`fix` rounds the real and imaginary components independently, exactly like MATLAB.

### Will `fix` stay on the GPU?
Yes, when the active provider implements `unary_fix`. Otherwise, RunMat gathers to the host and applies the CPU implementation, guaranteeing MATLAB-compatible behaviour.

## See Also
[floor](./floor), [ceil](./ceil), [round](./round), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for `fix` lives at: [`crates/runmat-runtime/src/builtins/math/rounding/fix.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/fix.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding::fix")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fix",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_fix" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_fix to keep fix on device; otherwise the runtime gathers to host and applies CPU truncation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding::fix")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fix",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            let truncated = format!("trunc({input})");
            Ok(format!("select({0}, {1}, {0} == {1})", truncated, zero))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL truncation; providers can substitute custom kernels when unary_fix is available.",
};

const BUILTIN_NAME: &str = "fix";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "fix",
    category = "math/rounding",
    summary = "Round scalars, vectors, matrices, or N-D tensors toward zero.",
    keywords = "fix,truncate,rounding,toward zero,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::rounding::fix"
)]
async fn fix_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => fix_gpu(handle).await,
        Value::Complex(re, im) => Ok(Value::Complex(fix_scalar(re), fix_scalar(im))),
        Value::ComplexTensor(ct) => fix_complex_tensor(ct),
        Value::CharArray(ca) => fix_char_array(ca),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|err| builtin_error(err))?;
            fix_tensor(tensor).map(tensor::tensor_into_value)
        }
        Value::String(_) | Value::StringArray(_) => {
            Err(builtin_error("fix: expected numeric or logical input"))
        }
        other => fix_numeric(other),
    }
}

async fn fix_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_fix(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    fix_tensor(tensor).map(tensor::tensor_into_value)
}

fn fix_numeric(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(n) => Ok(Value::Num(fix_scalar(n))),
        Value::Int(i) => Ok(Value::Num(fix_scalar(i.to_f64()))),
        Value::Bool(b) => Ok(Value::Num(fix_scalar(if b { 1.0 } else { 0.0 }))),
        Value::Tensor(t) => fix_tensor(t).map(tensor::tensor_into_value),
        other => {
            let tensor =
                tensor::value_into_tensor_for("fix", other).map_err(|err| builtin_error(err))?;
            Ok(fix_tensor(tensor).map(tensor::tensor_into_value)?)
        }
    }
}

fn fix_tensor(mut tensor: Tensor) -> BuiltinResult<Tensor> {
    for value in &mut tensor.data {
        *value = fix_scalar(*value);
    }
    Ok(tensor)
}

fn fix_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let data = ct
        .data
        .iter()
        .map(|&(re, im)| (fix_scalar(re), fix_scalar(im)))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(data, ct.shape.clone())
        .map_err(|e| builtin_error(format!("fix: {e}")))?;
    Ok(Value::ComplexTensor(tensor))
}

fn fix_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| fix_scalar(ch as u32 as f64))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| builtin_error(format!("fix: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn fix_scalar(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let truncated = value.trunc();
    if truncated == 0.0 {
        0.0
    } else {
        truncated
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_builtins::{ComplexTensor, IntValue, LogicalArray};

    fn fix_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::fix_builtin(value))
    }

    fn assert_error_contains(error: RuntimeError, needle: &str) {
        assert!(
            error.message().contains(needle),
            "unexpected error: {}",
            error.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_scalar_positive_and_negative() {
        let input = Value::Tensor(
            Tensor::new(vec![-3.7, -2.4, -0.6, 0.0, 0.6, 2.4, 3.7], vec![7, 1]).unwrap(),
        );
        let result = fix_builtin(input).expect("fix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![-3.0, -2.0, 0.0, 0.0, 0.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_tensor_matrix() {
        let tensor = Tensor::new(vec![1.9, 4.1, -2.8, 0.5], vec![2, 2]).unwrap();
        let result = fix_builtin(Value::Tensor(tensor)).expect("fix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 4.0, -2.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_complex_number() {
        let result = fix_builtin(Value::Complex(1.9, -2.2)).expect("fix");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_char_array_returns_numeric_tensor() {
        let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
        let result = fix_builtin(Value::CharArray(chars)).expect("fix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![65.0, 66.0, 67.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_logical_array() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = fix_builtin(Value::LogicalArray(logical)).expect("fix");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 0.0, 1.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_bool_promotes_to_numeric() {
        let result = fix_builtin(Value::Bool(true)).expect("fix");
        match result {
            Value::Num(v) => assert_eq!(v, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_int_value_promotes() {
        let value = Value::Int(IntValue::I32(-42));
        let result = fix_builtin(value).expect("fix");
        match result {
            Value::Num(v) => assert_eq!(v, -42.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_string_errors() {
        let err = fix_builtin(Value::from("abc")).unwrap_err();
        assert_error_contains(err, "expected numeric");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_preserves_special_values_and_canonicalizes_negative_zero() {
        let tensor = Tensor::new(
            vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0],
            vec![4, 1],
        )
        .unwrap();
        let result = fix_builtin(Value::Tensor(tensor)).expect("fix");
        let Value::Tensor(out) = result else {
            panic!("expected tensor result");
        };
        assert!(out.data[0].is_nan(), "NaN should propagate");
        assert_eq!(out.data[1], f64::INFINITY);
        assert_eq!(out.data[2], f64::NEG_INFINITY);
        assert_eq!(out.data[3], 0.0);
        assert!(
            out.data[3].is_sign_positive(),
            "negative zero should canonicalize to +0"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_complex_tensor_rounds_components() {
        let tensor = ComplexTensor::new(vec![(1.9, -2.6), (-3.4, 0.2)], vec![2, 1]).unwrap();
        let result = fix_builtin(Value::ComplexTensor(tensor)).expect("fix");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.data, vec![(1.0, -2.0), (-3.0, 0.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fix_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-1.9, -0.1, 0.1, 2.6], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fix_builtin(Value::GpuTensor(handle)).expect("fix");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data, vec![-1.0, 0.0, 0.0, 2.0]);
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
    fn fix_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![-3.7, -0.4, 0.4, 3.7], vec![4, 1]).unwrap();
        let cpu = fix_tensor(tensor.clone()).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(fix_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }
}
