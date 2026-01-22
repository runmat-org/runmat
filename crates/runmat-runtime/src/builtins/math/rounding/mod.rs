//! MATLAB-compatible `mod` builtin plus rounding helpers for RunMat.

pub(crate) mod ceil;
pub(crate) mod fix;
pub(crate) mod floor;
pub(crate) mod rem;
pub(crate) mod round;

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
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
        name = "mod",
        builtin_path = "crate::builtins::math::rounding"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "mod"
category: "math/rounding"
keywords: ["mod", "modulus", "remainder", "rounding", "gpu"]
summary: "Compute the MATLAB-style modulus a - b .* floor(./b) for scalars, matrices, N-D tensors, and complex values."
references: ["https://www.mathworks.com/help/matlab/ref/mod.html"]
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Composed from elem_div → unary_floor → elem_mul → elem_sub when all GPU operands share a shape; otherwise RunMat gathers to the host."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::rounding::mod::tests"
  integration: "builtins::math::rounding::mod::tests::mod_gpu_pair_roundtrip"
---

# What does the `mod` function do in MATLAB / RunMat?
`C = mod(A, B)` returns the modulus after division such that `C` has the same sign as `B` and satisfies `A = B.*Q + C` with `Q = floor(./B)`.
The definition holds for scalars, vectors, matrices, higher-dimensional tensors, and complex numbers.

## How does the `mod` function behave in MATLAB / RunMat?
- Works with MATLAB-style implicit expansion (broadcasting) between `A` and `B`.
- Returns `NaN` for elements where `B` is zero or both arguments are non-finite in incompatible ways (`Inf` modulo finite, `NaN` inputs, etc.).
- Logical and integer inputs are promoted to double precision; character arrays operate on their Unicode code points.
- Complex inputs use the MATLAB definition `mod(a, b) = a - b.*floor(a./b)` with complex division and component-wise `floor`.
- Empty arrays propagate emptiness while retaining their shapes.

## `mod` Function GPU Execution Behaviour
When both operands are GPU tensors with the same shape, RunMat composes `mod` from the provider hooks `elem_div`, `unary_floor`, `elem_mul`, and `elem_sub`.
This keeps the computation on the device when those hooks are implemented (the shipped WGPU backend and in-process provider expose them).
For mixed residency, shape-mismatched operands, or providers that lack any of these hooks, RunMat gathers to the host, applies the CPU implementation, and returns a host-resident result.

## Examples of using the `mod` function in MATLAB / RunMat

### Computing the modulus of positive integers

```matlab
r = mod(17, 5);
```

Expected output:

```matlab
r = 2;
```

### Modulus with negative divisors keeps the divisor's sign

```matlab
values = [-7 -3 4 9];
mods   = mod(values, -4);
```

Expected output:

```matlab
mods = [-3 -3 0 -3];
```

### Broadcasting a scalar divisor across a matrix

```matlab
A = [4.5  7.1; -2.3  0.4];
result = mod(A, 2);
```

Expected output:

```matlab
result =
    [0.5  1.1;
     1.7  0.4]
```

### MATLAB-compatible modulus for complex numbers

```matlab
z = [3 + 4i, -2 + 5i];
div = 2 + 1i;
res = mod(z, div);
```

Expected output:

```matlab
res =
    [0.0 + 0.0i, 0.0 + 1.0i]
```

### Handling zeros in the divisor

```matlab
warn = mod([2, 0, -2], [0, 0, 0]);
```

Expected output:

```matlab
warn = [NaN NaN NaN];
```

### Using `mod` with character arrays

```matlab
letters = mod('ABC', 5);
```

Expected output:

```matlab
letters = [0 1 2];
```

### Staying on the GPU when hooks are available

```matlab
G = gpuArray(-5:5);
H = mod(G, 4);
cpuCopy = gather(H);
```

Expected output:

```matlab
cpuCopy = [3 0 1 2 3 0 1 2 3 0 1];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Usually not. When the provider exposes the elementwise hooks noted above, `mod` executes entirely on the GPU.
Otherwise RunMat gathers transparently, ensuring MATLAB-compatible behaviour without manual intervention.
Explicit `gpuArray` / `gather` calls remain available for scripts that mirror MathWorks MATLAB workflows.

## FAQ

1. **How is `mod` different from `rem`?** `mod` uses `floor` and keeps the sign of the divisor. `rem` uses `fix` and keeps the sign of the dividend.
2. **What happens when the divisor is zero?** The result is `NaN` (or `NaN + NaNi` for complex inputs), matching MATLAB semantics.
3. **Does `mod` support complex numbers?** Yes. Both operands can be complex; the runtime applies MATLAB's definition with complex division and component-wise `floor`.
4. **Do GPU sources need identical shapes?** Yes. The device fast path currently requires both operands to share the same shape. Other cases fall back to the CPU implementation automatically.
5. **Are empty arrays preserved?** Yes. Empty inputs return empty outputs with the same shape.
6. **Will `mod` ever change integer classes?** Inputs promote to double precision internally; results are reported as double scalars or tensors, mirroring MATLAB's default numeric type.

## See Also
[rem](./rem), [floor](./floor), [fix](./fix), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/rounding/mod.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/mod.rs)
- Found a bug or behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mod",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_div",
            commutative: false,
        },
        ProviderHook::Unary { name: "unary_floor" },
        ProviderHook::Binary {
            name: "elem_mul",
            commutative: false,
        },
        ProviderHook::Binary {
            name: "elem_sub",
            commutative: false,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers can keep mod on-device by composing elem_div → unary_floor → elem_mul → elem_sub for matching shapes. Future backends may expose a dedicated elem_mod hook.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mod",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let a = ctx
                .inputs
                .first()
                .ok_or(FusionError::MissingInput(0))?;
            let b = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            Ok(format!("{a} - {b} * floor({a} / {b})"))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion generates floor(a / b) followed by a - b * q; providers may substitute specialised kernels when available.",
};

const BUILTIN_NAME: &str = "mod";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "mod",
    category = "math/rounding",
    summary = "MATLAB-compatible modulus a - b .* floor(a./b) with support for complex values and broadcasting.",
    keywords = "mod,modulus,remainder,gpu",
    accel = "binary",
    builtin_path = "crate::builtins::math::rounding"
)]
async fn mod_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    match (lhs, rhs) {
        (Value::GpuTensor(a), Value::GpuTensor(b)) => mod_gpu_pair(a, b).await,
        (Value::GpuTensor(a), other) => {
            let gathered = gpu_helpers::gather_tensor_async(&a).await?;
            mod_host(Value::Tensor(gathered), other)
        }
        (other, Value::GpuTensor(b)) => {
            let gathered = gpu_helpers::gather_tensor_async(&b).await?;
            mod_host(other, Value::Tensor(gathered))
        }
        (left, right) => mod_host(left, right),
    }
}

async fn mod_gpu_pair(a: GpuTensorHandle, b: GpuTensorHandle) -> BuiltinResult<Value> {
    if a.device_id == b.device_id {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&a) {
            if a.shape == b.shape {
                if let Ok(div) = provider.elem_div(&a, &b).await {
                    match provider.unary_floor(&div).await {
                        Ok(floored) => match provider.elem_mul(&b, &floored).await {
                            Ok(mul) => match provider.elem_sub(&a, &mul).await {
                                Ok(out) => {
                                    let _ = provider.free(&div);
                                    let _ = provider.free(&floored);
                                    let _ = provider.free(&mul);
                                    return Ok(Value::GpuTensor(out));
                                }
                                Err(_) => {
                                    let _ = provider.free(&mul);
                                    let _ = provider.free(&floored);
                                    let _ = provider.free(&div);
                                }
                            },
                            Err(_) => {
                                let _ = provider.free(&floored);
                                let _ = provider.free(&div);
                            }
                        },
                        Err(_) => {
                            let _ = provider.free(&div);
                        }
                    }
                }
            }
        }
    }
    let left = gpu_helpers::gather_tensor_async(&a).await?;
    let right = gpu_helpers::gather_tensor_async(&b).await?;
    mod_host(Value::Tensor(left), Value::Tensor(right))
}

fn mod_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    let left = value_into_numeric_array(lhs, "mod")?;
    let right = value_into_numeric_array(rhs, "mod")?;
    match align_numeric_arrays(left, right)? {
        NumericPair::Real(a, b) => compute_mod_real(&a, &b),
        NumericPair::Complex(a, b) => compute_mod_complex(&a, &b),
    }
}

fn compute_mod_real(a: &Tensor, b: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&a.shape, &b.shape)
        .map_err(|err| builtin_error(format!("mod: {err}")))?;
    if plan.is_empty() {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("mod: {e}")))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut result = vec![0.0f64; plan.len()];
    for (out_idx, idx_a, idx_b) in plan.iter() {
        let aval = a.data[idx_a];
        let bval = b.data[idx_b];
        result[out_idx] = mod_real_scalar(aval, bval);
    }
    let tensor = Tensor::new(result, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("mod: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn compute_mod_complex(a: &ComplexTensor, b: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&a.shape, &b.shape)
        .map_err(|err| builtin_error(format!("mod: {err}")))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("mod: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut result = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_a, idx_b) in plan.iter() {
        let (ar, ai) = a.data[idx_a];
        let (br, bi) = b.data[idx_b];
        result[out_idx] = mod_complex_scalar(ar, ai, br, bi);
    }
    let tensor = ComplexTensor::new(result, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("mod: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn mod_real_scalar(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if b == 0.0 {
        return f64::NAN;
    }
    if !a.is_finite() && b.is_finite() {
        return f64::NAN;
    }
    let quotient = (a / b).floor();
    let mut remainder = a - b * quotient;
    if remainder == 0.0 {
        remainder = 0.0;
    }
    if b.is_infinite() && a.is_finite() {
        return a;
    }
    if !remainder.is_finite() && !a.is_finite() {
        return f64::NAN;
    }
    let same_sign = remainder == 0.0 || remainder.signum() == b.signum();
    if !same_sign {
        remainder += b;
    }
    if remainder == -0.0 {
        remainder = 0.0;
    }
    remainder
}

fn mod_complex_scalar(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    if (ar.is_nan() || ai.is_nan()) || (br.is_nan() || bi.is_nan()) {
        return (f64::NAN, f64::NAN);
    }
    if br == 0.0 && bi == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if !ar.is_finite() || !ai.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let (qr, qi) = complex_div(ar, ai, br, bi);
    if !qr.is_finite() && !qi.is_finite() && br.is_finite() && bi.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let (fr, fi) = (qr.floor(), qi.floor());
    let (mulr, muli) = complex_mul(br, bi, fr, fi);
    let (rr, ri) = (ar - mulr, ai - muli);
    (normalize_zero(rr), normalize_zero(ri))
}

fn normalize_zero(value: f64) -> f64 {
    if value == -0.0 {
        0.0
    } else {
        value
    }
}

fn complex_mul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn complex_div(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    let denom = br * br + bi * bi;
    if denom == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    ((ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom)
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn value_into_numeric_array(value: Value, name: &str) -> BuiltinResult<NumericArray> {
    match value {
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("{name}: {e}")))?;
            Ok(NumericArray::Complex(tensor))
        }
        Value::ComplexTensor(ct) => Ok(NumericArray::Complex(ct)),
        Value::CharArray(ca) => {
            let data: Vec<f64> = ca.data.iter().map(|&ch| ch as u32 as f64).collect();
            let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
                .map_err(|e| builtin_error(format!("{name}: {e}")))?;
            Ok(NumericArray::Real(tensor))
        }
        Value::String(_) | Value::StringArray(_) => Err(builtin_error(format!(
            "{name}: expected numeric input, got string"
        ))),
        Value::GpuTensor(_) => Err(builtin_error(format!(
            "{name}: internal error converting GPU tensor"
        ))),
        other => {
            let tensor =
                tensor::value_into_tensor_for(name, other).map_err(|err| builtin_error(err))?;
            Ok(NumericArray::Real(tensor))
        }
    }
}

enum NumericArray {
    Real(Tensor),
    Complex(ComplexTensor),
}

enum NumericPair {
    Real(Tensor, Tensor),
    Complex(ComplexTensor, ComplexTensor),
}

fn align_numeric_arrays(lhs: NumericArray, rhs: NumericArray) -> BuiltinResult<NumericPair> {
    match (lhs, rhs) {
        (NumericArray::Real(a), NumericArray::Real(b)) => Ok(NumericPair::Real(a, b)),
        (left, right) => {
            let lc = into_complex(left)?;
            let rc = into_complex(right)?;
            Ok(NumericPair::Complex(lc, rc))
        }
    }
}

fn into_complex(input: NumericArray) -> BuiltinResult<ComplexTensor> {
    match input {
        NumericArray::Real(t) => {
            let Tensor { data, shape, .. } = t;
            let complex: Vec<(f64, f64)> = data.into_iter().map(|re| (re, 0.0)).collect();
            ComplexTensor::new(complex, shape).map_err(|e| builtin_error(format!("mod: {e}")))
        }
        NumericArray::Complex(ct) => Ok(ct),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, Tensor};

    fn mod_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(super::mod_builtin(lhs, rhs))
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
    fn mod_positive_values() {
        let result = mod_builtin(Value::Num(17.0), Value::Num(5.0)).expect("mod");
        match result {
            Value::Num(v) => assert!((v - 2.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_negative_divisor_keeps_sign() {
        let tensor = Tensor::new(vec![-7.0, -3.0, 4.0, 9.0], vec![4, 1]).unwrap();
        let divisor = Tensor::new(vec![-4.0], vec![1, 1]).unwrap();
        let result =
            mod_builtin(Value::Tensor(tensor), Value::Tensor(divisor)).expect("mod broadcast");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.data, vec![-3.0, -3.0, 0.0, -3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_negative_numerator_positive_divisor() {
        let result = mod_builtin(Value::Num(-3.0), Value::Num(2.0)).expect("mod");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_zero_divisor_returns_nan() {
        let result = mod_builtin(Value::Num(3.0), Value::Num(0.0)).expect("mod");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_matrix_scalar_broadcast() {
        let matrix = Tensor::new(vec![4.5, 7.1, -2.3, 0.4], vec![2, 2]).unwrap();
        let result = mod_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = [0.5, 1.1, 1.7, 0.4];
                for (a, b) in t.data.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_complex_operands() {
        let complex =
            ComplexTensor::new(vec![(3.0, 4.0), (-2.0, 5.0)], vec![1, 2]).expect("complex tensor");
        let divisor = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).expect("divisor");
        let result = mod_builtin(Value::ComplexTensor(complex), Value::ComplexTensor(divisor))
            .expect("complex mod");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                let expected = [(0.0, 0.0), (0.0, 1.0)];
                for ((re, im), (er, ei)) in out.data.iter().zip(expected.iter()) {
                    assert!((re - er).abs() < 1e-12);
                    assert!((im - ei).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_char_array_support() {
        let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
        let result = mod_builtin(Value::CharArray(chars), Value::Num(5.0)).expect("mod");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![0.0, 1.0, 2.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_string_input_errors() {
        let err = mod_builtin(Value::from("abc"), Value::Num(3.0))
            .expect_err("string inputs should error");
        assert_error_contains(err, "expected numeric input");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_logical_array_support() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let value =
            mod_builtin(Value::LogicalArray(logical), Value::Num(2.0)).expect("logical mod");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 0.0, 1.0, 0.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_vector_broadcasting() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap();
        let result = mod_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("vector broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_nan_inputs_propagate() {
        let result = mod_builtin(Value::Num(f64::NAN), Value::Num(3.0)).expect("mod");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![-5.0, -3.0, 0.0, 1.0, 6.0, 9.0], vec![3, 2]).unwrap();
            let divisor = Tensor::new(vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0], vec![3, 2]).unwrap();
            let a_view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let b_view = runmat_accelerate_api::HostTensorView {
                data: &divisor.data,
                shape: &divisor.shape,
            };
            let a_handle = provider.upload(&a_view).expect("upload a");
            let b_handle = provider.upload(&b_view).expect("upload b");
            let result =
                mod_builtin(Value::GpuTensor(a_handle), Value::GpuTensor(b_handle)).expect("mod");
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.shape, vec![3, 2]);
            assert_eq!(gathered.data, vec![3.0, 1.0, 0.0, 1.0, 2.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mod_int_scalar_promotes() {
        let result =
            mod_builtin(Value::Int(IntValue::I32(-7)), Value::Int(IntValue::I32(4))).expect("mod");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn mod_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let numer = Tensor::new(vec![-5.0, -3.25, 0.0, 1.75, 6.5, 9.0], vec![3, 2]).unwrap();
        let denom = Tensor::new(vec![4.0, -2.5, 3.0, 3.0, 2.0, -5.0], vec![3, 2]).unwrap();
        let cpu_value =
            mod_host(Value::Tensor(numer.clone()), Value::Tensor(denom.clone())).expect("cpu mod");

        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let numer_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &numer.data,
                shape: &numer.shape,
            })
            .expect("upload numer");
        let denom_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &denom.data,
                shape: &denom.shape,
            })
            .expect("upload denom");

        let gpu_value = block_on(mod_gpu_pair(numer_handle, denom_handle)).expect("gpu mod");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu result");

        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).expect("scalar tensor"),
            other => panic!("unexpected CPU result {other:?}"),
        };

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (gpu, cpu) in gpu_tensor.data.iter().zip(cpu_tensor.data.iter()) {
            assert!(
                (gpu - cpu).abs() <= tol,
                "|{gpu} - {cpu}| exceeded tolerance {tol}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
