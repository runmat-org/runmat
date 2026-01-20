//! MATLAB-compatible `rem` builtin with GPU-aware semantics for RunMat.

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
        name = "rem",
        builtin_path = "crate::builtins::math::rounding::rem"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "rem"
category: "math/rounding"
keywords: ["rem", "remainder", "truncate", "rounding", "gpu"]
summary: "Compute the MATLAB remainder a - b .* fix(./b) for scalars, matrices, N-D tensors, and complex values."
references: ["https://www.mathworks.com/help/matlab/ref/rem.html"]
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Composed from elem_div → unary_fix → elem_mul → elem_sub when GPU operands share a shape; otherwise RunMat gathers to the host."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::rounding::rem::tests"
  integration: "builtins::math::rounding::rem::tests::rem_gpu_pair_roundtrip"
---

# What does the `rem` function do in MATLAB / RunMat?
`C = rem(A, B)` returns the remainder after division with the quotient rounded toward zero: `C = A - B .* fix(A ./ B)`.
The result has the same sign as the dividend (`A`) and a magnitude strictly smaller than `abs(B)` when `B` is finite.

## How does the `rem` function behave in MATLAB / RunMat?
- Supports scalars, vectors, matrices, and higher-dimensional tensors with MATLAB-style implicit expansion (broadcasting).
- Returns `NaN` when the divisor is zero or when the dividend is non-finite while the divisor is finite.
- Infinite divisors leave finite dividends unchanged (because the quotient is zero).
- Logical and integer inputs promote to double precision; character arrays operate on their Unicode code points.
- Complex inputs follow MATLAB semantics: `rem(z, w) = z - w .* fix(z ./ w)` with `fix` applied component-wise to the complex quotient.
- Empty inputs propagate emptiness while preserving shape.

## `rem` Function GPU Execution Behaviour
When both operands are GPU tensors with identical shapes and the active provider exposes `elem_div`, `unary_fix`, `elem_mul`, and `elem_sub`, RunMat evaluates `rem`
fully on the device by chaining those kernels. Intermediate buffers stay device-side, fusion-aware planners can inline the sequence, and the final remainder handle
remains on the GPU for downstream work. If residency is mixed, shapes differ, or any hook is unavailable, RunMat automatically gathers to the host, executes the
MATLAB-compatible fallback, and returns a CPU tensor while preserving observable semantics.

## Examples of using the `rem` function in MATLAB / RunMat

### Remainder of positive integers

```matlab
r = rem(17, 5);
```

Expected output:

```matlab
r = 2;
```

### Remainder keeps dividend sign for negatives

```matlab
values = [-7 -3 4 9];
remainders = rem(values, 4);
```

Expected output:

```matlab
remainders = [-3 -3 0 1];
```

### Remainder with a negative divisor

```matlab
r = rem(7, -4);
```

Expected output:

```matlab
r = 3;
```

### Applying remainder across arrays with broadcasting

```matlab
A = [4.5 7.1; -2.3 0.4];
B = rem(A, 2);
```

Expected output:

```matlab
B = [0.5 1.1;
     -0.3 0.4];
```

### Complex remainders follow fix-based quotient

```matlab
z = [3 + 4i, -2 + 5i];
w = 2 + 1i;
res = rem(z, w);
```

Expected output:

```matlab
res = [0.0 + 0.0i  0.0 + 1.0i];
```

### Keeping `rem` on the GPU when kernels are available

```matlab
G = gpuArray(-5:5);
H = rem(G, 4);
hostCopy = gather(H);
```

Expected output:

```matlab
hostCopy = [-1 0 -3 -2 -1 0 1 2 3 0 1];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Typically no. The native auto-offload planner keeps tensors on the GPU whenever the provider implements the chained kernels described above or when `rem` participates
in a larger fused expression. If any hook is missing—or when inputs start on the host—RunMat gathers automatically, executes the CPU fallback, and returns the result.
You can still call `gpuArray` and `gather` explicitly for compatibility with MathWorks MATLAB scripts or to control residency manually.

## FAQ

### How is `rem` different from `mod`?
`rem` rounds the quotient with `fix`, so the remainder inherits the dividend's sign. `mod` rounds with `floor`, which ties the remainder to the divisor's sign. Both functions share broadcasting rules, but they differ whenever negative values are involved.

### What happens when the divisor is zero?
MATLAB defines `rem(A, 0)` as `NaN` (and `NaN + NaNi` for complex operands). RunMat mirrors this behaviour so error handling and edge cases match scripts written for MATLAB.

### Does `rem` support infinite inputs?
Yes. When `B` is infinite and `A` is finite, the quotient truncates to zero and the result is exactly `A`. If the dividend is infinite while the divisor is finite, the remainder is `NaN`, preserving MATLAB's rules for indeterminate operations.

### Can `rem` operate on complex numbers?
Absolutely. Both operands may be complex; RunMat computes `rem(z, w)` as `z - w .* fix(z ./ w)`, applying `fix` to the real and imaginary components independently so the result matches MATLAB's complex arithmetic.

### Does `rem` change integer or logical inputs?
Internally, inputs promote to double precision before the remainder is computed. The result is returned as doubles, aligning with MATLAB's numeric tower while still supporting logical, integer, and character array arguments.

### Will `rem` stay on the GPU?
It stays on-device when the provider implements `elem_div`, `unary_fix`, `elem_mul`, and `elem_sub`; fusion-aware planners can keep intermediates resident for downstream kernels. Otherwise the runtime gathers the operands, executes the CPU fallback, and returns a host tensor so correctness is unaffected.

## See Also
[mod](./mod), [fix](./fix), [floor](./floor), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/rounding/rem.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/rem.rs)
- Found a bug or behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::rounding::rem")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rem",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "elem_div",
            commutative: false,
        },
        ProviderHook::Unary { name: "unary_fix" },
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
        "Providers can compose rem from elem_div → unary_fix → elem_mul → elem_sub. Kernels fall back to host when any hook is missing or shapes differ.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::rounding::rem")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rem",
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
            Ok(format!("{a} - {b} * trunc({a} / {b})"))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion expands rem as trunc(a / b) followed by a - b * q; providers may override with specialised kernels.",
};

const BUILTIN_NAME: &str = "rem";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "rem",
    category = "math/rounding",
    summary = "MATLAB-compatible remainder a - b .* fix(a./b) with support for complex values and broadcasting.",
    keywords = "rem,remainder,truncate,gpu",
    accel = "binary",
    builtin_path = "crate::builtins::math::rounding::rem"
)]
fn rem_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    match (lhs, rhs) {
        (Value::GpuTensor(a), Value::GpuTensor(b)) => rem_gpu_pair(a, b),
        (Value::GpuTensor(a), other) => {
            let gathered = gpu_helpers::gather_tensor(&a)?;
            rem_host(Value::Tensor(gathered), other)
        }
        (other, Value::GpuTensor(b)) => {
            let gathered = gpu_helpers::gather_tensor(&b)?;
            rem_host(other, Value::Tensor(gathered))
        }
        (left, right) => rem_host(left, right),
    }
}

fn rem_gpu_pair(a: GpuTensorHandle, b: GpuTensorHandle) -> BuiltinResult<Value> {
    if a.device_id == b.device_id {
        if let Some(provider) = runmat_accelerate_api::provider_for_handle(&a) {
            if a.shape == b.shape {
                if let Ok(div) = provider.elem_div(&a, &b) {
                    match provider.unary_fix(&div) {
                        Ok(fixed) => match provider.elem_mul(&b, &fixed) {
                            Ok(mul) => match provider.elem_sub(&a, &mul) {
                                Ok(out) => {
                                    let _ = provider.free(&div);
                                    let _ = provider.free(&fixed);
                                    let _ = provider.free(&mul);
                                    return Ok(Value::GpuTensor(out));
                                }
                                Err(_) => {
                                    let _ = provider.free(&mul);
                                    let _ = provider.free(&fixed);
                                    let _ = provider.free(&div);
                                }
                            },
                            Err(_) => {
                                let _ = provider.free(&fixed);
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
    let left = gpu_helpers::gather_tensor(&a)?;
    let right = gpu_helpers::gather_tensor(&b)?;
    rem_host(Value::Tensor(left), Value::Tensor(right))
}

fn rem_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    let left = value_into_numeric_array(lhs, "rem")?;
    let right = value_into_numeric_array(rhs, "rem")?;
    match align_numeric_arrays(left, right)? {
        NumericPair::Real(a, b) => compute_rem_real(&a, &b),
        NumericPair::Complex(a, b) => compute_rem_complex(&a, &b),
    }
}

fn compute_rem_real(a: &Tensor, b: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&a.shape, &b.shape)
        .map_err(|err| builtin_error(format!("rem: {err}")))?;
    if plan.is_empty() {
        let tensor = Tensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("rem: {e}")))?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    let mut result = vec![0.0f64; plan.len()];
    for (out_idx, idx_a, idx_b) in plan.iter() {
        result[out_idx] = rem_real_scalar(a.data[idx_a], b.data[idx_b]);
    }
    let tensor = Tensor::new(result, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("rem: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn compute_rem_complex(a: &ComplexTensor, b: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&a.shape, &b.shape)
        .map_err(|err| builtin_error(format!("rem: {err}")))?;
    if plan.is_empty() {
        let tensor = ComplexTensor::new(Vec::new(), plan.output_shape().to_vec())
            .map_err(|e| builtin_error(format!("rem: {e}")))?;
        return Ok(complex_tensor_into_value(tensor));
    }
    let mut result = vec![(0.0f64, 0.0f64); plan.len()];
    for (out_idx, idx_a, idx_b) in plan.iter() {
        let (ar, ai) = a.data[idx_a];
        let (br, bi) = b.data[idx_b];
        result[out_idx] = rem_complex_scalar(ar, ai, br, bi);
    }
    let tensor = ComplexTensor::new(result, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("rem: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn rem_real_scalar(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if b == 0.0 {
        return f64::NAN;
    }
    if !a.is_finite() && b.is_finite() {
        return f64::NAN;
    }
    if b.is_infinite() && a.is_finite() {
        return normalize_zero(a);
    }
    let quotient = (a / b).trunc();
    if !quotient.is_finite() && b.is_finite() {
        return f64::NAN;
    }
    let remainder = a - b * quotient;
    normalize_zero(remainder)
}

fn rem_complex_scalar(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
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
    let (fr, fi) = (qr.trunc(), qi.trunc());
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
            let lc = into_complex("rem", left)?;
            let rc = into_complex("rem", right)?;
            Ok(NumericPair::Complex(lc, rc))
        }
    }
}

fn into_complex(name: &str, input: NumericArray) -> BuiltinResult<ComplexTensor> {
    match input {
        NumericArray::Real(t) => {
            let Tensor { data, shape, .. } = t;
            let complex: Vec<(f64, f64)> = data.into_iter().map(|re| (re, 0.0)).collect();
            ComplexTensor::new(complex, shape).map_err(|e| builtin_error(format!("{name}: {e}")))
        }
        NumericArray::Complex(ct) => Ok(ct),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeError;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, Tensor};

    fn assert_error_contains(error: RuntimeError, needle: &str) {
        assert!(
            error.message().contains(needle),
            "unexpected error: {}",
            error.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_positive_values() {
        let result = rem_builtin(Value::Num(17.0), Value::Num(5.0)).expect("rem");
        match result {
            Value::Num(v) => assert!((v - 2.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_negative_dividend_keeps_sign() {
        let result = rem_builtin(Value::Num(-7.0), Value::Num(4.0)).expect("rem");
        match result {
            Value::Num(v) => assert!((v + 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_negative_divisor_retains_dividend_sign() {
        let result = rem_builtin(Value::Num(7.0), Value::Num(-4.0)).expect("rem");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_zero_divisor_returns_nan() {
        let result = rem_builtin(Value::Num(3.0), Value::Num(0.0)).expect("rem");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_infinite_divisor_returns_dividend() {
        let result = rem_builtin(Value::Num(4.5), Value::Num(f64::INFINITY)).expect("rem");
        match result {
            Value::Num(v) => assert!((v - 4.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_infinite_dividend_returns_nan() {
        let result = rem_builtin(Value::Num(f64::INFINITY), Value::Num(3.0)).expect("rem");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_matrix_scalar_broadcast() {
        let matrix = Tensor::new(vec![4.5, 7.1, -2.3, 0.4], vec![2, 2]).unwrap();
        let result = rem_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(
                    t.data
                        .iter()
                        .map(|v| (v * 10.0).round() / 10.0)
                        .collect::<Vec<_>>(),
                    vec![0.5, 1.1, -0.3, 0.4]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_complex_support() {
        let lhs = ComplexTensor::new(vec![(3.0, 4.0), (-2.0, 5.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).unwrap();
        let result =
            rem_builtin(Value::ComplexTensor(lhs), Value::ComplexTensor(rhs)).expect("complex rem");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![(0.0, 0.0), (0.0, 1.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_char_array_support() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result =
            rem_builtin(Value::CharArray(chars.clone()), Value::Num(5.0)).expect("char rem");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = chars
                    .data
                    .iter()
                    .map(|&ch| rem_real_scalar(ch as u32 as f64, 5.0))
                    .collect();
                assert_eq!(t.data, expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_logical_array_support() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let value =
            rem_builtin(Value::LogicalArray(logical), Value::Num(2.0)).expect("logical rem");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 0.0, 1.0, 0.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_broadcasting_between_vectors() {
        let lhs = Tensor::new(vec![1.0, -2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap();
        let result = rem_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("broadcast");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, -2.0, 1.0, -2.0, 1.0, -2.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_gpu_pair_roundtrip() {
        test_support::with_test_provider(|provider| {
            let numer = Tensor::new(vec![-5.0, -3.0, 0.0, 1.0, 6.0, 9.0], vec![3, 2]).unwrap();
            let denom = Tensor::new(vec![4.0; 6], vec![3, 2]).unwrap();
            let numer_view = runmat_accelerate_api::HostTensorView {
                data: &numer.data,
                shape: &numer.shape,
            };
            let denom_view = runmat_accelerate_api::HostTensorView {
                data: &denom.data,
                shape: &denom.shape,
            };
            let numer_handle = provider.upload(&numer_view).expect("upload numer");
            let denom_handle = provider.upload(&denom_view).expect("upload denom");
            let result = rem_builtin(
                Value::GpuTensor(numer_handle),
                Value::GpuTensor(denom_handle),
            )
            .expect("rem");
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.shape, vec![3, 2]);
            assert_eq!(gathered.data, vec![-1.0, -3.0, 0.0, 1.0, 2.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_int_inputs_promote() {
        let result =
            rem_builtin(Value::Int(IntValue::I32(-7)), Value::Int(IntValue::I32(4))).expect("rem");
        match result {
            Value::Num(v) => assert!((v + 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rem_string_input_errors() {
        let err = rem_builtin(Value::from("abc"), Value::Num(3.0))
            .expect_err("string inputs should error");
        assert_error_contains(err, "expected numeric input");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rem_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let numer = Tensor::new(vec![-5.0, -3.25, 0.0, 1.75, 6.5, 9.0], vec![3, 2]).unwrap();
        let denom = Tensor::new(vec![4.0, -2.5, 3.0, 3.0, 2.0, -5.0], vec![3, 2]).unwrap();
        let cpu_value =
            rem_host(Value::Tensor(numer.clone()), Value::Tensor(denom.clone())).expect("cpu rem");

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

        let gpu_value = rem_gpu_pair(numer_handle, denom_handle).expect("gpu rem");
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
