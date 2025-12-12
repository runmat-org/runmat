//! MATLAB-compatible `mpower` builtin (matrix power) with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "mpower",
        builtin_path = "crate::builtins::math::linalg::ops::mpower"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "mpower"
category: "math/linalg/ops"
keywords: ["mpower", "matrix power", "linear algebra", "gpu"]
summary: "Raise scalars or square matrices to a scalar power (A^B) following MATLAB's matrix power semantics."
references: ["https://www.mathworks.com/help/matlab/ref/mpower.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Repeatedly invokes the provider matmul hook with binary exponentiation; falls back to the host when matmul or eye-like allocation is unavailable."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::ops::mpower::tests::matrix_square_power"
  gpu: "builtins::math::linalg::ops::mpower::tests::gpu_matrix_power_roundtrip"
  doc: "builtins::math::linalg::ops::mpower::tests::doc_examples_present"
---

# What does the `mpower` function do in MATLAB / RunMat?
`C = mpower(A, B)` evaluates the matrix power `A^B`. For scalars it behaves like ordinary exponentiation, while square matrices use repeated matrix multiplication with MATLAB-compatible rules for identity, errors, and complex promotion.

## How does the `mpower` function behave in MATLAB / RunMat?
- `A` must be square when it is a matrix. If `A` is `m × n` with `m != n`, RunMat throws `Matrix must be square for matrix power: m×n`.
- The exponent `B` must be a scalar numeric value. Integer-valued doubles are accepted (`2.0` is treated as `2`); non-integer exponents raise `Matrix power requires integer exponent`.
- `A^0` returns the identity matrix of the same size as `A`. Scalar inputs follow the expected rule `x^0 = 1`.
- Positive integers are evaluated with binary exponentiation (`A^5` → `A * A * A * A * A`). Negative exponents are currently unsupported and produce the MATLAB-style message `Negative matrix powers not supported yet`.
- Complex scalars are handled exactly; complex matrices are supported so long as the exponent is an integer.

## `mpower` Function GPU Execution Behaviour
1. When either operand already resides on the GPU, RunMat asks the active acceleration provider for repeated matrix multiplications using the provider’s `matmul` hook.
2. Identity results (`A^0`) are generated with `eye_like` when the provider exposes it; otherwise the runtime fabricates the identity on the host and uploads it.
3. If the provider declines any step (missing `matmul`, unsupported precision, unimplemented identity), RunMat gathers operands to the host, evaluates the CPU reference implementation, and uploads the result back to the device when possible so downstream code can stay on the GPU.

## Examples of using the `mpower` function in MATLAB / RunMat

### Raising a matrix to an integer power

```matlab
A = [1 3; 2 4];
C = mpower(A, 2);
```

Expected output:

```matlab
C =
     7    15
    10    22
```

### Returning the identity matrix when exponent is zero

```matlab
A = [5 2; 7 1];
I = mpower(A, 0);
```

Expected output:

```matlab
I =
     1     0
     0     1
```

### Scalar bases behave like ordinary exponentiation

```matlab
result = mpower(4, 0.5);
```

Expected output:

```matlab
result = 2
```

### Non-integer exponents raise an error for matrices

```matlab
A = [1 2; 3 4];
C = mpower(A, 1.5);
```

Expected output:

```matlab
Error using  mpower
Matrix power requires integer exponent.
```

### Computing matrix powers on the GPU

```matlab
G = gpuArray([2 0; 0 2]);
H = mpower(G, 3);
gather(H)
```

Expected output:

```matlab
ans =
     8     0
     0     8
```

## GPU residency in RunMat (Do I need `gpuArray`?)
When a provider with matrix multiplication is active, RunMat keeps inputs and results on the GPU automatically—`gpuArray(A)^3` never leaves device memory. If the provider cannot service the request, the runtime gathers to the host, computes the reference result, and (when the outcome is real) uploads it back so the calling code still sees a `gpuArray`.

## FAQ

### Does `mpower` support non-square matrices?
No. `mpower` requires `A` to be square. Use `A.^B` for element-wise powers on non-square arrays.

### Can I raise a matrix to a non-integer exponent?
Not yet. RunMat matches MATLAB by requiring the exponent to be an integer-valued scalar for matrix powers. Non-integer exponents raise `Matrix power requires integer exponent`.

### Are negative exponents supported?
Negative matrix exponents will return with `Negative matrix powers not supported yet`. Use `inv(A)` manually for now.

### How does `mpower` differ from the element-wise power operator?
`mpower` performs matrix multiplication repeatedly. Element-wise power (`.^`) raises each entry independently and supports arbitrary array shapes.

### Will results stay on the GPU?
Yes—provided the acceleration provider implements `matmul`. When it does not, RunMat computes on the host but attempts to re-upload the result so GPU pipelines continue smoothly.

### Is there a limit on the exponent magnitude?
Integers up to ±(2³¹−1) are supported. Exponents outside this range trigger a descriptive error.

## See Also
[mtimes](./mtimes), [power](../../elementwise/power), [eye](../../../array/creation/eye), [gpuArray](../../../acceleration/gpu/gpuArray), [gather](../../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/math/linalg/ops/mpower.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/mpower.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::mpower")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mpower",
    op_kind: GpuOpKind::MatMul,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Binary {
            name: "matmul",
            commutative: false,
        },
        ProviderHook::Custom("eye_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses repeated provider matmul calls via binary exponentiation; falls back to the host implementation when matmul or identity creation is unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::ops::mpower")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mpower",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion relies on the provider matmul hook; when unavailable the runtime executes the CPU fallback.",
};

#[runtime_builtin(
    name = "mpower",
    category = "math/linalg/ops",
    summary = "Matrix power with MATLAB-compatible semantics.",
    keywords = "mpower,matrix power,linear algebra,gpu",
    accel = "matmul",
    builtin_path = "crate::builtins::math::linalg::ops::mpower"
)]
fn mpower_builtin(base: Value, exponent: Value) -> Result<Value, String> {
    mpower_eval(&base, &exponent)
}

pub(crate) fn mpower_eval(base: &Value, exponent: &Value) -> Result<Value, String> {
    if let Some(result) = try_gpu_mpower(base, exponent)? {
        return Ok(result);
    }

    let base_host = crate::dispatcher::gather_if_needed(base)?;
    let exponent_host = crate::dispatcher::gather_if_needed(exponent)?;
    let result = crate::elementwise::power(&base_host, &exponent_host)?;

    if matches!(base, Value::GpuTensor(_)) {
        if let Value::Tensor(tensor) = result {
            if let Some(provider) = runmat_accelerate_api::provider() {
                let view = HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                };
                if let Ok(handle) = provider.upload(&view) {
                    return Ok(Value::GpuTensor(handle));
                }
            }
            return Ok(Value::Tensor(tensor));
        }
    }

    Ok(result)
}

fn try_gpu_mpower(base: &Value, exponent: &Value) -> Result<Option<Value>, String> {
    // Only attempt a GPU path when the base already resides on the GPU.
    let handle = match base {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };

    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let exponent_value = match parse_integer_exponent(exponent)? {
        Some(value) => value,
        None => return Ok(None),
    };

    if exponent_value < 0 {
        return Err("Negative matrix powers not supported yet".to_string());
    }
    let shape = handle.shape.clone();
    if shape.len() != 2 {
        return Ok(None);
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows != cols {
        return Err(format!(
            "Matrix must be square for matrix power: {}x{}",
            rows, cols
        ));
    }

    if exponent_value == 0 {
        match gpu_identity_like(provider, handle, rows) {
            Ok(Some(identity)) => return Ok(Some(Value::GpuTensor(identity))),
            Ok(None) => return Ok(None),
            Err(err) => return Err(err),
        }
    }

    if exponent_value == 1 {
        return Ok(Some(Value::GpuTensor(handle.clone())));
    }

    gpu_binary_exponentiation(provider, handle, exponent_value as u32)
}

fn gpu_identity_like(
    provider: &'static dyn AccelProvider,
    prototype: &GpuTensorHandle,
    size: usize,
) -> Result<Option<GpuTensorHandle>, String> {
    match provider.eye_like(prototype) {
        Ok(handle) => Ok(Some(handle)),
        Err(_) => {
            let eye = crate::matrix::matrix_eye(size);
            let view = HostTensorView {
                data: &eye.data,
                shape: &eye.shape,
            };
            match provider.upload(&view) {
                Ok(handle) => Ok(Some(handle)),
                Err(_) => Ok(None),
            }
        }
    }
}

fn gpu_binary_exponentiation(
    provider: &'static dyn AccelProvider,
    base: &GpuTensorHandle,
    exponent: u32,
) -> Result<Option<Value>, String> {
    let mut exp = exponent;
    let mut base_state = HandleState::borrowed(base);
    let mut result_state: Option<HandleState> = None;

    while exp > 0 {
        if exp & 1 == 1 {
            if let Some(ref mut current) = result_state {
                match provider.matmul(&current.handle, &base_state.handle) {
                    Ok(new_handle) => {
                        if current.owned {
                            let _ = provider.free(&current.handle);
                        }
                        current.handle = new_handle;
                        current.owned = true;
                    }
                    Err(_) => {
                        if current.owned {
                            let _ = provider.free(&current.handle);
                        }
                        if base_state.owned {
                            let _ = provider.free(&base_state.handle);
                        }
                        return Ok(None);
                    }
                }
            } else {
                result_state = Some(HandleState::borrowed(&base_state.handle));
            }
        }

        exp >>= 1;
        if exp > 0 {
            match provider.matmul(&base_state.handle, &base_state.handle) {
                Ok(new_handle) => {
                    if base_state.owned {
                        let _ = provider.free(&base_state.handle);
                    }
                    base_state.handle = new_handle;
                    base_state.owned = true;
                }
                Err(_) => {
                    if base_state.owned {
                        let _ = provider.free(&base_state.handle);
                    }
                    if let Some(current) = result_state.take() {
                        if current.owned {
                            let _ = provider.free(&current.handle);
                        }
                    }
                    return Ok(None);
                }
            }
        }
    }

    if base_state.owned {
        let _ = provider.free(&base_state.handle);
    }

    let result_state = match result_state {
        Some(state) => state,
        None => return Ok(None),
    };

    Ok(Some(Value::GpuTensor(result_state.handle)))
}

fn parse_integer_exponent(value: &Value) -> Result<Option<i32>, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw > i32::MAX as i64 || raw < i32::MIN as i64 {
                return Err(
                    "mpower: exponent magnitude exceeds supported range (|n| ≤ 2^31−1)".to_string(),
                );
            }
            Ok(Some(raw as i32))
        }
        Value::Num(n) => {
            if !n.is_finite() || n.fract() != 0.0 {
                return Err("Matrix power requires integer exponent".to_string());
            }
            if *n > i32::MAX as f64 || *n < i32::MIN as f64 {
                return Err(
                    "mpower: exponent magnitude exceeds supported range (|n| ≤ 2^31−1)".to_string(),
                );
            }
            Ok(Some(*n as i32))
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            let scalar = t.data[0];
            if scalar.fract() != 0.0 || !scalar.is_finite() {
                return Err("Matrix power requires integer exponent".to_string());
            }
            if scalar > i32::MAX as f64 || scalar < i32::MIN as f64 {
                return Err(
                    "mpower: exponent magnitude exceeds supported range (|n| ≤ 2^31−1)".to_string(),
                );
            }
            Ok(Some(scalar as i32))
        }
        _ => Ok(None),
    }
}

#[derive(Clone)]
struct HandleState {
    handle: GpuTensorHandle,
    owned: bool,
}

impl HandleState {
    fn borrowed(handle: &GpuTensorHandle) -> Self {
        Self {
            handle: handle.clone(),
            owned: false,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matrix_square_power() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result =
            mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(2))).expect("mpower");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![7.0, 10.0, 15.0, 22.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zero_exponent_returns_identity() {
        let matrix = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let result =
            mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(0))).expect("mpower");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_inputs_match_standard_power() {
        let result = mpower_builtin(Value::Num(4.0), Value::Num(0.5)).expect("mpower");
        match result {
            Value::Num(v) => assert!((v - 2.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_integer_exponent_errors() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = mpower_builtin(Value::Tensor(matrix), Value::Num(1.5)).unwrap_err();
        assert!(
            err.contains("Matrix power requires integer exponent"),
            "{err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn negative_exponent_errors() {
        let matrix = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(-1))).unwrap_err();
        assert!(
            err.contains("Negative matrix powers not supported yet"),
            "{err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_square_matrix_errors() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = mpower_builtin(Value::Tensor(matrix), Value::Int(IntValue::I32(2))).unwrap_err();
        assert!(
            err.contains("Matrix must be square for matrix power"),
            "{err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_scalar_power() {
        let result =
            mpower_builtin(Value::Complex(2.0, 1.0), Value::Int(IntValue::I32(3))).expect("mpower");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!((im - 11.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_matrix_power_roundtrip() {
        test_support::with_test_provider(|provider| {
            let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &matrix.data,
                shape: &matrix.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = mpower_builtin(Value::GpuTensor(handle), Value::Int(IntValue::I32(3)))
                .expect("gpu mpower");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![37.0, 54.0, 81.0, 118.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
