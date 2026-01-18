//! MATLAB-compatible `cond` builtin with GPU-aware fallbacks.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderCondNorm};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::linalg::matrix_dimensions_for;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cond";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = NAME,
        builtin_path = "crate::builtins::math::linalg::solve::cond"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "cond"
category: "math/linalg/solve"
keywords: ["cond", "condition number", "norm", "linear algebra", "gpu"]
summary: "Compute the matrix condition number with MATLAB-compatible norm choices."
references: ["https://www.mathworks.com/help/matlab/ref/cond.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "RunMat gathers GPU inputs to the host when a provider does not expose a native cond hook, then re-uploads the scalar result to preserve residency."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::solve::cond::tests"
  gpu: "builtins::math::linalg::solve::cond::tests::cond_gpu_round_trip_matches_cpu"
  wgpu: "builtins::math::linalg::solve::cond::tests::cond_wgpu_matches_cpu"
  doc: "builtins::math::linalg::solve::cond::tests::doc_examples_present"
---

# What does the `cond` function do in MATLAB / RunMat?
`k = cond(A)` returns the condition number of matrix `A`, measuring how sensitive solutions to
linear systems are to small perturbations in `A` or in the right-hand side. By default, `cond`
computes the 2-norm condition number using the ratio of the largest to smallest singular values.

## How does the `cond` function behave in MATLAB / RunMat?
- `cond(A)` and `cond(A, 2)` use the singular values of `A`, so rectangular matrices are supported.
- `cond(A, 1)`, `cond(A, Inf)`, and `cond(A, 'fro')` require a square, invertible matrix and use
  the definition `norm(A, p) * norm(inv(A), p)` with MATLAB's norm semantics.
- Scalars behave like 1x1 matrices. Non-zero scalars have condition number `1`, while `cond(0) = Inf`.
- Empty matrices return `0`, matching MATLAB's convention.
- Singular or rank-deficient matrices return `Inf`.
- Complex inputs are handled in full complex arithmetic.

## `cond` function GPU execution behaviour
When the input already lives on a GPU, RunMat first looks for an acceleration provider that exposes
the custom `cond` hook registered below. Current providers gather the matrix to host memory, reuse
the shared CPU implementation, and then re-upload the scalar so downstream GPU computations preserve
residency. This mirrors MATLAB semantics while keeping the user-facing API uniform.

## Examples of using the `cond` function in MATLAB / RunMat

### Condition number of the identity matrix
```matlab
A = eye(3);
k = cond(A);
```
Expected output:
```matlab
k = 1
```

### Diagnosing an ill-conditioned diagonal matrix
```matlab
D = diag([1, 1e-8]);
k = cond(D);
```
Expected output:
```matlab
k = 1.0e+8
```

### Condition number of a rectangular matrix (2-norm)
```matlab
A = [1 0; 0 1; 1 1];
k = cond(A, 2);
```
Expected output:
```matlab
k = 1.7321
```

### Using a different norm specification
```matlab
A = [4 -1; 2 3];
k1 = cond(A, 1);
kInf = cond(A, Inf);
```
Expected output (rounded):
```matlab
k1   = 2.1429
kInf = 2.1429
```

### Complex-valued matrices
```matlab
A = [1+2i 0; 3i 4-1i];
k = cond(A);
```
Expected output (rounded):
```matlab
k = 3.0327
```

### Empty inputs and GPU residency
```matlab
G = gpuArray([]);      % Empty 0x0 matrix on the GPU
k = cond(G);           % Returns 0 and keeps residency when possible
result = gather(k);
```
Expected output:
```matlab
result = 0
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Manual calls to `gpuArray` are rarely necessary. When matrices already reside on the device, RunMat
attempts to execute `cond` via the active provider. If no native implementation exists, the runtime
gathers the matrix, computes the condition number with the shared CPU path, and uploads the scalar
result back to the GPU. This preserves compatibility with MATLAB while keeping the workflow simple.

## FAQ

### What does a large condition number mean?
Large condition numbers (>> 1) indicate that small perturbations in the input can produce large
changes in the solution of a linear system involving `A`. Values close to `1` indicate a well-
conditioned matrix.

### Why does `cond` return `Inf` for singular matrices?
Singular matrices have at least one zero singular value (or an undefined inverse), so the condition
number is mathematically infinite. RunMat mirrors MATLAB and returns `Inf` in these cases.

### Does `cond` support rectangular matrices?
Yes for the default 2-norm: `cond(A)` uses singular values and accepts any two-dimensional matrix.
Norms `1`, `Inf`, and `'fro'` require a square, invertible matrix because they are defined using the
matrix inverse.

### How does `cond` handle empty matrices?
All norm choices return `0` for empty matrices (`0x0`), matching MATLAB's behaviour.

### Will calling `cond` move my data off the GPU?
Only when the active provider lacks a dedicated implementation. In that case RunMat gathers the data,
computes the scalar on the host, and uploads it back so subsequent GPU operations still see a
device-resident value.

## See Also
[rcond](./rcond), [inv](./inv), [pinv](./pinv), [linsolve](./linsolve), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/math/linalg/solve/cond.rs`
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose)
  with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::cond")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("cond"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("cond")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may expose a direct condition-number kernel; the reference backends gather to the host, evaluate the shared implementation, and upload the scalar result.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::cond")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fusible; cond consumes an entire matrix and returns a scalar diagnostic.",
};

#[runtime_builtin(
    name = "cond",
    category = "math/linalg/solve",
    summary = "Compute the matrix condition number with MATLAB-compatible norms.",
    keywords = "cond,condition number,norm,gpu",
    accel = "cond",
    builtin_path = "crate::builtins::math::linalg::solve::cond"
)]
fn cond_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let norm = parse_norm_argument(&rest)?;
    let result = match value {
        Value::GpuTensor(handle) => return cond_gpu(handle, norm),
        Value::ComplexTensor(matrix) => cond_complex_tensor_builtin(&matrix, norm)?,
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            cond_complex_tensor_builtin(&tensor, norm)?
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            cond_real_tensor_builtin(&tensor, norm)?
        }
    };
    Ok(Value::Num(result))
}

fn cond_gpu(handle: GpuTensorHandle, norm: CondNorm) -> BuiltinResult<Value> {
    let maybe_provider = runmat_accelerate_api::provider();

    if let Some(provider) = maybe_provider {
        if let Some(value) = cond_gpu_via_provider(provider, &handle, norm)? {
            return Ok(value);
        }
    }

    let gathered =
        gpu_helpers::gather_value(&Value::GpuTensor(handle.clone())).map_err(map_control_flow)?;

    let cond_value = match gathered {
        Value::Tensor(tensor) => cond_real_tensor_builtin(&tensor, norm)?,
        Value::ComplexTensor(tensor) => cond_complex_tensor_builtin(&tensor, norm)?,
        Value::Num(n) => {
            if n == 0.0 {
                f64::INFINITY
            } else {
                1.0
            }
        }
        Value::Complex(re, im) => {
            if re == 0.0 && im == 0.0 {
                f64::INFINITY
            } else {
                1.0
            }
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            cond_real_tensor_builtin(&tensor, norm)?
        }
    };

    if let Some(provider) = maybe_provider {
        match upload_scalar(provider, cond_value) {
            Ok(uploaded) => return Ok(Value::GpuTensor(uploaded)),
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(build_runtime_error("interaction pending...")
                        .with_builtin(NAME)
                        .build());
                }
            }
        }
    }

    Ok(Value::Num(cond_value))
}

fn cond_gpu_via_provider(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    norm: CondNorm,
) -> BuiltinResult<Option<Value>> {
    let provider_norm = ProviderCondNorm::from(norm);
    match provider.cond(handle, provider_norm) {
        Ok(result) => Ok(Some(Value::GpuTensor(result))),
        Err(_err) => Ok(None),
    }
}

fn cond_real_tensor_builtin(matrix: &Tensor, norm: CondNorm) -> BuiltinResult<f64> {
    cond_real_tensor(matrix, norm)
}

fn cond_complex_tensor_builtin(matrix: &ComplexTensor, norm: CondNorm) -> BuiltinResult<f64> {
    cond_complex_tensor(matrix, norm)
}

fn cond_real_tensor(matrix: &Tensor, norm: CondNorm) -> BuiltinResult<f64> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape).map_err(builtin_error)?;
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    if matrix.data.len() == 1 {
        return Ok(if matrix.data[0] == 0.0 {
            f64::INFINITY
        } else {
            1.0
        });
    }

    match norm {
        CondNorm::Two => cond_two_norm_real(matrix, rows, cols),
        _ => {
            if rows != cols {
                return Err(builtin_error(format!(
                    "{NAME}: matrix must be square for the requested norm."
                )));
            }
            cond_inverse_based_real(matrix, rows, norm)
        }
    }
}

fn cond_complex_tensor(matrix: &ComplexTensor, norm: CondNorm) -> BuiltinResult<f64> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape).map_err(builtin_error)?;
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    if matrix.data.len() == 1 {
        let (re, im) = matrix.data[0];
        let magnitude = re.hypot(im);
        return Ok(if magnitude == 0.0 { f64::INFINITY } else { 1.0 });
    }

    match norm {
        CondNorm::Two => cond_two_norm_complex(matrix, rows, cols),
        _ => {
            if rows != cols {
                return Err(builtin_error(format!(
                    "{NAME}: matrix must be square for the requested norm."
                )));
            }
            cond_inverse_based_complex(matrix, rows, norm)
        }
    }
}

fn cond_two_norm_real(matrix: &Tensor, rows: usize, cols: usize) -> BuiltinResult<f64> {
    let a = DMatrix::from_column_slice(rows, cols, &matrix.data);
    let svd = SVD::new(a, false, false);
    Ok(singular_value_cond(svd.singular_values.as_slice()))
}

fn cond_two_norm_complex(matrix: &ComplexTensor, rows: usize, cols: usize) -> BuiltinResult<f64> {
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let a = DMatrix::from_column_slice(rows, cols, &data);
    let svd = SVD::new(a, false, false);
    Ok(singular_value_cond(svd.singular_values.as_slice()))
}

fn cond_inverse_based_real(matrix: &Tensor, order: usize, norm: CondNorm) -> BuiltinResult<f64> {
    let dm = DMatrix::from_column_slice(order, order, &matrix.data);
    if let Some(inv) = dm.try_inverse() {
        let norm_a = matrix_norm_real(matrix.data.as_slice(), order, order, norm);
        let norm_inv = matrix_norm_real(inv.as_slice(), order, order, norm);
        let cond = norm_a * norm_inv;
        if cond.is_finite() {
            Ok(cond)
        } else {
            Ok(f64::INFINITY)
        }
    } else {
        Ok(f64::INFINITY)
    }
}

fn cond_inverse_based_complex(
    matrix: &ComplexTensor,
    order: usize,
    norm: CondNorm,
) -> BuiltinResult<f64> {
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let dm = DMatrix::from_column_slice(order, order, &data);
    if let Some(inv) = dm.try_inverse() {
        let norm_a = matrix_norm_complex(&data, order, order, norm);
        let norm_inv = matrix_norm_complex(inv.as_slice(), order, order, norm);
        let cond = norm_a * norm_inv;
        if cond.is_finite() {
            Ok(cond)
        } else {
            Ok(f64::INFINITY)
        }
    } else {
        Ok(f64::INFINITY)
    }
}

fn matrix_norm_real(data: &[f64], rows: usize, cols: usize, norm: CondNorm) -> f64 {
    match norm {
        CondNorm::One => {
            let mut max_sum: f64 = 0.0;
            for c in 0..cols {
                let mut sum = 0.0;
                for r in 0..rows {
                    sum += data[r + c * rows].abs();
                }
                max_sum = max_sum.max(sum);
            }
            max_sum
        }
        CondNorm::Inf => {
            let mut max_sum: f64 = 0.0;
            for r in 0..rows {
                let mut sum = 0.0;
                for c in 0..cols {
                    sum += data[r + c * rows].abs();
                }
                max_sum = max_sum.max(sum);
            }
            max_sum
        }
        CondNorm::Fro => {
            let sum_sq: f64 = data.iter().map(|v| v * v).sum();
            sum_sq.sqrt()
        }
        CondNorm::Two => unreachable!("matrix_norm_real not used for 2-norm"),
    }
}

fn matrix_norm_complex(data: &[Complex64], rows: usize, cols: usize, norm: CondNorm) -> f64 {
    match norm {
        CondNorm::One => {
            let mut max_sum: f64 = 0.0;
            for c in 0..cols {
                let mut sum = 0.0;
                for r in 0..rows {
                    sum += data[r + c * rows].norm();
                }
                max_sum = max_sum.max(sum);
            }
            max_sum
        }
        CondNorm::Inf => {
            let mut max_sum: f64 = 0.0;
            for r in 0..rows {
                let mut sum = 0.0;
                for c in 0..cols {
                    sum += data[r + c * rows].norm();
                }
                max_sum = max_sum.max(sum);
            }
            max_sum
        }
        CondNorm::Fro => {
            let sum_sq: f64 = data.iter().map(|v| v.norm_sqr()).sum();
            sum_sq.sqrt()
        }
        CondNorm::Two => unreachable!("matrix_norm_complex not used for 2-norm"),
    }
}

fn singular_value_cond(singular_values: &[f64]) -> f64 {
    if singular_values.is_empty() {
        return 0.0;
    }
    let mut min_sv = f64::INFINITY;
    let mut max_sv = 0.0_f64;
    for &sv in singular_values {
        let abs = sv.abs();
        if !abs.is_finite() {
            return f64::INFINITY;
        }
        min_sv = min_sv.min(abs);
        max_sv = max_sv.max(abs);
    }
    if min_sv == 0.0 {
        f64::INFINITY
    } else {
        max_sv / min_sv
    }
}

fn parse_norm_argument(args: &[Value]) -> BuiltinResult<CondNorm> {
    match args.len() {
        0 => Ok(CondNorm::Two),
        1 => parse_norm_value(&args[0]),
        _ => Err(builtin_error(format!("{NAME}: too many input arguments"))),
    }
}

fn parse_norm_value(value: &Value) -> BuiltinResult<CondNorm> {
    if let Some(text) = tensor::value_to_string(value) {
        return parse_norm_string(&text);
    }
    match value {
        Value::Num(n) => parse_norm_numeric(*n),
        Value::Int(i) => parse_norm_numeric(i.to_f64()),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => parse_norm_numeric(t.data[0]),
        Value::Bool(b) => {
            if *b {
                Ok(CondNorm::One)
            } else {
                Err(builtin_error(format!("{NAME}: norm must be 1, 2, Inf, or 'fro'")))
            }
        }
        Value::LogicalArray(logical) if logical.len() == 1 => {
            if logical.data[0] != 0 {
                Ok(CondNorm::One)
            } else {
                Err(builtin_error(format!("{NAME}: norm must be 1, 2, Inf, or 'fro'")))
            }
        }
        _ => Err(builtin_error(format!("{NAME}: norm must be 1, 2, Inf, or 'fro'"))),
    }
}

fn parse_norm_numeric(raw: f64) -> BuiltinResult<CondNorm> {
    if raw == 1.0 {
        Ok(CondNorm::One)
    } else if raw == 2.0 {
        Ok(CondNorm::Two)
    } else if raw.is_infinite() && raw.is_sign_positive() {
        Ok(CondNorm::Inf)
    } else {
        Err(builtin_error(format!("{NAME}: norm must be 1, 2, Inf, or 'fro'")))
    }
}

fn parse_norm_string(text: &str) -> BuiltinResult<CondNorm> {
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "2" | "two" => Ok(CondNorm::Two),
        "1" | "one" => Ok(CondNorm::One),
        "inf" | "infinity" => Ok(CondNorm::Inf),
        "fro" | "frobenius" => Ok(CondNorm::Fro),
        _ => Err(builtin_error(format!("{NAME}: unrecognised norm '{text}'"))),
    }
}

fn upload_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: f64,
) -> BuiltinResult<GpuTensorHandle> {
    let data = [value];
    let shape = [1usize, 1usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

/// Helper for provider backends that reuse the host implementation.
pub fn cond_host_real_for_provider(
    matrix: &Tensor,
    norm: ProviderCondNorm,
) -> BuiltinResult<f64> {
    cond_real_tensor(matrix, CondNorm::from(norm))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CondNorm {
    Two,
    One,
    Inf,
    Fro,
}

impl From<CondNorm> for ProviderCondNorm {
    fn from(value: CondNorm) -> Self {
        match value {
            CondNorm::Two => ProviderCondNorm::Two,
            CondNorm::One => ProviderCondNorm::One,
            CondNorm::Inf => ProviderCondNorm::Inf,
            CondNorm::Fro => ProviderCondNorm::Fro,
        }
    }
}

impl From<ProviderCondNorm> for CondNorm {
    fn from(value: ProviderCondNorm) -> Self {
        match value {
            ProviderCondNorm::Two => CondNorm::Two,
            ProviderCondNorm::One => CondNorm::One,
            ProviderCondNorm::Inf => CondNorm::Inf,
            ProviderCondNorm::Fro => CondNorm::Fro,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_identity_is_one() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_zero_is_infinite() {
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert!(value.is_infinite()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_rectangular_two_norm() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 2.414213562).abs() < 1e-9),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_one_norm_matches_manual() {
        let tensor = Tensor::new(vec![4.0, 2.0, -1.0, 3.0], vec![2, 2]).unwrap();
        let result =
            cond_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))]).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 2.142_857_142_857_143).abs() < 1e-9),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_infinity_norm() {
        let tensor = Tensor::new(vec![4.0, 2.0, -1.0, 3.0], vec![2, 2]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), vec![Value::from("inf")]).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 2.142_857_142_857_143).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_frobenius_norm() {
        let tensor = Tensor::new(vec![5.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), vec![Value::from("fro")]).expect("cond");
        match result {
            Value::Num(value) => assert!((value - 2.9).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_complex_matrix_supported() {
        let data = vec![(1.0, 2.0), (0.0, 0.0), (0.0, 3.0), (2.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = cond_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert!(value.is_finite() && value >= 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_rejects_non_square_for_other_norms() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = unwrap_error(
            cond_builtin(Value::Tensor(tensor), vec![Value::from("inf")]).unwrap_err(),
        );
        assert_eq!(
            err.message(),
            "cond: matrix must be square for the requested norm."
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_empty_returns_zero() {
        let tensor = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = cond_builtin(Value::Tensor(tensor), Vec::new()).expect("cond");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cond_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = cond_builtin(Value::GpuTensor(handle), Vec::new()).expect("cond");
            let gathered = test_support::gather(gpu_value).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            let expected = cond_builtin(Value::Tensor(tensor.clone()), Vec::new())
                .map(|v| match v {
                    Value::Num(n) => n,
                    _ => unreachable!(),
                })
                .expect("cpu cond");
            assert!((gathered.data[0] - expected).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cond_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![2.0, 0.0, 0.0, 0.2], vec![2, 2]).unwrap();
        let cpu = cond_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu");
        let cpu_value = match cpu {
            Value::Num(n) => n,
            _ => unreachable!(),
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu = cond_builtin(Value::GpuTensor(handle), Vec::new()).expect("cond");
        let gathered = test_support::gather(gpu).expect("gather");
        assert!((gathered.data[0] - cpu_value).abs() < 1e-9);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
