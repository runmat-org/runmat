//! MATLAB-compatible `mrdivide` builtin (`/`) for solving right-sided linear systems.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::linalg;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

const NAME: &str = "mrdivide";

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "mrdivide"
category: "math/linalg/ops"
keywords: ["mrdivide", "matrix right division", "linear systems", "least squares", "gpu"]
summary: "Solve X * B = A using MATLAB's right-division operator (`/`)."
references: ["https://www.mathworks.com/help/matlab/ref/double.mrdivide.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Prefers the accel provider's mrdivide hook; providers that lack it (including the current WGPU backend) gather to host, run the shared solver, then re-upload the result to keep residency transparent."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "uniform"
requires_feature: null
tested:
  unit: "builtins::math::linalg::ops::mrdivide::tests"
  gpu: "builtins::math::linalg::ops::mrdivide::tests::gpu_round_trip_matches_cpu"
  wgpu: "builtins::math::linalg::ops::mrdivide::tests::wgpu_round_trip_matches_cpu"
  doc: "builtins::math::linalg::ops::mrdivide::tests::doc_examples_present"
---

# What does the `mrdivide` function do in MATLAB / RunMat?
`X = A / B` (or `mrdivide(A, B)`) solves the right-sided linear system `X * B = A`. When `B` is
square and nonsingular the solution matches `A * inv(B)`. Rectangular or rank-deficient matrices are
handled via a minimum-norm least-squares solve, matching MATLAB's SVD-based semantics.

## How does the `mrdivide` function behave in MATLAB / RunMat?
- Scalars divide exactly: `A / s` scales `A` by `1/s`, while `s / B` requires `B` to be scalar.
- Logical and integer inputs are promoted to double precision before solving.
- Purely real operands produce real outputs; any complex operand promotes the computation (and result)
  to complex arithmetic.
- Inputs must be effectively two-dimensional; trailing singleton dimensions are allowed.
- The number of columns must agree (`size(A, 2) == size(B, 2)`), otherwise RunMat raises the MATLAB
  error `"Matrix dimensions must agree."`
- Underdetermined and overdetermined systems return the minimum-norm least-squares solution.

## `mrdivide` GPU execution behaviour
When a gpuArray provider is active, RunMat first offers the solve to its `mrdivide` hook. The WGPU
provider currently downloads the operands to the host, executes the same SVD-based solver used by the
CPU implementation, then uploads the result back to the device so residency remains transparent. If
no provider is available—or the provider declines the request—RunMat gathers any gpuArray inputs to
the host, computes the solution, and returns a host tensor.

## Examples of using the `mrdivide` function in MATLAB / RunMat

### Solving a square linear system

```matlab
A = [1 2; 3 4];
B = [5 6; 7 8];
X = A / B;

% Verify the solution
residual = X * B;
```

Expected output:

```matlab
X =
     3    -2
     2    -1

residual =
     1     2
     3     4
```

### Computing a least-squares right division

```matlab
A = [1 2 3];
B = [1 0 1; 0 1 1];
X = A / B;
```

Expected output:

```matlab
X = [1 2];
```

### Dividing by a scalar

```matlab
A = [2 4 6];
scaled = A / 2;
```

Expected output:

```matlab
scaled = [1 2 3];
```

### Right division with complex inputs

```matlab
A = [1+2i 3-4i];
B = [2-i 1+i];
X = A / B;
```

Expected output:

```matlab
X = -0.1429 - 0.2857i
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No manual care is required. If both operands already reside on the GPU and the provider supports
`mrdivide`, the solve stays on the device. When the provider falls back to the host (the current WGPU
implementation), the runtime seamlessly gathers data, executes the solve, and re-uploads the result
to keep downstream GPU pipelines working as expected.

## FAQ

### Why must `A` and `B` share the number of columns?
Right division solves `X * B = A`; matrix multiplication requires `size(A, 2) == size(B, 2)`.

### What happens if `B` is singular or rectangular?
RunMat matches MATLAB by computing the minimum-norm least-squares solution via singular-value
decomposition—no explicit call to `pinv` is required.

### Does `mrdivide` support higher-dimensional arrays?
No. Inputs must be effectively matrices (trailing singleton dimensions are allowed). Use `reshape`
or `(:)` to flatten higher-dimensional data before calling `mrdivide`.

### How are logical or integer arrays handled?
They are promoted to double precision (`true → 1`, `false → 0`) before solving, matching MATLAB
semantics.

### How does RunMat handle NaN or Inf values?
They propagate through the least-squares solve in the same way as MATLAB. NaNs in the inputs yield
NaNs in the output wherever they influence the solution.

## See Also
[mtimes](./mtimes), [svd](../../factor/svd), [lu](../../factor/lu), [gpuArray](../../../acceleration/gpu/gpuArray), [gather](../../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/math/linalg/ops/mrdivide.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/mrdivide.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mrdivide",
    op_kind: GpuOpKind::Custom("solve"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("mrdivide")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prefers the provider `mrdivide` hook; the WGPU provider currently performs the solve on the host and re-uploads the result.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mrdivide",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Right-division is a terminal operation and does not fuse with surrounding kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("mrdivide", DOC_MD);

#[runtime_builtin(
    name = "mrdivide",
    category = "math/linalg/ops",
    summary = "Solve X * B = A using MATLAB-compatible right division.",
    keywords = "mrdivide,matrix division,linear algebra,least squares,gpu",
    accel = "mrdivide"
)]
fn mrdivide_builtin(lhs: Value, rhs: Value) -> Result<Value, String> {
    mrdivide_eval(&lhs, &rhs)
}

pub(crate) fn mrdivide_eval(lhs: &Value, rhs: &Value) -> Result<Value, String> {
    if let Some(result) = try_gpu_mrdivide(lhs, rhs)? {
        return Ok(result);
    }
    let lhs_host = crate::dispatcher::gather_if_needed(lhs)?;
    let rhs_host = crate::dispatcher::gather_if_needed(rhs)?;
    mrdivide_cpu(lhs_host, rhs_host)
}

fn try_gpu_mrdivide(lhs: &Value, rhs: &Value) -> Result<Option<Value>, String> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    if contains_complex(lhs) || contains_complex(rhs) {
        return Ok(None);
    }

    let mut lhs_operand = match prepare_gpu_operand(lhs, provider)? {
        Some(op) => op,
        None => return Ok(None),
    };
    let mut rhs_operand = match prepare_gpu_operand(rhs, provider)? {
        Some(op) => op,
        None => {
            release_operand(provider, &mut lhs_operand);
            return Ok(None);
        }
    };

    if is_scalar_handle(rhs_operand.handle()) {
        release_operand(provider, &mut lhs_operand);
        release_operand(provider, &mut rhs_operand);
        return Ok(None);
    }

    let result = provider
        .mrdivide(lhs_operand.handle(), rhs_operand.handle())
        .ok();
    release_operand(provider, &mut lhs_operand);
    release_operand(provider, &mut rhs_operand);
    Ok(result.map(Value::GpuTensor))
}

fn mrdivide_cpu(lhs: Value, rhs: Value) -> Result<Value, String> {
    let lhs_numeric = classify_numeric(lhs)?;
    let rhs_numeric = classify_numeric(rhs)?;

    match (lhs_numeric, rhs_numeric) {
        (NumericInput::Real(lhs_r), NumericInput::Real(rhs_r)) => {
            let result = mrdivide_real(&lhs_r, &rhs_r)?;
            Ok(tensor::tensor_into_value(result))
        }
        (NumericInput::Complex(lhs_c), NumericInput::Complex(rhs_c)) => {
            let result = mrdivide_complex(&lhs_c, &rhs_c)?;
            Ok(complex_tensor_into_value(result))
        }
        (NumericInput::Complex(lhs_c), NumericInput::Real(rhs_r)) => {
            let rhs_c = promote_real_tensor(&rhs_r)?;
            let result = mrdivide_complex(&lhs_c, &rhs_c)?;
            Ok(complex_tensor_into_value(result))
        }
        (NumericInput::Real(lhs_r), NumericInput::Complex(rhs_c)) => {
            let lhs_c = promote_real_tensor(&lhs_r)?;
            let result = mrdivide_complex(&lhs_c, &rhs_c)?;
            Ok(complex_tensor_into_value(result))
        }
    }
}

/// Host implementation shared with acceleration providers that keep data on the CPU.
pub fn mrdivide_host_real_for_provider(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, String> {
    mrdivide_real(lhs, rhs)
}

enum NumericInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_numeric(value: Value) -> Result<NumericInput, String> {
    match value {
        Value::ComplexTensor(tensor) => {
            ensure_matrix_shape("mrdivide", &tensor.shape)?;
            Ok(NumericInput::Complex(tensor))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("{NAME}: {e}"))?;
            Ok(NumericInput::Complex(tensor))
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other)?;
            ensure_matrix_shape(NAME, &tensor.shape)?;
            Ok(NumericInput::Real(tensor))
        }
    }
}

fn mrdivide_real(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, String> {
    ensure_matrix_shape(NAME, &lhs.shape)?;
    ensure_matrix_shape(NAME, &rhs.shape)?;

    if tensor::is_scalar_tensor(rhs) {
        let divisor = rhs.data[0];
        let scaled = linalg::scalar_mul_real(lhs, divisor.recip());
        return Ok(scaled);
    }

    ensure_column_match(lhs.cols(), rhs.cols())?;

    if rhs.cols() == 0 {
        let rows = lhs.rows();
        let cols = rhs.rows();
        let result = Tensor::new(vec![0.0; rows * cols], vec![rows, cols])
            .map_err(|e| format!("{NAME}: {e}"))?;
        return Ok(result);
    }

    let lhs_matrix = DMatrix::from_column_slice(lhs.rows(), lhs.cols(), &lhs.data);
    let rhs_matrix = DMatrix::from_column_slice(rhs.rows(), rhs.cols(), &rhs.data);
    let solution = solve_real_matrix(&lhs_matrix, &rhs_matrix)?;
    matrix_real_to_tensor(solution)
}

fn mrdivide_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> Result<ComplexTensor, String> {
    ensure_matrix_shape(NAME, &lhs.shape)?;
    ensure_matrix_shape(NAME, &rhs.shape)?;

    if complex_tensor_is_scalar(rhs) {
        let divisor = Complex64::new(rhs.data[0].0, rhs.data[0].1);
        let inv = Complex64::new(1.0, 0.0) / divisor;
        let scaled = linalg::scalar_mul_complex_tensor(lhs, inv.re, inv.im);
        return Ok(scaled);
    }

    ensure_column_match(lhs.cols, rhs.cols)?;

    if rhs.cols == 0 {
        let rows = lhs.rows;
        let cols = rhs.rows;
        let result = ComplexTensor::new(vec![(0.0, 0.0); rows * cols], vec![rows, cols])
            .map_err(|e| format!("{NAME}: {e}"))?;
        return Ok(result);
    }

    let lhs_data: Vec<Complex64> = lhs
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let rhs_data: Vec<Complex64> = rhs
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let lhs_matrix = DMatrix::from_column_slice(lhs.rows, lhs.cols, &lhs_data);
    let rhs_matrix = DMatrix::from_column_slice(rhs.rows, rhs.cols, &rhs_data);
    let solution = solve_complex_matrix(&lhs_matrix, &rhs_matrix)?;
    matrix_complex_to_tensor(solution)
}

fn solve_real_matrix(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let rhs_t = rhs.transpose();
    let lhs_t = lhs.transpose();
    let svd = SVD::new(rhs_t.clone(), true, true);
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), rhs_t.nrows(), rhs_t.ncols());
    let solved = svd.solve(&lhs_t, tol).map_err(|e| format!("{NAME}: {e}"))?;
    Ok(solved.transpose())
}

fn solve_complex_matrix(
    lhs: &DMatrix<Complex64>,
    rhs: &DMatrix<Complex64>,
) -> Result<DMatrix<Complex64>, String> {
    let rhs_t = rhs.transpose();
    let lhs_t = lhs.transpose();
    let svd = SVD::new(rhs_t.clone(), true, true);
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), rhs_t.nrows(), rhs_t.ncols());
    let solved = svd.solve(&lhs_t, tol).map_err(|e| format!("{NAME}: {e}"))?;
    Ok(solved.transpose())
}

fn compute_svd_tolerance(singular_values: &[f64], rows: usize, cols: usize) -> f64 {
    let max_sv = singular_values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_dim = rows.max(cols) as f64;
    f64::EPSILON * max_dim * max_sv.max(1.0)
}

fn matrix_real_to_tensor(matrix: DMatrix<f64>) -> Result<Tensor, String> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    Tensor::new(matrix.as_slice().to_vec(), vec![rows, cols]).map_err(|e| format!("{NAME}: {e}"))
}

fn matrix_complex_to_tensor(matrix: DMatrix<Complex64>) -> Result<ComplexTensor, String> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<(f64, f64)> = matrix.as_slice().iter().map(|c| (c.re, c.im)).collect();
    ComplexTensor::new(data, vec![rows, cols]).map_err(|e| format!("{NAME}: {e}"))
}

fn promote_real_tensor(tensor: &Tensor) -> Result<ComplexTensor, String> {
    let data: Vec<(f64, f64)> = tensor.data.iter().map(|&re| (re, 0.0)).collect();
    ComplexTensor::new(data, tensor.shape.clone()).map_err(|e| format!("{NAME}: {e}"))
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> Result<(), String> {
    if is_effectively_matrix(shape) {
        Ok(())
    } else {
        Err(format!("{name}: inputs must be 2-D matrices or vectors"))
    }
}

fn ensure_column_match(lhs_cols: usize, rhs_cols: usize) -> Result<(), String> {
    if lhs_cols == rhs_cols {
        Ok(())
    } else {
        Err("Matrix dimensions must agree.".to_string())
    }
}

fn is_effectively_matrix(shape: &[usize]) -> bool {
    match shape.len() {
        0 | 1 | 2 => true,
        _ => shape.iter().skip(2).all(|&dim| dim == 1),
    }
}

fn contains_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

fn complex_tensor_is_scalar(tensor: &ComplexTensor) -> bool {
    tensor.data.len() == 1
}

fn is_scalar_handle(handle: &GpuTensorHandle) -> bool {
    handle.shape.iter().copied().product::<usize>() == 1
}

struct PreparedOperand {
    handle: GpuTensorHandle,
    owned: bool,
}

impl PreparedOperand {
    fn borrowed(handle: &GpuTensorHandle) -> Self {
        Self {
            handle: handle.clone(),
            owned: false,
        }
    }

    fn owned(handle: GpuTensorHandle) -> Self {
        Self {
            handle,
            owned: true,
        }
    }

    fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }
}

fn prepare_gpu_operand(
    value: &Value,
    provider: &'static dyn AccelProvider,
) -> Result<Option<PreparedOperand>, String> {
    match value {
        Value::GpuTensor(handle) => {
            if is_scalar_handle(handle) {
                Ok(None)
            } else {
                Ok(Some(PreparedOperand::borrowed(handle)))
            }
        }
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                Ok(None)
            } else {
                let uploaded = upload_tensor(provider, tensor)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(None)
            } else {
                let tensor = tensor::logical_to_tensor(logical)?;
                let uploaded = upload_tensor(provider, &tensor)?;
                Ok(Some(PreparedOperand::owned(uploaded)))
            }
        }
        _ => Ok(None),
    }
}

fn upload_tensor(
    provider: &'static dyn AccelProvider,
    tensor: &Tensor,
) -> Result<GpuTensorHandle, String> {
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    provider.upload(&view).map_err(|e| format!("{NAME}: {e}"))
}

fn release_operand(provider: &'static dyn AccelProvider, operand: &mut PreparedOperand) {
    if operand.owned {
        let _ = provider.free(&operand.handle);
        operand.owned = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    #[test]
    fn divides_scalar_by_scalar() {
        let result = mrdivide_builtin(Value::Num(6.0), Value::Num(2.0)).expect("mrdivide");
        match result {
            Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn divides_matrix_by_scalar() {
        let tensor = Tensor::new(vec![2.0, 4.0, 6.0], vec![1, 3]).expect("tensor");
        let result = mrdivide_builtin(Value::Tensor(tensor), Value::Num(2.0)).expect("mrdivide");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, vec![1.0, 2.0, 3.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn solves_square_system() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result = mrdivide_builtin(Value::Tensor(a), Value::Tensor(b)).expect("mrdivide");
        let gathered = test_support::gather(result).expect("gather");
        let expected = vec![3.0, 2.0, -2.0, -1.0];
        assert_eq!(gathered.shape, vec![2, 2]);
        for (val, exp) in gathered.data.iter().zip(expected.into_iter()) {
            assert!((val - exp).abs() < 1e-12);
        }
    }

    #[test]
    fn solves_least_squares() {
        let a = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let result = mrdivide_builtin(Value::Tensor(a), Value::Tensor(b)).expect("mrdivide");
        let gathered = test_support::gather(result).expect("gather");
        let expected = vec![1.0, 3.0, 2.0, 4.0];
        assert_eq!(gathered.shape, vec![2, 2]);
        for (val, exp) in gathered.data.iter().zip(expected.into_iter()) {
            assert!((val - exp).abs() < 1e-10);
        }
    }

    #[test]
    fn supports_complex_inputs() {
        let a = ComplexTensor::new(
            vec![(1.0, 2.0), (5.0, 6.0), (3.0, -4.0), (7.0, -2.0)],
            vec![2, 2],
        )
        .unwrap();
        let b = ComplexTensor::new(
            vec![(2.0, -1.0), (1.0, 0.5), (0.5, 1.0), (3.0, 2.0)],
            vec![2, 2],
        )
        .unwrap();
        let result =
            mrdivide_builtin(Value::ComplexTensor(a), Value::ComplexTensor(b)).expect("mrdivide");
        match result {
            Value::ComplexTensor(out) => {
                let expected = [
                    (-0.7902439, 1.28780488),
                    (-0.72780488, 3.2897561),
                    (0.48780488, -1.6097561),
                    (2.0097561, -2.31219512),
                ];
                for (value, (er, ei)) in out.data.iter().zip(expected.into_iter()) {
                    let (vr, vi) = *value;
                    assert!((vr - er).abs() < 1e-6);
                    assert!((vi - ei).abs() < 1e-6);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn reports_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = mrdivide_builtin(Value::Tensor(a), Value::Tensor(b)).unwrap_err();
        assert!(
            err.contains("Matrix dimensions must agree"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

            let cpu = mrdivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone()))
                .expect("cpu mrdivide");
            let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

            let view_a = HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.data,
                shape: &b.shape,
            };
            let ha = provider.upload(&view_a).expect("upload A");
            let hb = provider.upload(&view_b).expect("upload B");
            let result =
                mrdivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
                    .expect("gpu mrdivide");
            let gathered = test_support::gather(result).expect("gather");
            let _ = provider.free(&ha);
            let _ = provider.free(&hb);

            assert_eq!(gathered.shape, cpu_tensor.shape);
            for (gpu, cpu) in gathered.data.iter().zip(cpu_tensor.data.iter()) {
                assert!((gpu - cpu).abs() < 1e-12);
            }
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_round_trip_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = match runmat_accelerate_api::provider() {
            Some(p) => p,
            None => panic!("wgpu provider not available"),
        };

        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let cpu = mrdivide_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone()))
            .expect("cpu mrdivide");
        let cpu_tensor = test_support::gather(cpu).expect("cpu gather");

        let view_a = HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload A");
        let hb = provider.upload(&view_b).expect("upload B");
        let gpu_value = mrdivide_eval(&Value::GpuTensor(ha.clone()), &Value::GpuTensor(hb.clone()))
            .expect("gpu mrdivide");
        let gathered = test_support::gather(gpu_value).expect("gather");
        let _ = provider.free(&ha);
        let _ = provider.free(&hb);

        assert_eq!(gathered.shape, cpu_tensor.shape);
        for (gpu, cpu) in gathered.data.iter().zip(cpu_tensor.data.iter()) {
            assert!((gpu - cpu).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
