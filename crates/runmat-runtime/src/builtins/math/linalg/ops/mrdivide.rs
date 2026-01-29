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
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "mrdivide";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::mrdivide")]
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

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::ops::mrdivide"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mrdivide",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Right-division is a terminal operation and does not fuse with surrounding kernels.",
};

#[runtime_builtin(
    name = "mrdivide",
    category = "math/linalg/ops",
    summary = "Solve X * B = A using MATLAB-compatible right division.",
    keywords = "mrdivide,matrix division,linear algebra,least squares,gpu",
    accel = "mrdivide",
    builtin_path = "crate::builtins::math::linalg::ops::mrdivide"
)]
async fn mrdivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    mrdivide_eval(&lhs, &rhs).await
}

pub(crate) async fn mrdivide_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
    if let Some(result) = try_gpu_mrdivide(lhs, rhs).await? {
        return Ok(result);
    }

    let lhs_host = crate::dispatcher::gather_if_needed_async(lhs)
        .await
        .map_err(map_control_flow)?;
    let rhs_host = crate::dispatcher::gather_if_needed_async(rhs)
        .await
        .map_err(map_control_flow)?;
    mrdivide_cpu(lhs_host, rhs_host)
}

async fn try_gpu_mrdivide(lhs: &Value, rhs: &Value) -> BuiltinResult<Option<Value>> {
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
        .await
        .ok();
    release_operand(provider, &mut lhs_operand);
    release_operand(provider, &mut rhs_operand);
    Ok(result.map(Value::GpuTensor))
}

fn mrdivide_cpu(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
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
pub fn mrdivide_host_real_for_provider(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
    mrdivide_real(lhs, rhs)
}

enum NumericInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_numeric(value: Value) -> BuiltinResult<NumericInput> {
    match value {
        Value::ComplexTensor(tensor) => {
            ensure_matrix_shape("mrdivide", &tensor.shape)?;
            Ok(NumericInput::Complex(tensor))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
            Ok(NumericInput::Complex(tensor))
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            ensure_matrix_shape(NAME, &tensor.shape)?;
            Ok(NumericInput::Real(tensor))
        }
    }
}

fn mrdivide_real(lhs: &Tensor, rhs: &Tensor) -> BuiltinResult<Tensor> {
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
            .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
        return Ok(result);
    }

    let lhs_matrix = DMatrix::from_column_slice(lhs.rows(), lhs.cols(), &lhs.data);
    let rhs_matrix = DMatrix::from_column_slice(rhs.rows(), rhs.cols(), &rhs.data);
    let solution = solve_real_matrix(&lhs_matrix, &rhs_matrix)?;
    matrix_real_to_tensor(solution)
}

fn mrdivide_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> BuiltinResult<ComplexTensor> {
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
            .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
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

fn solve_real_matrix(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> BuiltinResult<DMatrix<f64>> {
    let rhs_t = rhs.transpose();
    let lhs_t = lhs.transpose();
    let svd = SVD::new(rhs_t.clone(), true, true);
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), rhs_t.nrows(), rhs_t.ncols());
    let solved = svd
        .solve(&lhs_t, tol)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    Ok(solved.transpose())
}

fn solve_complex_matrix(
    lhs: &DMatrix<Complex64>,
    rhs: &DMatrix<Complex64>,
) -> BuiltinResult<DMatrix<Complex64>> {
    let rhs_t = rhs.transpose();
    let lhs_t = lhs.transpose();
    let svd = SVD::new(rhs_t.clone(), true, true);
    let tol = compute_svd_tolerance(svd.singular_values.as_slice(), rhs_t.nrows(), rhs_t.ncols());
    let solved = svd
        .solve(&lhs_t, tol)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
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

fn matrix_real_to_tensor(matrix: DMatrix<f64>) -> BuiltinResult<Tensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    Tensor::new(matrix.as_slice().to_vec(), vec![rows, cols])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn matrix_complex_to_tensor(matrix: DMatrix<Complex64>) -> BuiltinResult<ComplexTensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<(f64, f64)> = matrix.as_slice().iter().map(|c| (c.re, c.im)).collect();
    ComplexTensor::new(data, vec![rows, cols]).map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn promote_real_tensor(tensor: &Tensor) -> BuiltinResult<ComplexTensor> {
    let data: Vec<(f64, f64)> = tensor.data.iter().map(|&re| (re, 0.0)).collect();
    ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn ensure_matrix_shape(name: &str, shape: &[usize]) -> BuiltinResult<()> {
    if is_effectively_matrix(shape) {
        Ok(())
    } else {
        Err(builtin_error(format!(
            "{name}: inputs must be 2-D matrices or vectors"
        )))
    }
}

fn ensure_column_match(lhs_cols: usize, rhs_cols: usize) -> BuiltinResult<()> {
    if lhs_cols == rhs_cols {
        Ok(())
    } else {
        Err(builtin_error("Matrix dimensions must agree."))
    }
}

fn is_effectively_matrix(shape: &[usize]) -> bool {
    match shape.len() {
        0..=2 => true,
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
) -> BuiltinResult<Option<PreparedOperand>> {
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
                let tensor = tensor::logical_to_tensor(logical).map_err(builtin_error)?;
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
) -> BuiltinResult<GpuTensorHandle> {
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn release_operand(provider: &'static dyn AccelProvider, operand: &mut PreparedOperand) {
    if operand.owned {
        let _ = provider.free(&operand.handle);
        operand.owned = false;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn divides_scalar_by_scalar() {
        let result = mrdivide_builtin(Value::Num(6.0), Value::Num(2.0)).expect("mrdivide");
        match result {
            Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn divides_matrix_by_scalar() {
        let tensor = Tensor::new(vec![2.0, 4.0, 6.0], vec![1, 3]).expect("tensor");
        let result = mrdivide_builtin(Value::Tensor(tensor), Value::Num(2.0)).expect("mrdivide");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, vec![1.0, 2.0, 3.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reports_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = unwrap_error(mrdivide_builtin(Value::Tensor(a), Value::Tensor(b)).unwrap_err());
        assert!(
            err.message().contains("Matrix dimensions must agree"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    fn mrdivide_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(super::mrdivide_builtin(lhs, rhs))
    }

    fn mrdivide_eval(lhs: &Value, rhs: &Value) -> BuiltinResult<Value> {
        block_on(super::mrdivide_eval(lhs, rhs))
    }
}
