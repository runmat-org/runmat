//! MATLAB-compatible `inv` builtin with GPU-aware fallbacks.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, ProviderInvOptions};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "inv";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::inv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("inv"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("inv")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement a native inverse; the reference WGPU backend gathers to the host implementation and re-uploads the result.",
};

fn builtin_error(message: String) -> RuntimeError {
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::inv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Matrix inversion is a terminal operation and does not participate in fusion pipelines.",
};

#[runtime_builtin(
    name = "inv",
    category = "math/linalg/solve",
    summary = "Compute the inverse of a square matrix.",
    keywords = "inv,matrix inverse,linear solve,gpu",
    accel = "inv",
    builtin_path = "crate::builtins::math::linalg::solve::inv"
)]
async fn inv_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => inv_gpu(handle).await,
        Value::ComplexTensor(tensor) => inv_complex_value(tensor),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            inv_complex_value(tensor)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            inv_real_value(tensor)
        }
    }
}

async fn inv_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let options = ProviderInvOptions::default();
        match provider.inv(&handle, options).await {
            Ok(result) => return Ok(Value::GpuTensor(result)),
            Err(_) => {
                // Fall back to host implementation and attempt to re-upload.
            }
        }
        let gathered = gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(map_control_flow)?;
        let inv = inv_real_tensor(&gathered)?;
        if let Ok(uploaded) = provider.upload(&runmat_accelerate_api::HostTensorView {
            data: &inv.data,
            shape: &inv.shape,
        }) {
            return Ok(Value::GpuTensor(uploaded));
        }
        return Ok(tensor::tensor_into_value(inv));
    }

    let gathered = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(map_control_flow)?;
    let inv = inv_real_tensor(&gathered)?;
    Ok(tensor::tensor_into_value(inv))
}

fn inv_real_value(tensor: Tensor) -> BuiltinResult<Value> {
    let inv = inv_real_tensor(&tensor)?;
    Ok(tensor::tensor_into_value(inv))
}

fn inv_complex_value(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let inv = inv_complex_tensor(&tensor)?;
    if inv.data.len() == 1 {
        let (re, im) = inv.data[0];
        Ok(Value::Complex(re, im))
    } else {
        Ok(Value::ComplexTensor(inv))
    }
}

fn inv_real_tensor(matrix: &Tensor) -> BuiltinResult<Tensor> {
    inv_real_tensor_impl(matrix)
}

fn inv_complex_tensor(matrix: &ComplexTensor) -> BuiltinResult<ComplexTensor> {
    inv_complex_tensor_impl(matrix)
}

fn inv_real_tensor_impl(matrix: &Tensor) -> BuiltinResult<Tensor> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows == 0 && cols == 0 {
        return Tensor::new(Vec::new(), matrix.shape.clone())
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 || cols == 0 {
        return Tensor::new(Vec::new(), matrix.shape.clone())
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let dm = DMatrix::from_column_slice(rows, cols, &matrix.data);
    let inverse = dm.try_inverse().ok_or_else(|| {
        builtin_error(format!("{NAME}: matrix is singular to working precision."))
    })?;
    matrix_to_tensor(NAME, inverse, &matrix.shape)
}

fn inv_complex_tensor_impl(matrix: &ComplexTensor) -> BuiltinResult<ComplexTensor> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows == 0 && cols == 0 {
        return ComplexTensor::new(Vec::new(), matrix.shape.clone())
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 || cols == 0 {
        return ComplexTensor::new(Vec::new(), matrix.shape.clone())
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let dm = DMatrix::from_column_slice(rows, cols, &data);
    let inverse = dm.try_inverse().ok_or_else(|| {
        builtin_error(format!("{NAME}: matrix is singular to working precision."))
    })?;
    matrix_to_complex_tensor(NAME, inverse, &matrix.shape)
}

fn matrix_dimensions(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => {
            if shape[0] == 1 {
                Ok((1, 1))
            } else {
                Err(builtin_error(format!(
                    "{NAME}: input must be a square matrix."
                )))
            }
        }
        _ => {
            if shape.len() > 2 && shape.iter().skip(2).any(|&dim| dim != 1) {
                Err(builtin_error(format!(
                    "{NAME}: inputs must be 2-D matrices."
                )))
            } else {
                Ok((shape[0], shape[1]))
            }
        }
    }
}

fn matrix_to_tensor(label: &str, matrix: DMatrix<f64>, shape: &[usize]) -> BuiltinResult<Tensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    debug_assert_eq!(rows * cols, matrix.len());
    Tensor::new(matrix.as_slice().to_vec(), shape.to_vec())
        .map_err(|e| builtin_error(format!("{label}: {e}")))
}

fn matrix_to_complex_tensor(
    label: &str,
    matrix: DMatrix<Complex64>,
    shape: &[usize],
) -> BuiltinResult<ComplexTensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<(f64, f64)> = matrix.as_slice().iter().map(|c| (c.re, c.im)).collect();
    debug_assert_eq!(rows * cols, matrix.len());
    ComplexTensor::new(data, shape.to_vec()).map_err(|e| builtin_error(format!("{label}: {e}")))
}

/// Host helper used by acceleration providers that delegate `inv` back to the CPU path.
pub fn inv_host_real_for_provider(matrix: &Tensor) -> BuiltinResult<Tensor> {
    inv_real_tensor_impl(matrix)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::IntValue;
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_scalar_num() {
        let result = inv_builtin(Value::Num(4.0)).expect("inv");
        match result {
            Value::Num(v) => assert!((v - 0.25).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_square_matrix() {
        let data = vec![4.0, 1.0, -2.0, 3.0];
        let tensor = Tensor::new(data.clone(), vec![2, 2]).unwrap();
        let result = inv_builtin(Value::Tensor(tensor)).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let a = DMatrix::from_column_slice(2, 2, &data);
                let inv_m = DMatrix::from_column_slice(2, 2, &out.data);
                let identity = &a * &inv_m;
                for r in 0..2 {
                    for c in 0..2 {
                        let expected = if r == c { 1.0 } else { 0.0 };
                        assert!((identity[(r, c)] - expected).abs() < 1e-12);
                    }
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_empty_matrix_returns_empty() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = inv_builtin(Value::Tensor(tensor.clone())).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert!(out.data.is_empty());
                assert_eq!(out.shape, vec![0, 0]);
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_trailing_singleton_dimension_preserved() {
        let tensor =
            Tensor::new(vec![4.0, 0.0, 0.0, 2.0], vec![2, 2, 1]).expect("tensor construction");
        let result = inv_builtin(Value::Tensor(tensor)).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2, 1]);
                let expected = vec![0.25, 0.0, 0.0, 0.5];
                assert_eq!(out.data, expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_complex_scalar() {
        let result = inv_builtin(Value::Complex(2.0, -1.0)).expect("inv");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.0, 0.0) / Complex64::new(2.0, -1.0);
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_complex_matrix() {
        let raw = vec![(1.0, 2.0), (0.0, 3.0), (0.0, 0.0), (4.0, -1.0)];
        let tensor = ComplexTensor::new(raw.clone(), vec![2, 2]).unwrap();
        let result = inv_builtin(Value::ComplexTensor(tensor)).expect("inv");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let input: Vec<Complex64> =
                    raw.iter().map(|&(re, im)| Complex64::new(re, im)).collect();
                let inv_vec: Vec<Complex64> = out
                    .data
                    .iter()
                    .map(|&(re, im)| Complex64::new(re, im))
                    .collect();
                let a = DMatrix::from_column_slice(2, 2, &input);
                let inv_m = DMatrix::from_column_slice(2, 2, &inv_vec);
                let identity = &a * &inv_m;
                for r in 0..2 {
                    for c in 0..2 {
                        let expected = if r == c {
                            Complex64::new(1.0, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        };
                        let delta = identity[(r, c)] - expected;
                        assert!(delta.norm() < 1e-10, "identity mismatch at ({r},{c})");
                    }
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_rejects_higher_rank_tensor() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = unwrap_error(inv_builtin(Value::Tensor(tensor)).unwrap_err());
        assert!(err.message().contains("2-D"), "{err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_non_square_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = unwrap_error(inv_builtin(Value::Tensor(tensor)).unwrap_err());
        assert!(err.message().contains("square matrix"), "{err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_singular_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let err = unwrap_error(inv_builtin(Value::Tensor(tensor)).unwrap_err());
        assert!(err.message().contains("singular"), "{err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = inv_builtin(Value::GpuTensor(handle)).expect("gpu inv");
            let gathered = test_support::gather(gpu_value).expect("gather");
            let cpu = inv_real_tensor(&tensor).expect("cpu");
            assert_eq!(gathered.shape, cpu.shape);
            for (a, b) in gathered.data.iter().zip(cpu.data.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_scalar_int_promotes() {
        let result = inv_builtin(Value::Int(IntValue::I32(2))).expect("inv");
        match result {
            Value::Num(v) => assert!((v - 0.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn inv_wgpu_matches_cpu() {
        let _ = provider::register_wgpu_provider(WgpuProviderOptions::default());

        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let cpu = inv_real_tensor(&tensor).expect("cpu");

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = inv_builtin(Value::GpuTensor(handle)).expect("gpu inv");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gathered.shape, cpu.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered.data.iter().zip(cpu.data.iter()) {
            assert!((*a - *b).abs() < tol, "expected {b}, got {a}");
        }
    }

    fn inv_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::inv_builtin(value))
    }
}
