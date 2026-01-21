//! MATLAB-compatible `rcond` builtin with GPU-aware fallbacks.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::linalg::{matrix_dimensions_for, singular_value_rcond};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const NAME: &str = "rcond";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::rcond")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rcond"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("rcond")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may reuse dense solver factorizations to expose rcond; current backends gather to the host and re-upload a scalar value when possible.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::rcond")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fusible; rcond consumes an entire matrix and returns a scalar estimate.",
};

#[runtime_builtin(
    name = "rcond",
    category = "math/linalg/solve",
    summary = "Estimate the reciprocal condition number of a square matrix.",
    keywords = "rcond,condition number,reciprocal,gpu",
    accel = "rcond",
    builtin_path = "crate::builtins::math::linalg::solve::rcond"
)]
fn rcond_builtin(value: Value) -> Result<Value, String> {
    let estimate = match value {
        Value::GpuTensor(handle) => return rcond_gpu(handle),
        Value::ComplexTensor(matrix) => rcond_complex_tensor(&matrix)?,
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("{NAME}: {e}"))?;
            rcond_complex_tensor(&tensor)?
        }
        other => {
            let matrix = tensor::value_into_tensor_for(NAME, other)?;
            rcond_real_tensor(&matrix)?
        }
    };
    Ok(Value::Num(estimate))
}

fn rcond_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    let (rows, cols) = matrix_dimensions_for(NAME, &handle.shape)?;
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
    }

    if rows == 0 {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(uploaded) = upload_scalar(provider, f64::INFINITY) {
                return Ok(Value::GpuTensor(uploaded));
            }
        }
        return Ok(Value::Num(f64::INFINITY));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.rcond(&handle) {
            Ok(result) => return Ok(Value::GpuTensor(result)),
            Err(_) => {
                if let Ok(Some(value)) = rcond_gpu_via_linsolve(provider, &handle, rows) {
                    return Ok(value);
                }
            }
        }
    }

    let gathered = gpu_helpers::gather_value(&Value::GpuTensor(handle.clone()))
        .map_err(|e| format!("{NAME}: {e}"))?;
    let estimate = match gathered {
        Value::Tensor(tensor) => rcond_real_tensor(&tensor)?,
        Value::ComplexTensor(tensor) => rcond_complex_tensor(&tensor)?,
        Value::Num(n) => {
            if n == 0.0 {
                0.0
            } else {
                1.0
            }
        }
        Value::Complex(re, im) => {
            if re == 0.0 && im == 0.0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other)?;
            rcond_real_tensor(&tensor)?
        }
    };

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(uploaded) = upload_scalar(provider, estimate) {
            return Ok(Value::GpuTensor(uploaded));
        }
    }

    Ok(Value::Num(estimate))
}

fn rcond_gpu_via_linsolve(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    order: usize,
) -> Result<Option<Value>, String> {
    if order == 0 {
        return upload_scalar(provider, f64::INFINITY).map(|gpu| Some(Value::GpuTensor(gpu)));
    }

    // Attempt to reuse provider linsolve to retrieve an rcond estimate.
    let identity = match upload_identity(provider, order) {
        Ok(id) => id,
        Err(_) => return Ok(None),
    };

    let options = runmat_accelerate_api::ProviderLinsolveOptions {
        rcond: None,
        ..Default::default()
    };
    let outcome = provider.linsolve(handle, &identity, &options);

    let _ = provider.free(&identity);

    let result = match outcome {
        Ok(res) => res,
        Err(_) => return Ok(None),
    };

    let rcond_value = result.reciprocal_condition;
    let _ = provider.free(&result.solution);

    if let Ok(uploaded) = upload_scalar(provider, rcond_value) {
        return Ok(Some(Value::GpuTensor(uploaded)));
    }

    Ok(Some(Value::Num(rcond_value)))
}

fn rcond_real_tensor(matrix: &Tensor) -> Result<f64, String> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape)?;
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
    }
    if rows == 0 {
        return Ok(f64::INFINITY);
    }
    if matrix.data.len() == 1 {
        return Ok(if matrix.data[0] == 0.0 { 0.0 } else { 1.0 });
    }
    let a = DMatrix::from_column_slice(rows, cols, &matrix.data);
    let svd = SVD::new(a, false, false);
    Ok(singular_value_rcond(svd.singular_values.as_slice()))
}

fn rcond_complex_tensor(matrix: &ComplexTensor) -> Result<f64, String> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape)?;
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
    }
    if rows == 0 {
        return Ok(f64::INFINITY);
    }
    if matrix.data.len() == 1 {
        let (re, im) = matrix.data[0];
        let magnitude = re.hypot(im);
        return Ok(if magnitude == 0.0 { 0.0 } else { 1.0 });
    }
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let a = DMatrix::from_column_slice(rows, cols, &data);
    let svd = SVD::new(a, false, false);
    Ok(singular_value_rcond(svd.singular_values.as_slice()))
}

fn upload_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: f64,
) -> Result<GpuTensorHandle, String> {
    let data = [value];
    let shape = [1usize, 1usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider.upload(&view).map_err(|e| format!("{NAME}: {e}"))
}

fn upload_identity(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    n: usize,
) -> Result<GpuTensorHandle, String> {
    if n == 0 {
        return upload_scalar(provider, 0.0);
    }
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        data[i + i * n] = 1.0;
    }
    let shape = vec![n, n];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider.upload(&view).map_err(|e| format!("{NAME}: {e}"))
}

/// Host-accessible helper for acceleration providers that gather matrices.
pub fn rcond_host_real_for_provider(matrix: &Tensor) -> Result<f64, String> {
    rcond_real_tensor(matrix)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_identity_is_one() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert!((value - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_zero_is_zero() {
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_nearly_singular() {
        let tensor =
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0e-8], vec![2, 2]).expect("tensor construction");
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert!((value - 1.0e-8).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_singular_matrix_zero() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_complex_matrix_supported() {
        let data = vec![(1.0, 2.0), (0.0, 0.0), (0.0, 3.0), (2.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = rcond_builtin(Value::ComplexTensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => {
                assert!(value > 0.0 && value <= 1.0, "unexpected rcond {value}")
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_rejects_non_square() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = rcond_builtin(Value::Tensor(tensor)).unwrap_err();
        assert_eq!(err, "rcond: input must be a square matrix.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_handles_empty_matrix() {
        let tensor = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert!(value.is_infinite() && value.is_sign_positive()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_accepts_scalar_int() {
        let int_value = Value::Int(IntValue::I32(5));
        let result = rcond_builtin(int_value).expect("rcond");
        match result {
            Value::Num(value) => assert!((value - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 0.0, 0.0, 0.5], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = rcond_builtin(Value::GpuTensor(handle)).expect("rcond");
            let gathered = test_support::gather(gpu_value).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert!((gathered.data[0] - 0.25).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rcond_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tol = match runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .precision()
        {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        let tensor = Tensor::new(vec![3.0, 0.0, 0.0, 0.25], vec![2, 2]).unwrap();
        let cpu_value = rcond_builtin(Value::Tensor(tensor.clone())).expect("cpu rcond");
        let cpu_scalar = match cpu_value {
            Value::Num(v) => v,
            other => panic!("expected scalar CPU value, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = rcond_builtin(Value::GpuTensor(handle)).expect("gpu rcond");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, vec![1, 1]);
        assert!((gathered.data[0] - cpu_scalar).abs() < tol);
    }
}
