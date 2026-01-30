//! MATLAB-compatible `pinv` builtin backed by SVD with GPU-aware fallbacks.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPinvOptions};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::linalg::{
    matrix_dimensions_for, parse_tolerance_arg, svd_default_tolerance,
};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::pinv_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "pinv";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::pinv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pinv",
    op_kind: GpuOpKind::Custom("pinv"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("pinv")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement a native GPU pseudoinverse; the reference WGPU backend gathers to host SVD and re-uploads the result.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::pinv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pinv",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Pseudoinverses are standalone solves and do not participate in fusion plans.",
};

#[runtime_builtin(
    name = "pinv",
    category = "math/linalg/solve",
    summary = "Compute the Moore–Penrose pseudoinverse of a matrix using SVD.",
    keywords = "pinv,pseudoinverse,svd,least squares,gpu",
    accel = "pinv",
    type_resolver(pinv_type),
    builtin_path = "crate::builtins::math::linalg::solve::pinv"
)]
async fn pinv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let tol = parse_tolerance_arg(NAME, &rest).map_err(builtin_error)?;
    match value {
        Value::GpuTensor(handle) => pinv_gpu(handle, tol).await,
        Value::ComplexTensor(t) => pinv_complex_value(t, tol),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            pinv_complex_value(tensor, tol)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            pinv_real_value(tensor, tol)
        }
    }
}

async fn pinv_gpu(handle: GpuTensorHandle, tol: Option<f64>) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let options = ProviderPinvOptions { tolerance: tol };
        match provider.pinv(&handle, options).await {
            Ok(result) => return Ok(Value::GpuTensor(result)),
            Err(_) => {
                // Fall through to host implementation and attempt to re-upload
            }
        }
        let gathered = gpu_helpers::gather_tensor_async(&handle)
            .await
            .map_err(map_control_flow)?;
        let pinv = pinv_real_tensor(&gathered, tol)?;
        if let Ok(uploaded) = provider.upload(&HostTensorView {
            data: &pinv.data,
            shape: &pinv.shape,
        }) {
            return Ok(Value::GpuTensor(uploaded));
        }
        return Ok(tensor::tensor_into_value(pinv));
    }
    let gathered = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(map_control_flow)?;
    let pinv = pinv_real_tensor(&gathered, tol)?;
    Ok(tensor::tensor_into_value(pinv))
}

fn pinv_real_value(tensor: Tensor, tol: Option<f64>) -> BuiltinResult<Value> {
    let pinv = pinv_real_tensor(&tensor, tol)?;
    Ok(tensor::tensor_into_value(pinv))
}

fn pinv_complex_value(tensor: ComplexTensor, tol: Option<f64>) -> BuiltinResult<Value> {
    let pinv = pinv_complex_tensor(&tensor, tol)?;
    Ok(complex_tensor_into_value(pinv))
}

fn pinv_real_tensor(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<Tensor> {
    pinv_real_tensor_impl(matrix, tol)
}

fn pinv_complex_tensor(matrix: &ComplexTensor, tol: Option<f64>) -> BuiltinResult<ComplexTensor> {
    pinv_complex_tensor_impl(matrix, tol)
}

fn pinv_real_tensor_impl(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<Tensor> {
    let (rows, cols) =
        matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(builtin_error)?;
    if rows == 0 || cols == 0 {
        return Tensor::new(vec![0.0; cols * rows], vec![cols, rows])
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let dm = DMatrix::from_column_slice(rows, cols, &matrix.data);
    let pinv = pseudoinverse_real(&dm, tol)?;
    matrix_to_tensor(NAME, pinv)
}

fn pinv_complex_tensor_impl(
    matrix: &ComplexTensor,
    tol: Option<f64>,
) -> BuiltinResult<ComplexTensor> {
    let (rows, cols) =
        matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(builtin_error)?;
    if rows == 0 || cols == 0 {
        return ComplexTensor::new(vec![(0.0, 0.0); cols * rows], vec![cols, rows])
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let dm = DMatrix::from_column_slice(rows, cols, &data);
    let pinv = pseudoinverse_complex(&dm, tol)?;
    matrix_to_complex_tensor(NAME, pinv)
}

fn pseudoinverse_real(matrix: &DMatrix<f64>, tol: Option<f64>) -> BuiltinResult<DMatrix<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let svd = SVD::new(matrix.clone(), true, true);
    let cutoff =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    svd.pseudo_inverse(cutoff)
        .map_err(|msg| builtin_error(format!("{NAME}: failed to compute pseudoinverse ({msg})")))
}

fn pseudoinverse_complex(
    matrix: &DMatrix<Complex64>,
    tol: Option<f64>,
) -> BuiltinResult<DMatrix<Complex64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let svd = SVD::new(matrix.clone(), true, true);
    let cutoff =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    let u = svd.u.ok_or_else(|| {
        builtin_error(format!(
            "{NAME}: failed to compute pseudoinverse (missing U)"
        ))
    })?;
    let v_t = svd.v_t.ok_or_else(|| {
        builtin_error(format!(
            "{NAME}: failed to compute pseudoinverse (missing Vᴴ)"
        ))
    })?;
    let mut sigma_plus = DMatrix::<Complex64>::zeros(cols, rows);
    for (idx, value) in svd.singular_values.iter().enumerate() {
        if value.is_infinite() || *value > cutoff {
            sigma_plus[(idx, idx)] = Complex64::new(1.0 / *value, 0.0);
        }
    }
    let v = v_t.adjoint();
    let u_h = u.adjoint();
    Ok(v * sigma_plus * u_h)
}

fn matrix_to_tensor(label: &str, matrix: DMatrix<f64>) -> BuiltinResult<Tensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    Tensor::new(matrix.as_slice().to_vec(), vec![rows, cols])
        .map_err(|e| builtin_error(format!("{label}: {e}")))
}

fn matrix_to_complex_tensor(
    label: &str,
    matrix: DMatrix<Complex64>,
) -> BuiltinResult<ComplexTensor> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<(f64, f64)> = matrix.as_slice().iter().map(|c| (c.re, c.im)).collect();
    ComplexTensor::new(data, vec![rows, cols]).map_err(|e| builtin_error(format!("{label}: {e}")))
}

/// Host helper used by acceleration providers that delegate `pinv` back to the CPU path.
pub fn pinv_host_real_for_provider(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<Tensor> {
    pinv_real_tensor_impl(matrix, tol)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntValue, Type};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    fn approx_equal(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (lhs, rhs) in a.iter().zip(b.iter()) {
            assert!(
                (lhs - rhs).abs() <= tol,
                "expected {lhs} ≈ {rhs} within {tol}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_scalar_real() {
        let result = pinv_builtin(Value::Num(4.0), Vec::new()).expect("pinv");
        match result {
            Value::Num(v) => assert!((v - 0.25).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn pinv_type_swaps_dims() {
        let out = pinv_type(&[Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        }]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(2)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_rank_deficient_square() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = pinv_builtin(Value::Tensor(tensor), Vec::new()).expect("pinv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                approx_equal(&out.data, &[0.04, 0.08, 0.08, 0.16], 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_rectangular() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], vec![3, 2]).unwrap();
        let result = pinv_builtin(Value::Tensor(tensor), Vec::new()).expect("pinv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                approx_equal(&out.data, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_custom_tolerance_zeroes_small_singular_values() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1e-12], vec![2, 2]).unwrap();
        let result = pinv_builtin(Value::Tensor(tensor), vec![Value::Num(1e-6)]).expect("pinv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                approx_equal(&out.data, &[1.0, 0.0, 0.0, 0.0], 1e-9);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_complex_diagonal() {
        let tensor = ComplexTensor::new(
            vec![(2.0, 1.0), (0.0, 0.0), (0.0, 0.0), (3.0, -2.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = pinv_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("pinv");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let expected = [
                    (0.4, -0.2),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.23076923076923078, 0.15384615384615385),
                ];
                for (actual, expected) in out.data.iter().zip(expected.iter()) {
                    assert!((actual.0 - expected.0).abs() < 1e-12);
                    assert!((actual.1 - expected.1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = pinv_builtin(Value::GpuTensor(handle), Vec::new()).expect("gpu pinv");
            let gathered = test_support::gather(result).expect("gather");
            let cpu = pinv_real_tensor(&tensor, None).expect("cpu pinv");
            approx_equal(&gathered.data, &cpu.data, 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn pinv_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let cpu = pinv_real_tensor(&tensor, None).expect("cpu pinv");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = pinv_builtin(Value::GpuTensor(handle), Vec::new()).expect("gpu pinv");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape, "shape mismatch");

        match runmat_accelerate_api::provider().unwrap().precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => {
                approx_equal(&gathered.data, &cpu.data, 1e-10);
            }
            runmat_accelerate_api::ProviderPrecision::F32 => {
                approx_equal(&gathered.data, &cpu.data, 5e-5);
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_rejects_negative_tolerance() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = unwrap_error(
            pinv_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(-1))]).unwrap_err(),
        );
        assert!(err.message().contains("tolerance"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_tolerance_accepts_boolean() {
        let result =
            pinv_builtin(Value::Num(4.0), vec![Value::Bool(true)]).expect("pinv with bool tol");
        match result {
            Value::Num(v) => assert!((v - 0.25).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_tolerance_rejects_non_scalar_tensor() {
        let tol_tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = unwrap_error(
            pinv_builtin(Value::Tensor(tensor), vec![Value::Tensor(tol_tensor)]).unwrap_err(),
        );
        assert!(err.message().contains("tolerance must be a real scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pinv_tolerance_rejects_char_array() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let chars = CharArray::new("hi".chars().collect(), 1, 2).unwrap();
        let err = unwrap_error(
            pinv_builtin(Value::Tensor(tensor), vec![Value::CharArray(chars)]).unwrap_err(),
        );
        assert!(err.message().contains("tolerance must be a real scalar"));
    }

    fn pinv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::pinv_builtin(value, rest))
    }
}
