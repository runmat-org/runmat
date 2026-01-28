//! MATLAB-compatible `rank` builtin that counts singular values above a tolerance.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::linalg::{
    matrix_dimensions_for, parse_tolerance_arg, svd_default_tolerance,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rank";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::rank")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("matrix-rank"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("rank")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may keep the computation on-device via the `rank` hook; the reference backend gathers to the host and re-uploads a scalar.",
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::rank")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`rank` terminates fusion plans and executes eagerly via an SVD.",
};

#[runtime_builtin(
    name = "rank",
    category = "math/linalg/solve",
    summary = "Compute the numerical rank of a matrix using SVD with MATLAB-compatible tolerance handling.",
    keywords = "rank,svd,tolerance,matrix,gpu",
    accel = "rank",
    builtin_path = "crate::builtins::math::linalg::solve::rank"
)]
async fn rank_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let tol = parse_tolerance_arg(NAME, &rest).map_err(builtin_error)?;
    match value {
        Value::GpuTensor(handle) => rank_gpu(handle, tol).await,
        Value::ComplexTensor(tensor) => rank_complex_tensor_value(tensor, tol),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            rank_complex_tensor_value(tensor, tol)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            rank_real_tensor_value(tensor, tol)
        }
    }
}

async fn rank_gpu(handle: GpuTensorHandle, tol: Option<f64>) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.rank(&handle, tol).await {
            Ok(device_scalar) => return Ok(Value::GpuTensor(device_scalar)),
            Err(_) => {
                // Fall through to host-based fallback.
            }
        }
    }

    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(map_control_flow)?;
    let rank = rank_scalar_from_value(gathered, tol)?;

    if let Some(provider) = runmat_accelerate_api::provider() {
        match upload_rank_scalar(provider, rank) {
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

    Ok(Value::Num(rank))
}

fn upload_rank_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    rank: f64,
) -> BuiltinResult<GpuTensorHandle> {
    let data = [rank];
    let shape = [1usize, 1usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn rank_real_tensor_value(tensor: Tensor, tol: Option<f64>) -> BuiltinResult<Value> {
    let rank = rank_real_tensor(&tensor, tol)?;
    Ok(Value::Num(rank as f64))
}

fn rank_complex_tensor_value(tensor: ComplexTensor, tol: Option<f64>) -> BuiltinResult<Value> {
    let rank = rank_complex_tensor(&tensor, tol)?;
    Ok(Value::Num(rank as f64))
}

fn rank_scalar_from_value(value: Value, tol: Option<f64>) -> BuiltinResult<f64> {
    match value {
        Value::Tensor(t) => rank_real_tensor(&t, tol).map(|r| r as f64),
        Value::ComplexTensor(t) => rank_complex_tensor(&t, tol).map(|r| r as f64),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            rank_complex_tensor(&tensor, tol).map(|r| r as f64)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            rank_real_tensor(&tensor, tol).map(|r| r as f64)
        }
    }
}

fn rank_real_tensor(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<usize> {
    rank_real_tensor_impl(matrix, tol)
}

fn rank_complex_tensor(matrix: &ComplexTensor, tol: Option<f64>) -> BuiltinResult<usize> {
    rank_complex_tensor_impl(matrix, tol)
}

fn rank_real_tensor_impl(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<usize> {
    let (rows, cols) =
        matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(builtin_error)?;
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    let dm = DMatrix::from_column_slice(rows, cols, &matrix.data);
    let svd = SVD::new(dm, false, false);
    let cutoff =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    Ok(svd
        .singular_values
        .iter()
        .filter(|&&value| value.is_infinite() || value > cutoff)
        .count())
}

fn rank_complex_tensor_impl(matrix: &ComplexTensor, tol: Option<f64>) -> BuiltinResult<usize> {
    let (rows, cols) =
        matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(builtin_error)?;
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let dm = DMatrix::from_column_slice(rows, cols, &data);
    let svd = SVD::new(dm, false, false);
    let cutoff =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    Ok(svd
        .singular_values
        .iter()
        .filter(|&&value| value.is_infinite() || value > cutoff)
        .count())
}

/// Host helper used by acceleration providers that defer rank to the shared implementation.
pub fn rank_host_real_for_provider(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<usize> {
    rank_real_tensor_impl(matrix, tol)
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_full_matrix() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = rank_real_tensor_value(tensor, None).expect("rank");
        match result {
            Value::Num(r) => assert_eq!(r, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_singular_matrix() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = rank_real_tensor_value(tensor, None).expect("rank");
        match result {
            Value::Num(r) => assert_eq!(r, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_default_tolerance_reduces_rank() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1e-16], vec![2, 2]).unwrap();
        let rank = rank_real_tensor(&tensor, None).expect("rank");
        assert_eq!(rank, 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_custom_tolerance_behavior() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1e-4], vec![2, 2]).unwrap();
        let default_rank = rank_real_tensor(&tensor, None).expect("rank");
        let custom_rank = rank_real_tensor(&tensor, Some(1e-3)).expect("rank");
        assert_eq!(default_rank, 2);
        assert_eq!(custom_rank, 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_empty_matrix_returns_zero() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let result = rank_real_tensor_value(tensor, None).expect("rank");
        match result {
            Value::Num(r) => assert_eq!(r, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_vector_input() {
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0], vec![3, 1]).unwrap();
        let rank = rank_real_tensor(&tensor, None).expect("rank");
        assert_eq!(rank, 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_zero_vector_is_zero() {
        let tensor = Tensor::new(vec![0.0, 0.0, 0.0], vec![3, 1]).unwrap();
        let rank = rank_real_tensor(&tensor, None).expect("rank");
        assert_eq!(rank, 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_invalid_shape_errors() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let err = unwrap_error(rank_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err());
        assert!(
            err.message().contains("2-D matrices or vectors"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_negative_tolerance_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err =
            unwrap_error(rank_builtin(Value::Tensor(tensor), vec![Value::Num(-1.0)]).unwrap_err());
        assert!(
            err.message().contains("tolerance must be >= 0"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_non_scalar_tolerance_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let tol = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = unwrap_error(
            rank_builtin(Value::Tensor(tensor), vec![Value::Tensor(tol)]).unwrap_err(),
        );
        assert!(
            err.message().contains("tolerance must be a real scalar"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_complex_matrix() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (2.0, -1.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = rank_complex_tensor_value(tensor, None).expect("rank");
        match result {
            Value::Num(r) => assert_eq!(r, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_scalar_bool_and_int() {
        let bool_rank = rank_builtin(Value::Bool(false), Vec::new()).expect("rank");
        let int_rank = rank_builtin(Value::Int(IntValue::I32(5)), Vec::new()).expect("rank");
        match bool_rank {
            Value::Num(r) => assert_eq!(r, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match int_rank {
            Value::Num(r) => assert_eq!(r, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_gpu_round_trip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = rank_builtin(Value::GpuTensor(handle), Vec::new()).expect("rank");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data[0], 1.0);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rank_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());

        let tensor = Tensor::new(
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let cpu_rank = rank_real_tensor(&tensor, None).expect("cpu rank") as f64;

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = rank_builtin(Value::GpuTensor(handle), Vec::new()).expect("rank");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gathered.data, vec![cpu_rank]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    fn rank_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::rank_builtin(value, rest))
    }
}
