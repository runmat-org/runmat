//! MATLAB-compatible `rcond` builtin with GPU-aware fallbacks.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_accelerate_api::HostTensorView;
#[cfg(test)]
use runmat_builtins::RCOND_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, RCOND_ERROR_INVALID_INPUT, RCOND_INTEGER_INPUT_EXTENSION,
    RCOND_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::linalg::matrix_dimensions_for;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rcond";

use crate::builtins::common::linalg::one_norm;

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
    notes: "The shipping providers gather to the shared LU/1-norm estimator and restore a class-preserving scalar; unsupported hooks fall back through the exact input owner.",
};

fn rcond_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    rcond_error_with_message(message, &RCOND_ERROR_INVALID_INPUT)
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
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::solve::rcond"
)]
async fn rcond_builtin(value: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RCOND_INTEGER_INPUT_EXTENSION,
            NAME,
        )?;
    }
    if crate::builtins::common::validation::value_has_logical_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RCOND_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&value, NAME)?;
    if crate::builtins::common::validation::value_has_native_integer_class(&value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
    {
        return Err(builtin_error(
            "rcond: integer input must be exactly representable as double",
        ));
    }
    let single_output = matches!(&value, Value::Tensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32)
        || matches!(&value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == runmat_value::NumericDType::F32);
    let estimate = match value {
        Value::GpuTensor(handle) => return rcond_gpu(handle).await,
        Value::ComplexTensor(matrix) => rcond_complex_tensor(&matrix)?,
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            rcond_complex_tensor(&tensor)?
        }
        other => {
            let matrix = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            rcond_real_tensor(&matrix)?
        }
    };
    host_scalar(estimate, single_output)
}

fn host_scalar(estimate: f64, single: bool) -> BuiltinResult<Value> {
    if single {
        Tensor::from_f32(vec![estimate as f32], vec![1, 1])
            .map(Value::Tensor)
            .map_err(builtin_error)
    } else {
        Ok(Value::Num(estimate))
    }
}

async fn rcond_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let (rows, cols) = matrix_dimensions_for(NAME, handle.shape()).map_err(builtin_error)?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }

    let owner = Some(gpu_helpers::require_resident_compute_owner(
        &handle,
        || builtin_error("rcond: resident input has no exact owning provider"),
    )?);
    let nonfloating = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    let source_single = !nonfloating
        && runmat_accelerate_api::handle_precision(&handle)
            == Some(runmat_accelerate_api::ProviderPrecision::F32);

    if rows != 0
        && !nonfloating
        && runmat_accelerate_api::handle_storage(&handle)
            == runmat_accelerate_api::GpuTensorStorage::Real
    {
        if let Some(provider) = owner {
            let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
            match provider.rcond(&handle).await {
                Ok(result) if valid_provider_rcond_result(provider, &handle, &result) => {
                    let result = gpu_helpers::propagate_owned_output_provenance(result, [&handle])?;
                    return gpu_helpers::resident_gpu_value(result);
                }
                Ok(result) => {
                    let mut error = builtin_error(
                        "rcond: provider returned a malformed, aliased, or foreign result",
                    );
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    return Err(gpu_helpers::reject_gpu_output(result, error));
                }
                Err(error) if error.is_unsupported() => {}
                Err(error) => {
                    let mut failure =
                        builtin_error(format!("rcond: provider execution failed: {error}"));
                    failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    return Err(crate::runtime_error::with_provider_execution_source(
                        failure, error,
                    ));
                }
            }
        }
    }

    let estimate = if rows == 0 {
        f64::INFINITY
    } else {
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await
            .map_err(map_control_flow)?;
        match gathered {
            Value::Tensor(tensor) => rcond_real_tensor(&tensor)?,
            Value::ComplexTensor(tensor) => rcond_complex_tensor(&tensor)?,
            Value::Num(n) => {
                if !n.is_finite() {
                    f64::NAN
                } else if n == 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
            Value::Complex(re, im) => {
                if !re.is_finite() || !im.is_finite() {
                    f64::NAN
                } else if re == 0.0 && im == 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
            other => {
                let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
                rcond_real_tensor(&tensor)?
            }
        }
    };

    let host = host_scalar(estimate, source_single)?;
    let restored = if owner.is_some() {
        gpu_helpers::restore_class_preserving_value(&handle, host, NAME)?
    } else {
        host
    };
    if runmat_accelerate_api::handle_is_explicit(&handle)
        && !matches!(restored, Value::GpuTensor(_))
    {
        return Err(builtin_error(
            "rcond: provider cannot preserve explicit gpuArray output residency",
        ));
    }
    Ok(restored)
}

fn valid_provider_rcond_result(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    source: &GpuTensorHandle,
    result: &GpuTensorHandle,
) -> bool {
    gpu_helpers::unary_gpu_output_matches_shape(
        result,
        source,
        provider,
        &[1, 1],
        gpu_helpers::UnaryGpuOutputContract {
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(source),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}

fn rcond_real_tensor(matrix: &Tensor) -> BuiltinResult<f64> {
    rcond_real_tensor_impl(matrix)
}

fn rcond_complex_tensor(matrix: &ComplexTensor) -> BuiltinResult<f64> {
    rcond_complex_tensor_impl(matrix)
}

fn rcond_real_tensor_impl(matrix: &Tensor) -> BuiltinResult<f64> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape()).map_err(builtin_error)?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 {
        return Ok(f64::INFINITY);
    }
    let values = tensor::tensor_values_f64_cow(matrix);
    if values.len() == 1 {
        return Ok(if !values[0].is_finite() {
            f64::NAN
        } else if values[0] == 0.0 {
            0.0
        } else {
            1.0
        });
    }
    let a = DMatrix::from_column_slice(rows, cols, &values);
    Ok(one_norm::reciprocal_one_norm(a))
}

fn rcond_complex_tensor_impl(matrix: &ComplexTensor) -> BuiltinResult<f64> {
    let (rows, cols) = matrix_dimensions_for(NAME, matrix.shape()).map_err(builtin_error)?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 {
        return Ok(f64::INFINITY);
    }
    if matrix.len() == 1 {
        let (re, im) = matrix.materialize_f64()[0];
        // A finite nonzero scalar has condition one even when computing its
        // modulus would overflow or underflow the floating-point range.
        return Ok(if !re.is_finite() || !im.is_finite() {
            f64::NAN
        } else if re == 0.0 && im == 0.0 {
            0.0
        } else {
            1.0
        });
    }
    let a = if let Some(values) = matrix.as_f64_slice() {
        DMatrix::from_iterator(
            rows,
            cols,
            values.iter().map(|value| Complex64::new(value.0, value.1)),
        )
    } else if let Some(values) = matrix.as_f32_slice() {
        DMatrix::from_iterator(
            rows,
            cols,
            values
                .iter()
                .map(|value| Complex64::new(f64::from(value.0), f64::from(value.1))),
        )
    } else {
        // Public admission rejects typed complex integer input. Keep the
        // internal storage path single-materialization as well, moving each
        // value directly into the LU matrix without another full-size vector.
        DMatrix::from_iterator(
            rows,
            cols,
            matrix
                .materialize_f64()
                .into_iter()
                .map(|(re, im)| Complex64::new(re, im)),
        )
    };
    Ok(one_norm::reciprocal_one_norm(a))
}

/// Host-accessible helper for acceleration providers that gather matrices.
pub fn rcond_host_real_for_provider(matrix: &Tensor) -> BuiltinResult<f64> {
    rcond_real_tensor_impl(matrix)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

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

    #[test]
    fn rcond_reads_typed_integer_tensor_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![2, 0, 0, 4]), vec![2, 2])
            .expect("integer");
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert!((value - 0.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn rcond_resident_integer_gates_then_uses_checked_owner_fallback() {
        test_support::with_test_provider(|provider| {
            let input =
                Tensor::new_integer(IntegerStorage::U16(vec![2, 0, 0, 4]), vec![2, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let err = rcond_builtin(Value::GpuTensor(handle.clone()))
                    .expect_err("strict mode must gate resident integer input");
                assert_eq!(
                    err.identifier(),
                    RCOND_INTEGER_INPUT_EXTENSION.error_identifier
                );
            }
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let Value::GpuTensor(output) =
                rcond_builtin(Value::GpuTensor(handle)).expect("resident rcond")
            else {
                panic!("resident fallback must restore output");
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(&output), None);
            let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather output");
            assert_eq!(gathered.materialize_f64(), vec![0.5]);
        });
    }

    #[test]
    fn rcond_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = RCOND_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"c = rcond(A)"));
    }

    #[test]
    fn rcond_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = RCOND_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.RCOND.INVALID_INPUT"));
        assert!(codes.contains(&"RM.RCOND.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_scalar_zero_and_finite_extremes() {
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }

        for (re, im, expected) in [
            (0.0, 0.0, 0.0),
            (1e308, 1e308, 1.0),
            (f64::MAX, f64::MAX, 1.0),
            (f64::from_bits(1), f64::from_bits(1), 1.0),
            (f64::INFINITY, 1.0, f64::NAN),
        ] {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).unwrap();
            let result = rcond_builtin(Value::ComplexTensor(tensor)).expect("complex rcond");
            match result {
                Value::Num(value) if expected.is_nan() => assert!(value.is_nan()),
                Value::Num(value) => assert_eq!(value, expected, "scalar ({re:e}, {im:e})"),
                other => panic!("expected complex-scalar rcond, got {other:?}"),
            }
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
                let expected = 1.0 / ((3.0 + 5.0_f64.sqrt()) * (3.0 / 5.0 + 1.0 / 5.0_f64.sqrt()));
                assert!((value - expected).abs() < 1e-8, "unexpected rcond {value}")
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_rejects_non_square() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = unwrap_error(rcond_builtin(Value::Tensor(tensor)).unwrap_err());
        assert_eq!(err.message(), "rcond: input must be a square matrix.");
        assert_eq!(err.identifier(), RCOND_ERROR_INVALID_INPUT.identifier);
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
    fn rcond_single_host_retains_class_and_uses_one_norm() {
        let matrix = Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = rcond_builtin(Value::Tensor(matrix)).expect("rcond");
        let Value::Tensor(result) = result else {
            panic!("Single input must return a Single tensor");
        };
        assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(result.shape(), [1, 1]);
        assert!((result.materialize_f64()[0] - 1.0 / 21.0).abs() < 1e-6);

        let complex = ComplexTensor::from_f32(
            vec![(1.0, 2.0), (0.0, 3.0), (0.0, 0.0), (2.0, -1.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = rcond_builtin(Value::ComplexTensor(complex)).expect("complex rcond");
        let Value::Tensor(result) = result else {
            panic!("complex Single input must return a real Single tensor");
        };
        assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::F32);
        assert!((result.materialize_f64()[0] - 0.1823725421878943).abs() < 1e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_accepts_scalar_int() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
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
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = rcond_builtin(Value::GpuTensor(handle)).expect("rcond");
            let gathered = test_support::gather(gpu_value).expect("gather");
            assert_eq!(gathered.shape(), [1, 1]);
            assert!((gathered.materialize_f64()[0] - 0.25).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rcond_wgpu_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
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
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let input_precision = runmat_accelerate_api::handle_precision(&handle);
        let gpu_value = rcond_builtin(Value::GpuTensor(handle)).expect("gpu rcond");
        let Value::GpuTensor(ref output) = gpu_value else {
            panic!("provider rcond must retain resident output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(output),
            input_precision
        );
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape(), [1, 1]);
        assert!((gathered.materialize_f64()[0] - cpu_scalar).abs() < tol);
    }

    fn rcond_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::rcond_builtin(value))
    }
}
