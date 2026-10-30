//! MATLAB-compatible `inv` builtin with GPU-aware fallbacks.

#[cfg(test)]
use nalgebra::DMatrix;
#[cfg(test)]
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, ProviderInvOptions};
#[cfg(test)]
use runmat_builtins::INV_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, INV_ERROR_INTERNAL, INV_ERROR_INVALID_INPUT,
    INV_INTEGER_INPUT_EXTENSION, INV_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::NumericDType;
use runmat_value::{ComplexTensor, Tensor, Value};

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

fn inv_error_with_message(
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
    inv_error_with_message(message, &INV_ERROR_INVALID_INPUT)
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
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::solve::inv"
)]
async fn inv_builtin(value: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INV_INTEGER_INPUT_EXTENSION, NAME)?;
        if !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
        {
            return Err(builtin_error(
                "inv: integer input must be exactly representable as double",
            ));
        }
    }
    if crate::builtins::common::validation::value_has_logical_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INV_LOGICAL_INPUT_EXTENSION, NAME)?;
    }
    crate::builtins::common::validation::reject_typed_complex_integer(&value, NAME)?;
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
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        inv_error_with_message(
            "inv: no acceleration provider owns the input handle",
            &INV_ERROR_INTERNAL,
        )
    })?;
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let expected_storage = runmat_accelerate_api::handle_storage(&handle);
    let expected_precision = runmat_accelerate_api::handle_precision(&handle);
    if runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle)
    {
        if !matches!(
            expected_storage,
            runmat_accelerate_api::GpuTensorStorage::Real
                | runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        ) || expected_precision.is_none()
            || !gpu_helpers::gpu_class_metadata_matches(&handle, expected_precision, None, false)
        {
            return Err(inv_error_with_message(
                "inv: input handle has contradictory floating metadata",
                &INV_ERROR_INTERNAL,
            ));
        }
        let options = ProviderInvOptions::default();
        match provider.inv(&handle, options).await {
            Ok(result)
                if valid_inv_gpu_output(
                    &handle,
                    &result,
                    provider,
                    expected_storage,
                    expected_precision,
                ) =>
            {
                let result = gpu_helpers::propagate_owned_output_provenance(result, [&handle])?;
                return gpu_helpers::resident_gpu_value(result);
            }
            Ok(result) => {
                let mut error = inv_error_with_message(
                    "inv: provider returned an invalid inverse result",
                    &INV_ERROR_INTERNAL,
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(gpu_helpers::reject_gpu_output(result, error));
            }
            Err(error) if error.is_unsupported() => {}
            Err(error) => {
                let mut failure =
                    build_runtime_error(format!("inv: provider execution failed: {error}"))
                        .with_builtin(NAME)
                        .with_identifier(
                            INV_ERROR_INTERNAL
                                .identifier
                                .expect("inv internal descriptor identifier"),
                        )
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build();
                failure = crate::runtime_error::with_provider_execution_source(failure, error);
                return Err(failure);
            }
        }
    }
    let gathered = gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    let gathered = gathered.map_err(map_control_flow)?;
    let result = match gathered {
        Value::ComplexTensor(tensor) => Value::ComplexTensor(inv_complex_tensor(&tensor)?),
        Value::Complex(real, imaginary) => {
            let tensor =
                ComplexTensor::new(vec![(real, imaginary)], vec![1, 1]).map_err(builtin_error)?;
            Value::ComplexTensor(inv_complex_tensor(&tensor)?)
        }
        real => {
            let tensor = tensor::value_into_tensor_for(NAME, real).map_err(builtin_error)?;
            Value::Tensor(inv_real_tensor(&tensor)?)
        }
    };
    gpu_helpers::restore_class_preserving_value(&handle, result, NAME)
}

fn valid_inv_gpu_output(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    expected_storage: runmat_accelerate_api::GpuTensorStorage,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: expected_storage,
            precision: expected_precision,
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}

fn inv_real_value(tensor: Tensor) -> BuiltinResult<Value> {
    let inv = inv_real_tensor(&tensor)?;
    Ok(tensor::tensor_into_value(inv))
}

fn inv_complex_value(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let inv = inv_complex_tensor(&tensor)?;
    if inv.numeric_dtype() == NumericDType::F64 && inv.materialize_f64().len() == 1 {
        let (re, im) = inv.materialize_f64()[0];
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
    crate::builtins::common::linalg::inverse::real(matrix).map_err(host_inverse_error)
}
fn inv_complex_tensor_impl(matrix: &ComplexTensor) -> BuiltinResult<ComplexTensor> {
    crate::builtins::common::linalg::inverse::complex(matrix).map_err(host_inverse_error)
}
fn host_inverse_error(
    source: crate::builtins::common::linalg::inverse::InverseError,
) -> RuntimeError {
    build_runtime_error(format!("{NAME}: {source}"))
        .with_builtin(NAME)
        .with_identifier(
            INV_ERROR_INVALID_INPUT
                .identifier
                .expect("inverse input error identifier"),
        )
        .with_source(source)
        .build()
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
    use runmat_value::{IntValue, IntegerStorage};
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

    #[test]
    fn inv_descriptor_signatures_cover_core_form() {
        let labels: Vec<&str> = INV_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"X = inv(A)"));
    }

    #[test]
    fn inv_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = INV_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.INV.INVALID_INPUT"));
        assert!(codes.contains(&"RM.INV.INTERNAL"));
    }

    #[test]
    fn inv_provider_unsupported_classification_is_exact() {
        let unsupported = runmat_accelerate_api::ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation("inv unavailable"),
        );
        assert!(unsupported.is_unsupported());
        assert!(!unsupported.with_unproven_effects().is_unsupported());
        assert!(
            !runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                "inv not supported by provider"
            ))
            .is_unsupported()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_square_matrix() {
        let data = vec![4.0, 1.0, -2.0, 3.0];
        let tensor = Tensor::new(data.clone(), vec![2, 2]).unwrap();
        let result = inv_builtin(Value::Tensor(tensor)).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 2]);
                let a = DMatrix::from_column_slice(2, 2, &data);
                let inv_m = DMatrix::from_column_slice(2, 2, &out.materialize_f64());
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

    #[test]
    fn inv_preserves_native_single_for_real_and_complex_inputs() {
        let real = Tensor::from_f32(vec![4.0, 0.0, 0.0, 2.0], vec![2, 2]).expect("single");
        let Value::Tensor(real_output) = inv_builtin(Value::Tensor(real)).expect("single inv")
        else {
            panic!("expected single tensor")
        };
        assert_eq!(real_output.numeric_dtype(), NumericDType::F32);
        assert_eq!(real_output.materialize_f64(), vec![0.25, 0.0, 0.0, 0.5]);

        let complex =
            ComplexTensor::from_f32(vec![(2.0, 0.0)], vec![1, 1]).expect("complex single");
        let Value::ComplexTensor(complex_output) =
            inv_builtin(Value::ComplexTensor(complex)).expect("complex single inv")
        else {
            panic!("expected complex single tensor")
        };
        assert_eq!(complex_output.numeric_dtype(), NumericDType::F32);
        assert_eq!(complex_output.materialize_f64(), vec![(0.5, 0.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            || Tensor::new_integer(IntegerStorage::U64(vec![4, 1, 2, 3]), vec![2, 2]).unwrap();
        let error = inv_builtin(Value::Tensor(tensor()))
            .expect_err("compatible mode rejects integer matrices");
        assert_eq!(
            error.identifier(),
            INV_INTEGER_INPUT_EXTENSION.error_identifier
        );

        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = inv_builtin(Value::Tensor(tensor())).expect("RunMat integer inv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), [2, 2]);
                let expected = [0.3, -0.1, -0.2, 0.4];
                for (actual, expected) in out.materialize_f64().iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12);
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
                assert!(out.materialize_f64().is_empty());
                assert_eq!(out.shape(), [0, 0]);
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
                assert_eq!(out.shape(), [2, 2, 1]);
                let expected = vec![0.25, 0.0, 0.0, 0.5];
                assert_eq!(out.materialize_f64(), expected);
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
                assert_eq!(out.shape(), [2, 2]);
                let input: Vec<Complex64> =
                    raw.iter().map(|&(re, im)| Complex64::new(re, im)).collect();
                let inv_vec: Vec<Complex64> = out
                    .materialize_f64()
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
        assert_eq!(err.identifier(), INV_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_singular_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let err = unwrap_error(inv_builtin(Value::Tensor(tensor)).unwrap_err());
        assert!(err.message().contains("singular"), "{err}");
        assert_eq!(err.identifier(), INV_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = inv_builtin(Value::GpuTensor(handle)).expect("gpu inv");
            let gathered = test_support::gather(gpu_value).expect("gather");
            let cpu = inv_real_tensor(&tensor).expect("cpu");
            assert_eq!(gathered.shape(), cpu.shape());
            for (a, b) in gathered
                .materialize_f64()
                .iter()
                .zip(cpu.materialize_f64().iter())
            {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn inv_gpu_output_validation_uses_immutable_expected_metadata() {
        test_support::with_test_provider(|provider| {
            let input = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[2.0],
                    shape: &[1, 1],
                })
                .expect("upload input");
            let output = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[0.5],
                    shape: &[1, 1],
                })
                .expect("upload output");
            let expected_storage = runmat_accelerate_api::handle_storage(&input);
            let expected_precision = runmat_accelerate_api::handle_precision(&input);
            assert!(valid_inv_gpu_output(
                &input,
                &output,
                provider,
                expected_storage,
                expected_precision,
            ));
            for data in [
                runmat_accelerate_api::HostNumericDataView::F32(&[0.5]),
                runmat_accelerate_api::HostNumericDataView::U64(&[1]),
            ] {
                let wrong_class = runmat_accelerate_api::upload_numeric_protecting(
                    provider,
                    &runmat_accelerate_api::HostNumericTensorView {
                        data,
                        shape: &[1, 1],
                        storage: runmat_accelerate_api::GpuTensorStorage::Real,
                    },
                    &[&input, &output],
                )
                .unwrap();
                assert!(!valid_inv_gpu_output(
                    &input,
                    &wrong_class,
                    provider,
                    expected_storage,
                    expected_precision,
                ));
            }
            input.allocation_lease().validate_live().unwrap();
            output.allocation_lease().validate_live().unwrap();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn inv_scalar_int_promotes() {
        let error = inv_builtin(Value::Int(IntValue::I32(2)))
            .expect_err("compatible mode rejects integer scalar");
        assert_eq!(
            error.identifier(),
            INV_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = inv_builtin(Value::Int(IntValue::I32(2))).expect("RunMat integer inv");
        match result {
            Value::Num(v) => assert!((v - 0.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn inv_wgpu_matches_cpu() {
        if provider::register_wgpu_provider(WgpuProviderOptions::default()).is_err() {
            return;
        }

        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let cpu = inv_real_tensor(&tensor).expect("cpu");

        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = inv_builtin(Value::GpuTensor(handle)).expect("gpu inv");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gathered.shape(), cpu.shape());
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered
            .materialize_f64()
            .iter()
            .zip(cpu.materialize_f64().iter())
        {
            assert!((*a - *b).abs() < tol, "expected {b}, got {a}");
        }

        let complex = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).unwrap();
        let source = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
        let result = inv_builtin(Value::GpuTensor(source)).expect("complex gpu inv");
        let Value::GpuTensor(output) = result else {
            panic!("complex inverse did not restore resident output");
        };
        let gathered = block_on(gpu_helpers::download_value_preserving_residency_async(
            provider, &output,
        ))
        .unwrap();
        let Value::ComplexTensor(gathered) = gathered else {
            panic!("complex inverse lost complex storage");
        };
        let (re, im) = gathered.materialize_f64()[0];
        assert!((re - 0.4).abs() < tol);
        assert!((im + 0.2).abs() < tol);
    }

    fn inv_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::inv_builtin(value))
    }
}
