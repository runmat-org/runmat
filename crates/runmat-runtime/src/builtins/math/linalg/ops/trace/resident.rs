//! Resident diagonal reduction and class-preserving host fallback for `trace`.
use super::*;
use gpu_helpers::{GpuOutputAliasPolicy, UnaryGpuOutputContract};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};

pub(super) async fn trace_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    ensure_matrix_shape(NAME, handle.shape())?;
    let (rows, _) = matrix_extents_from_shape(handle.shape());
    let diag_len = rows;

    let floating_input = runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle);
    if diag_len != 0 && floating_input {
        if let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) {
            let diagonal = match provider.diag_extract(&handle, 0) {
                Ok(diagonal) => Some(diagonal),
                Err(source) if gpu_helpers::provider_hook_is_unsupported(&source) => None,
                Err(source) => return Err(trace_provider_failure(source)),
            };
            if let Some(diagonal) = diagonal {
                let diagonal = admit_resident(
                    provider,
                    &handle,
                    diagonal,
                    &[diag_len, 1],
                    runmat_accelerate_api::handle_precision(&handle),
                    &[&handle],
                )?;
                let reduced = match provider.reduce_sum(&diagonal).await {
                    Ok(result) => Some(result),
                    Err(source) if source.is_unsupported() => None,
                    Err(source) => return Err(trace_provider_failure(source)),
                };
                if let Some(result) = reduced {
                    let result = admit_resident(
                        provider,
                        &handle,
                        result,
                        &[1, 1],
                        Some(ProviderPrecision::F64),
                        &[&handle, &diagonal],
                    )?;
                    let result = if runmat_accelerate_api::handle_precision(&handle)
                        == Some(ProviderPrecision::F32)
                    {
                        let cast = match provider.unary_single(&result).await {
                            Ok(cast) => Some(cast),
                            Err(source) if gpu_helpers::provider_hook_is_unsupported(&source) => {
                                None
                            }
                            Err(source) => return Err(trace_provider_failure(source)),
                        };
                        if let Some(cast) = cast {
                            Some(admit_resident(
                                provider,
                                &handle,
                                cast,
                                &[1, 1],
                                Some(ProviderPrecision::F32),
                                &[&handle, &diagonal, &result],
                            )?)
                        } else {
                            None
                        }
                    } else {
                        Some(result)
                    };
                    if let Some(mut result) = result {
                        gpu_helpers::propagate_output_provenance(&mut result, [&handle])?;
                        return Ok(Value::GpuTensor(result));
                    }
                }
            }
        }
    }

    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(map_control_flow)?;
    let traced = match gathered {
        Value::ComplexTensor(tensor) => trace_complex_tensor(tensor)?,
        other => trace_numeric(other)?,
    };
    let host = match traced {
        Value::Num(value) => {
            Value::Tensor(Tensor::new(vec![value], vec![1, 1]).map_err(trace_invalid_input)?)
        }
        value => value,
    };
    gpu_helpers::restore_class_preserving_value(&handle, host, NAME)
        .map_err(|error| trace_error_with_message(format!("trace: {error}"), &TRACE_ERROR_INTERNAL))
}

fn trace_provider_failure(
    source: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> RuntimeError {
    let mut error = trace_error(&TRACE_ERROR_INTERNAL);
    error = crate::runtime_error::with_provider_execution_source(error, source.into());
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

fn admit_resident(
    provider: &'static dyn AccelProvider,
    original: &GpuTensorHandle,
    output: GpuTensorHandle,
    shape: &[usize],
    precision: Option<ProviderPrecision>,
    borrowed: &[&GpuTensorHandle],
) -> BuiltinResult<GpuTensorHandle> {
    let valid = precision.is_some_and(|precision| {
        gpu_helpers::unary_gpu_output_matches_shape(
            &output,
            original,
            provider,
            shape,
            UnaryGpuOutputContract {
                storage: GpuTensorStorage::Real,
                precision: Some(precision),
                integer: None,
                logical: false,
                alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
            },
        )
    }) && output.allocation_lease().validate_live().is_ok()
        && gpu_helpers::expected_handle_numeric_element_type(&output).is_ok()
        && borrowed
            .iter()
            .all(|input| !output.shares_allocation_with(input));
    if valid {
        Ok(output)
    } else {
        let mut error = trace_error(&TRACE_ERROR_INTERNAL);
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(gpu_helpers::reject_gpu_output(output, error))
    }
}
