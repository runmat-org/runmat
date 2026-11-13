//! Resident diagonal reduction and class-preserving host fallback for `trace`.
use super::*;
use gpu_helpers::{GpuOutputAliasPolicy, UnaryGpuOutputContract};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderDoubleSumPreflight, ProviderPrecision,
};

pub(super) async fn trace_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    ensure_matrix_shape(NAME, handle.shape())?;
    gpu_helpers::require_resident_compute_owner(&handle, || {
        trace_error_with_message(
            "trace: resident input has no exact owning provider",
            &TRACE_ERROR_INTERNAL,
        )
    })?;
    let (rows, _) = matrix_extents_from_shape(handle.shape());
    let diag_len = rows;
    let storage = runmat_accelerate_api::handle_storage(&handle);

    let floating_input = runmat_accelerate_api::handle_integer_type(&handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(&handle);
    if diag_len != 0 && floating_input {
        // Resident trace uses the same double-accumulating sum contract as the
        // host path. Only the owning provider may certify this hook unavailable
        // before diagonal extraction creates an effect that forbids fallback.
        if let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle).filter(|provider| {
            provider.preflight_double_sum(&handle) != ProviderDoubleSumPreflight::Unsupported
        }) {
            let diagonal = match provider.diag_extract(&handle, 0) {
                Ok(diagonal) => Some(diagonal),
                Err(source) if gpu_helpers::provider_hook_allows_host_fallback(&source) => None,
                Err(source) => return Err(trace_provider_failure(source)),
            };
            if let Some(diagonal) = diagonal {
                let diagonal = admit_resident(
                    provider,
                    &handle,
                    diagonal,
                    &[diag_len, 1],
                    floating_contract(
                        storage,
                        runmat_accelerate_api::handle_precision(&handle),
                        GpuOutputAliasPolicy::AllowInput,
                    ),
                    &[],
                )?;
                let reduced = match provider.reduce_sum(&diagonal).await {
                    Ok(result) => Some(result),
                    Err(source) if source.is_unsupported() => {
                        return Err(trace_provider_failure(
                            runmat_accelerate_api::ProviderExecutionError::from(source)
                                .after_effect(),
                        ))
                    }
                    Err(source) => return Err(trace_provider_failure(source)),
                };
                if let Some(result) = reduced {
                    let diagonal_sibling = [&diagonal];
                    let protected_siblings = if diag_len == 1 {
                        &[][..]
                    } else {
                        &diagonal_sibling[..]
                    };
                    let result = admit_resident(
                        provider,
                        &handle,
                        result,
                        &[1, 1],
                        floating_contract(
                            storage,
                            Some(ProviderPrecision::F64),
                            if diag_len == 1 {
                                GpuOutputAliasPolicy::AllowInput
                            } else {
                                GpuOutputAliasPolicy::RequireDistinctAllocation
                            },
                        ),
                        protected_siblings,
                    )?;
                    let result = if runmat_accelerate_api::handle_precision(&handle)
                        == Some(ProviderPrecision::F32)
                    {
                        let cast = match provider.unary_single(&result).await {
                            Ok(cast) => Some(cast),
                            Err(source) if source.is_unsupported() => {
                                return Err(trace_provider_failure(source.after_effect()))
                            }
                            Err(source) => return Err(trace_provider_failure(source)),
                        };
                        if let Some(cast) = cast {
                            Some(admit_resident(
                                provider,
                                &handle,
                                cast,
                                &[1, 1],
                                floating_contract(
                                    storage,
                                    Some(ProviderPrecision::F32),
                                    GpuOutputAliasPolicy::RequireDistinctAllocation,
                                ),
                                &[&handle, &diagonal, &result],
                            )?)
                        } else {
                            None
                        }
                    } else {
                        Some(result)
                    };
                    if let Some(result) = result {
                        let result =
                            gpu_helpers::propagate_owned_output_provenance(result, [&handle])?;
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
    gpu_helpers::restore_class_preserving_value(&handle, host, NAME).map_err(trace_restore_failure)
}

pub(super) fn trace_restore_failure(source: RuntimeError) -> RuntimeError {
    let mut error = trace_error_with_message(format!("trace: {source}"), &TRACE_ERROR_INTERNAL);
    error.span = source.span;
    error.context = source.context.clone().with_builtin(NAME);
    error.gpu_gather_retry = source.gpu_gather_retry;
    error.source = Some(Box::new(source));
    error
}

fn trace_provider_failure(
    source: impl Into<runmat_accelerate_api::ProviderExecutionError>,
) -> RuntimeError {
    let mut error = trace_error(&TRACE_ERROR_INTERNAL);
    error = crate::runtime_error::with_provider_execution_source(error, source.into());
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

fn floating_contract(
    storage: GpuTensorStorage,
    precision: Option<ProviderPrecision>,
    alias: GpuOutputAliasPolicy,
) -> UnaryGpuOutputContract {
    UnaryGpuOutputContract {
        storage,
        precision,
        integer: None,
        logical: false,
        alias,
    }
}

fn admit_resident(
    provider: &'static dyn AccelProvider,
    original: &GpuTensorHandle,
    output: GpuTensorHandle,
    shape: &[usize],
    contract: UnaryGpuOutputContract,
    borrowed: &[&GpuTensorHandle],
) -> BuiltinResult<GpuTensorHandle> {
    let valid = contract.precision.is_some_and(|_| {
        gpu_helpers::unary_gpu_output_matches_shape(&output, original, provider, shape, contract)
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
