use super::host::single_from_gathered;
use super::single_error_with_detail;
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderPrecision,
};
use runmat_builtins::SINGLE_ERROR_INTERNAL;
use runmat_value::Value;

#[cfg(test)]
use single_from_gpu as cast_from_gpu;
#[cfg(test)]
const CAST_PRECISION: ProviderPrecision = ProviderPrecision::F32;

pub(super) async fn single_from_gpu(
    handle: GpuTensorHandle,
    borrowed: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    let mut protected = borrowed.to_vec();
    protected.push(&handle);
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        single_error_with_detail(&SINGLE_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    let input_element =
        gpu_helpers::expected_handle_numeric_element_type(&handle).map_err(|_| {
            single_error_with_detail(
                &SINGLE_ERROR_INTERNAL,
                "GPU input class metadata contradicts its physical storage",
            )
        })?;

    if handle.descriptor().storage.is_none() {
        return Err(single_error_with_detail(
            &runmat_builtins::SINGLE_ERROR_INVALID_INPUT,
            "GPU input requires explicit storage",
        ));
    }
    let _metadata_guard = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    let direct_eligible = matches!(
        input_element,
        NumericElementType::F32 | NumericElementType::F64
    ) && !runmat_accelerate_api::handle_is_logical(&handle)
        && runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::Real;
    if direct_eligible {
        let direct = provider.unary_single(&handle).await;
        _metadata_guard.restore();
        match direct {
            Ok(mut output)
                if valid_output(&output, &handle, provider)
                    && !borrowed.iter().any(|input| {
                        !input.shares_allocation_with(&handle)
                            && output.shares_allocation_with(input)
                    }) =>
            {
                output = output.with_provenance(input_provenance).map_err(|error| {
                    crate::build_runtime_error(format!(
                        "single: cannot preserve input intent: {error}"
                    ))
                    .with_builtin("single")
                    .with_source(error)
                    .build()
                })?;
                let output = gpu_helpers::resident_gpu_value(output)?;
                if let Some(recorder) =
                    runmat_accelerate_api::observation::current_provider_observation()
                {
                    recorder.confirm_provider_execution(
                        provider.device_id(),
                        &runmat_accelerate_api::ProviderOperationIdentity::new(
                            runmat_builtins::SINGLE_CATALOG_ENTRY.identity.name,
                        ),
                    );
                }
                return Ok(output);
            }
            Ok(output) => {
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    single_error_with_detail(
                        &SINGLE_ERROR_INTERNAL,
                        "provider unary_single returned malformed output",
                    ),
                ));
            }
            Err(error) if error.is_unsupported() => {}
            Err(error) => {
                let mut failure = single_error_with_detail(
                    &SINGLE_ERROR_INTERNAL,
                    format!("provider unary_single failed: {error}"),
                );
                failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(crate::runtime_error::with_provider_execution_source(
                    failure, error,
                ));
            }
        }
    }

    if let Some(recorder) = runmat_accelerate_api::observation::current_provider_observation() {
        recorder.execution_disposition(
            runmat_accelerate_api::observation::ProviderExecutionDisposition::SharedRuntimeFallback,
        );
    }
    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    _metadata_guard.restore();
    let gathered = gathered_result.map_err(contextual_upload_error)?;
    let converted = single_from_gathered(gathered)?;
    let upload = match &converted {
        Value::Tensor(tensor) => {
            gpu_helpers::upload_tensor_protecting(provider, tensor, &protected)
                .map_err(contextual_upload_error)
        }
        Value::ComplexTensor(tensor) => {
            gpu_helpers::upload_complex_tensor_protecting(provider, tensor, &protected)
        }
        other => {
            return Err(single_error_with_detail(
                &SINGLE_ERROR_INTERNAL,
                format!("single fallback produced unsupported value {other:?}"),
            ));
        }
    };
    let mut output = upload?;
    if !valid_output(&output, &handle, provider) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            single_error_with_detail(
                &SINGLE_ERROR_INTERNAL,
                "provider upload returned malformed single fallback output",
            ),
        ));
    }
    output = output.with_provenance(input_provenance).map_err(|error| {
        crate::build_runtime_error(format!("single: cannot preserve input intent: {error}"))
            .with_builtin("single")
            .with_source(error)
            .build()
    })?;
    if runmat_accelerate_api::handle_storage(&output) == GpuTensorStorage::ComplexInterleaved {
        gpu_helpers::complex_gpu_value(output)
    } else {
        gpu_helpers::resident_gpu_value(output)
    }
}

fn valid_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let expected_storage = runmat_accelerate_api::handle_storage(input);
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: expected_storage,
            precision: Some(ProviderPrecision::F32),
            integer: None,
            logical: false,
            // Immutable same-type casts may return the exact input view. Uploads
            // independently require a distinct physical allocation at their boundary.
            alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
        },
    )
}

fn contextual_upload_error(error: runmat_async::RuntimeError) -> runmat_async::RuntimeError {
    let mut contextual = single_error_with_detail(&SINGLE_ERROR_INTERNAL, error.message());
    contextual.source = Some(Box::new(error));
    contextual
}

#[cfg(test)]
mod owner_tests {
    use super::*;

    #[test]
    fn immutable_cast_adopts_exact_input_alias_without_retiring_caller() {
        super::super::super::cast_owner_tests::immutable_cast_adopts_exact_input_alias_without_retiring_caller(
            CAST_PRECISION,
            |handle, borrowed| futures::executor::block_on(cast_from_gpu(handle, borrowed)),
        );
    }

    #[test]
    fn immutable_cast_rejects_unrelated_borrowed_prototype_alias() {
        super::super::super::cast_owner_tests::immutable_cast_rejects_unrelated_borrowed_prototype_alias(
            CAST_PRECISION,
            |handle, borrowed| futures::executor::block_on(cast_from_gpu(handle, borrowed)),
        );
    }
}
