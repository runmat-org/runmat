use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, IntegerElementType, ProviderPrecision,
};
use runmat_types::IntegerClass;
use runmat_value::Value;

use super::class::IntegerClassExt;
use super::error::CastError;
use super::host::cast_host_value;

pub(super) async fn cast_gpu_value(
    handle: GpuTensorHandle,
    target: IntegerClass,
) -> Result<Value, CastError> {
    admit_gpu_value(&handle)?;
    let provider = crate::builtins::common::gpu_helpers::resolve_resident_compute_owner(&handle)
        .map_err(|decline| {
            retained_error(
                decline.with_domain_error(
                    crate::build_runtime_error("no acceleration provider owns the input handle")
                        .build(),
                ),
            )
        })?;

    let provenance = runmat_accelerate_api::handle_provenance(&handle)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return fallback_through_owner(provider, &handle, target).await;
    }

    let restore = crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let direct = provider
        .cast_to_integer(&handle, target.accelerator_type())
        .await;
    drop(restore);
    match direct {
        Ok(mut output) if valid_output(&output, &handle, provider, target.accelerator_type()) => {
            output = output.with_provenance(provenance).map_err(|error| {
                retained_error(
                    crate::build_runtime_error(format!(
                        "cannot preserve integer conversion intent: {error}"
                    ))
                    .with_source(error)
                    .build(),
                )
            })?;
            crate::builtins::common::gpu_helpers::resident_gpu_value(output).map_err(retained_error)
        }
        Ok(output) => Err(retained_error(
            crate::builtins::common::gpu_helpers::reject_gpu_output(
                output,
                crate::build_runtime_error(
                    "provider returned an invalid integer conversion result",
                )
                .build(),
            ),
        )),

        Err(error) if error.is_unsupported() => {
            fallback_through_owner(provider, &handle, target).await
        }
        Err(error) => {
            let contextual = crate::build_runtime_error(error.to_string())
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build();
            Err(retained_error(
                crate::runtime_error::with_provider_execution_source(contextual, error),
            ))
        }
    }
}

/// Side-effect-free resident admission, also used before operand preparation.
pub(super) fn admit_gpu_value(handle: &GpuTensorHandle) -> Result<(), CastError> {
    if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err() {
        return Err(CastError::Internal(
            "no acceleration provider owns the input handle".into(),
        ));
    }
    validate_input_metadata(handle)
}

async fn fallback_through_owner(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    target: IntegerClass,
) -> Result<Value, CastError> {
    let gathered = crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
        provider, handle,
    )
    .await;
    let converted = cast_host_value(gathered.map_err(retained_error)?, target)?;
    // Host conversion collapses real singletons to Int. Resident conversion
    // must retain the source's complete shape and restore native integer data.
    let converted = match converted {
        Value::Int(value) => Value::Tensor(
            runmat_value::Tensor::new_integer(target.storage(vec![value]), handle.shape().to_vec())
                .map_err(CastError::Internal)?,
        ),
        value => value,
    };
    crate::builtins::common::gpu_helpers::restore_class_preserving_value(
        handle,
        converted,
        target.class_name(),
    )
    .map_err(retained_error)
}

fn validate_input_metadata(handle: &GpuTensorHandle) -> Result<(), CastError> {
    let integer = runmat_accelerate_api::handle_integer_type(handle);
    let logical = runmat_accelerate_api::handle_is_logical(handle);
    let precision = runmat_accelerate_api::handle_precision(handle);
    if handle
        .descriptor()
        .storage
        .is_some_and(|storage| representation_is_consistent(storage, precision, integer, logical))
        && crate::builtins::common::gpu_helpers::gpu_class_metadata_matches(
            handle, precision, integer, logical,
        )
    {
        Ok(())
    } else {
        Err(CastError::Internal(
            "input handle has contradictory class metadata".into(),
        ))
    }
}

fn representation_is_consistent(
    storage: GpuTensorStorage,
    precision: Option<ProviderPrecision>,
    integer: Option<IntegerElementType>,
    logical: bool,
) -> bool {
    let numeric_storage = matches!(
        storage,
        GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
    );
    if integer.is_some() {
        numeric_storage
            && precision.is_none()
            && (!logical
                || (integer == Some(IntegerElementType::U8) && storage == GpuTensorStorage::Real))
    } else {
        numeric_storage && precision.is_some() && (!logical || storage == GpuTensorStorage::Real)
    }
}

fn valid_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    target: IntegerElementType,
) -> bool {
    output.descriptor().storage == Some(GpuTensorStorage::Real)
        && crate::builtins::common::gpu_helpers::unary_gpu_output_matches(
            output,
            input,
            provider,
            crate::builtins::common::gpu_helpers::UnaryGpuOutputContract {
                storage: GpuTensorStorage::Real,
                precision: None,
                integer: Some(target),
                logical: false,
                alias: crate::builtins::common::gpu_helpers::GpuOutputAliasPolicy::AllowInput,
            },
        )
}

fn retained_error(error: crate::RuntimeError) -> CastError {
    CastError::Runtime(Box::new(error))
}

#[cfg(test)]
mod tests {
    use super::representation_is_consistent;
    use runmat_accelerate_api::{GpuTensorStorage, ProviderPrecision};

    #[test]
    fn floating_input_requires_a_physical_precision() {
        for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
            assert!(representation_is_consistent(
                GpuTensorStorage::Real,
                Some(precision),
                None,
                false,
            ));
        }
        assert!(!representation_is_consistent(
            GpuTensorStorage::Real,
            None,
            None,
            false,
        ));
    }
}
