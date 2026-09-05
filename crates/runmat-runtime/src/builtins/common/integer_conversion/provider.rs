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
    let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(&handle)
        .ok_or_else(|| {
            CastError::Internal("no acceleration provider owns the input handle".into())
        })?;
    validate_input_metadata(&handle)?;

    let input_metadata = crate::builtins::common::gpu_helpers::snapshot_handle_metadata(&handle);
    let provenance = runmat_accelerate_api::handle_provenance(&handle)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return fallback_through_owner(provider, &handle, target, &input_metadata).await;
    }

    let direct = provider
        .cast_to_integer(&handle, target.accelerator_type())
        .await;
    crate::builtins::common::gpu_helpers::restore_handle_metadata(&handle, &input_metadata);
    match direct {
        Ok(mut output) if valid_output(&output, &handle, provider, target.accelerator_type()) => {
            runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
            Ok(crate::builtins::common::gpu_helpers::resident_gpu_value(
                output,
            ))
        }
        Ok(output) => {
            crate::builtins::common::gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
            Err(CastError::Internal(
                "provider returned an invalid integer conversion result".into(),
            ))
        }
        Err(error)
            if crate::builtins::common::gpu_helpers::provider_hook_is_unsupported(&error) =>
        {
            fallback_through_owner(provider, &handle, target, &input_metadata).await
        }
        Err(error) => Err(CastError::Internal(error.to_string())),
    }
}

async fn fallback_through_owner(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    target: IntegerClass,
    input_metadata: &crate::builtins::common::gpu_helpers::GpuHandleMetadataSnapshot,
) -> Result<Value, CastError> {
    let gathered = crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
        provider, handle,
    )
    .await;
    crate::builtins::common::gpu_helpers::restore_handle_metadata(handle, input_metadata);
    let converted = cast_host_value(
        gathered.map_err(|error| CastError::Internal(error.message().to_string()))?,
        target,
    )?;
    crate::builtins::common::gpu_helpers::restore_class_preserving_value(
        handle,
        converted,
        target.class_name(),
    )
    .map_err(|error| CastError::Internal(error.message().to_string()))
}

fn validate_input_metadata(handle: &GpuTensorHandle) -> Result<(), CastError> {
    let integer = runmat_accelerate_api::handle_integer_type(handle);
    let logical = runmat_accelerate_api::handle_is_logical(handle);
    let precision = runmat_accelerate_api::handle_precision(handle);
    let storage = runmat_accelerate_api::handle_storage(handle);
    if representation_is_consistent(storage, precision, integer, logical)
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
        numeric_storage && precision.is_none() && !logical
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
    crate::builtins::common::gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        crate::builtins::common::gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: None,
            integer: Some(target),
            logical: false,
            alias: crate::builtins::common::gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    )
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
