//! Resident metadata integrity and non-destructive owner-preserving download.

use super::{
    build_runtime_error, handle_integer_type, handle_is_logical, handle_storage, BuiltinResult,
    GpuTensorStorage, Value, VALIDATION_ERROR_PROVIDER_OWNERSHIP,
    VALIDATION_ERROR_PROVIDER_PAYLOAD,
};

async fn resident_host_value(value: &Value) -> BuiltinResult<Option<Value>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    let readback =
        runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
            build_runtime_error(VALIDATION_ERROR_PROVIDER_OWNERSHIP.message)
                .with_source(runmat_accelerate_api::ProviderExecutionError::from(source))
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
    let runtime = crate::context::legacy::active();
    crate::builtins::common::gpu_helpers::download_readback_value_in_context(
        runtime.as_ref(),
        &readback,
    )
    .await
    .map(Some)
}

pub(super) async fn host_value(value: &Value) -> BuiltinResult<Value> {
    validate_resident_metadata(value)?;
    Ok(match resident_host_value(value).await? {
        Some(value) => value,
        None => value.clone(),
    })
}

fn explicit_resident(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_provenance(handle) == Some(runmat_accelerate_api::GpuHandleProvenance::Explicit))
}

pub fn ensure_resident_extension(
    value: &Value,
    extension: &runmat_builtins::BuiltinExtensionDescriptor,
    builtin: &str,
) -> BuiltinResult<()> {
    if !super::cell_leaves::CellLeaves::new(value).any(explicit_resident) {
        return Ok(());
    }
    crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)
}

pub fn validate_resident_metadata(value: &Value) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        return Ok(());
    };
    runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
        build_runtime_error(VALIDATION_ERROR_PROVIDER_OWNERSHIP.message)
            .with_identifier(
                VALIDATION_ERROR_PROVIDER_OWNERSHIP
                    .identifier
                    .expect("validator provider-ownership descriptor identifier"),
            )
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(runmat_accelerate_api::ProviderExecutionError::from(source))
            .build()
    })?;
    let storage = handle_storage(handle);
    let integer = handle_integer_type(handle);
    let logical = handle_is_logical(handle);
    let precision = runmat_accelerate_api::handle_precision(handle);
    let expected_class = crate::builtins::common::gpu_helpers::expected_gpu_class_identity(
        precision, integer, logical,
    );
    let class_valid = runmat_accelerate_api::handle_class_identity(handle)
        .as_ref()
        .is_none_or(|class_identity| Some(class_identity) == expected_class.as_ref());
    let physical_valid = if integer.is_some() {
        storage == GpuTensorStorage::Real && precision.is_none() && !logical
    } else if logical {
        storage == GpuTensorStorage::Real && precision.is_some()
    } else {
        precision.is_some()
    };
    if !physical_valid || !class_valid {
        return Err(
            build_runtime_error(VALIDATION_ERROR_PROVIDER_PAYLOAD.message)
                .with_identifier(
                    VALIDATION_ERROR_PROVIDER_PAYLOAD
                        .identifier
                        .expect("validator provider-payload descriptor identifier"),
                )
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
                .into(),
        );
    }
    Ok(())
}
