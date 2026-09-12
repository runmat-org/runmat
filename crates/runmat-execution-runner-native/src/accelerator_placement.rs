use runmat_execution::resource::AcceleratorRequest;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(crate) fn active_provider_resource() -> NativeExecutionResult<AcceleratorRequest> {
    let capability = active_hardware_capability()?;
    let class = capability.device.backend.ok_or_else(|| {
        configuration(
            "the active acceleration provider does not declare a scheduler resource class",
        )
    })?;
    let memory_bytes_each = capability.max_allocation_bytes.ok_or_else(|| {
        configuration("the active acceleration provider does not declare a bounded allocation size")
    })?;
    if memory_bytes_each == 0 {
        return Err(configuration(
            "the active acceleration provider declares zero allocatable memory",
        ));
    }
    let request = AcceleratorRequest {
        class,
        count: 1,
        memory_bytes_each,
    };
    request.validate().map_err(configuration)?;
    Ok(request)
}

pub(crate) fn validate_active_provider_for(
    requests: &[AcceleratorRequest],
) -> NativeExecutionResult<()> {
    if requests.is_empty() {
        return Ok(());
    }
    for request in requests {
        request.validate().map_err(configuration)?;
    }
    active_hardware_capability()?;
    Ok(())
}

pub(crate) fn validate_local_provider_for(
    requests: &[AcceleratorRequest],
) -> NativeExecutionResult<()> {
    let [request] = requests else {
        return Err(configuration(
            "local accelerated execution requires exactly one accelerator request",
        ));
    };
    request.validate().map_err(configuration)?;
    let capability = active_hardware_capability()?;
    if capability.device.backend.as_deref() != Some(request.class.as_str())
        || capability
            .max_allocation_bytes
            .is_none_or(|bytes| bytes < request.memory_bytes_each)
    {
        return Err(configuration(
            "the local acceleration provider differs from its admitted resource request",
        ));
    }
    Ok(())
}

fn active_hardware_capability(
) -> NativeExecutionResult<runmat_accelerate_api::ProviderCapabilitySnapshot> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        configuration("accelerated execution requires an initialized acceleration provider")
    })?;
    let capability = provider.capability_snapshot();
    if capability.device.backend.as_deref() == Some("inprocess") {
        return Err(configuration(
            "accelerated execution cannot use the in-process compatibility provider",
        ));
    }
    Ok(capability)
}

fn configuration(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Configuration(error.to_string())
}
