use runmat_execution::resource::{AcceleratorDevice, AcceleratorFeature, AcceleratorRequest};

use crate::{NativeExecutionError, NativeExecutionResult};

/// One whole-device request for the active provider, in the same class,
/// provider and capacity terms the scheduler inventory reports for it.
pub fn active_provider_resource() -> NativeExecutionResult<AcceleratorRequest> {
    let device = active_execution_device()?;
    let required_features = if device.features.contains(&AcceleratorFeature::Compute) {
        std::collections::BTreeSet::from([AcceleratorFeature::Compute])
    } else {
        Default::default()
    };
    let request = AcceleratorRequest {
        class: device.class.clone(),
        count: 1,
        minimum_allocation_bytes: device.max_allocation_bytes,
        provider: Some(device.provider.clone()),
        required_features,
    };
    request.validate().map_err(configuration)?;
    if !request.matches(&device) {
        return Err(configuration(
            "the active acceleration provider cannot satisfy its own resource request",
        ));
    }
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

fn active_execution_device() -> NativeExecutionResult<AcceleratorDevice> {
    active_hardware_capability()?;
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        configuration("accelerated execution requires an initialized acceleration provider")
    })?;
    provider
        .execution_accelerator_device(1)
        .map_err(configuration)?
        .ok_or_else(|| {
            configuration("the active acceleration provider does not expose an execution device")
        })
}

fn configuration(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Configuration(error.to_string())
}
