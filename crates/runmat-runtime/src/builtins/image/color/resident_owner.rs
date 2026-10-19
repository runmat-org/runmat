//! Color-domain resident owner and live-allocation admission, independent of sample class.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

use super::common::builtin_error;
use crate::BuiltinResult;

pub(crate) fn exact_owner(
    handle: &GpuTensorHandle,
    name: &'static str,
) -> BuiltinResult<&'static dyn AccelProvider> {
    let owner = runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
        builtin_error(
            name,
            format!("{name}: no acceleration provider owns the input handle"),
        )
    })?;
    if owner.device_id() != handle.device_id() {
        return Err(builtin_error(
            name,
            format!(
                "{name}: input handle device {} is not owned by the selected provider device {}",
                handle.device_id(),
                owner.device_id()
            ),
        ));
    }
    Ok(owner)
}

pub(crate) fn validate_live(handle: &GpuTensorHandle, name: &'static str) -> BuiltinResult<()> {
    exact_owner(handle, name)?;
    handle.allocation_lease().validate_live().map_err(|error| {
        crate::runtime_error::with_provider_execution_source(
            builtin_error(
                name,
                format!("{name}: invalid resident allocation: {error}"),
            ),
            runmat_accelerate_api::ProviderExecutionError::from_source(error),
        )
    })
}
