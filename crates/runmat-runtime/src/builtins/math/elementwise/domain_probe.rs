use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

use crate::builtins::common::gpu_helpers;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GpuLowerBoundResult {
    Below,
    AtOrAbove,
    Unknown,
}

#[derive(Debug)]
pub(super) enum GpuDomainProbeError {
    ProviderReduce(runmat_accelerate_api::ProviderExecutionError),
    ProviderDownload(Box<crate::RuntimeError>),
    AliasedReduction,
    MalformedInput,
    MalformedReduction,
}

impl std::fmt::Display for GpuDomainProbeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProviderReduce(error) => write!(formatter, "provider reduce_min failed: {error}"),
            Self::ProviderDownload(error) => {
                write!(formatter, "provider reduce_min download failed: {error}")
            }
            Self::AliasedReduction => formatter.write_str("provider reduce_min aliased its input"),
            Self::MalformedInput => {
                formatter.write_str("range probe requires an owned real floating input")
            }
            Self::MalformedReduction => {
                formatter.write_str("provider reduce_min returned malformed output")
            }
        }
    }
}

impl std::error::Error for GpuDomainProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ProviderReduce(error) => Some(error),
            Self::ProviderDownload(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

/// Determine whether any real resident input value falls below a caller-defined bound.
///
/// A provider that does not implement `reduce_min` leaves the requirement
/// unknown so the caller can gather safely. Any other provider failure is
/// returned and must remain visible to the user.
pub(super) async fn probe_gpu_lower_bound(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    complex_below: f64,
) -> Result<GpuLowerBoundResult, GpuDomainProbeError> {
    let element = gpu_helpers::expected_handle_numeric_element_type(handle)
        .map_err(|_| GpuDomainProbeError::MalformedInput)?;
    if gpu_helpers::exact_provider_for_handle(handle)
        .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        || !gpu_helpers::numeric_descriptor_matches(handle, element, GpuTensorStorage::Real)
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Err(GpuDomainProbeError::MalformedInput);
    }
    handle
        .allocation_lease()
        .validate_live()
        .map_err(|_| GpuDomainProbeError::MalformedInput)?;
    if handle.shape().contains(&0) {
        return Ok(GpuLowerBoundResult::AtOrAbove);
    }
    if handle
        .shape()
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .is_none()
    {
        return Err(GpuDomainProbeError::MalformedInput);
    }

    let reduction = {
        let _metadata = gpu_helpers::ResidencyRestoreGuard::new(handle);
        provider.reduce_min(handle).await
    };
    let minimum = match reduction {
        Ok(handle) => handle,
        Err(error) if error.is_unsupported() => return Ok(GpuLowerBoundResult::Unknown),
        Err(error) => return Err(GpuDomainProbeError::ProviderReduce(error)),
    };

    if minimum.shares_allocation_with(handle) {
        return Err(GpuDomainProbeError::AliasedReduction);
    }
    if minimum.device_id() != handle.device_id()
        || gpu_helpers::exact_provider_for_handle(&minimum)
            .is_none_or(|owner| !runmat_accelerate_api::same_provider_instance(owner, provider))
        || runmat_accelerate_api::handle_storage(&minimum) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_precision(&minimum)
            != runmat_accelerate_api::handle_precision(handle)
        || runmat_accelerate_api::handle_integer_type(&minimum).is_some()
        || runmat_accelerate_api::handle_is_logical(&minimum)
        || !gpu_helpers::numeric_descriptor_matches(&minimum, element, GpuTensorStorage::Real)
        || minimum
            .shape()
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            != Some(1)
    {
        return Err(GpuDomainProbeError::MalformedReduction);
    }

    let _metadata = gpu_helpers::ResidencyRestoreGuard::new(handle);
    let host = gpu_helpers::download_floating_projection_async(provider, &minimum).await;
    let host = host.map_err(|error| GpuDomainProbeError::ProviderDownload(Box::new(error)))?;
    if host.data.iter().any(|value| value.is_nan()) {
        return Ok(GpuLowerBoundResult::Unknown);
    }
    if host.data.iter().any(|value| *value < complex_below) {
        Ok(GpuLowerBoundResult::Below)
    } else {
        Ok(GpuLowerBoundResult::AtOrAbove)
    }
}

#[cfg(test)]
#[path = "domain_probe/owning_tests.rs"]
mod owning_tests;

#[cfg(test)]
#[path = "domain_probe/error_tests.rs"]
mod error_tests;
