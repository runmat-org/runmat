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
    ProviderReduce(anyhow::Error),
    ProviderDownload(Box<crate::RuntimeError>),
    AliasedReduction,
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
            Self::MalformedReduction => {
                formatter.write_str("provider reduce_min returned malformed output")
            }
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
    if handle.shape.iter().product::<usize>() == 0 {
        return Ok(GpuLowerBoundResult::AtOrAbove);
    }

    let input_metadata = gpu_helpers::snapshot_handle_metadata(handle);
    let reduction = provider.reduce_min(handle).await;
    gpu_helpers::restore_handle_metadata(handle, &input_metadata);
    let minimum = match reduction {
        Ok(handle) => handle,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
            return Ok(GpuLowerBoundResult::Unknown)
        }
        Err(error) => return Err(GpuDomainProbeError::ProviderReduce(error)),
    };

    if gpu_helpers::same_gpu_handle(&minimum, handle) {
        return Err(GpuDomainProbeError::AliasedReduction);
    }
    if minimum.device_id != handle.device_id
        || gpu_helpers::exact_provider_for_handle(&minimum)
            .is_none_or(|owner| !std::ptr::eq(owner, provider))
        || runmat_accelerate_api::handle_storage(&minimum) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_precision(&minimum)
            != runmat_accelerate_api::handle_precision(handle)
        || runmat_accelerate_api::handle_integer_type(&minimum).is_some()
        || runmat_accelerate_api::handle_is_logical(&minimum)
        || minimum.shape.iter().product::<usize>() != 1
    {
        gpu_helpers::free_unprotected_exact_owner(&minimum, &[handle]);
        return Err(GpuDomainProbeError::MalformedReduction);
    }

    let host = gpu_helpers::download_floating_projection_async(provider, &minimum).await;
    gpu_helpers::free_unprotected_exact_owner(&minimum, &[handle]);
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
