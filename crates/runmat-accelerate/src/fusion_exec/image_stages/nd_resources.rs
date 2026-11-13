//! Exact resource admission for the selected first N-D mean hook only.
//! These estimates never claim the later image stages are feasible.

use super::*;
use runmat_accelerate_api::ProviderResourceEstimate;

fn selected_owner(provider: &'static dyn AccelProvider, handle: &GpuTensorHandle) -> bool {
    handle.device_id() == provider.device_id()
        && exact_provider_for_handle(handle)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}

fn checked_sum(parts: &[u64]) -> Option<u64> {
    parts
        .iter()
        .try_fold(0u64, |total, part| total.checked_add(*part))
}

pub(super) fn warm_mean(
    provider: &'static dyn AccelProvider,
    source: &GpuTensorHandle,
    pinned: &ProviderPinnedNdMean,
    dims: &[usize],
    shape: &[usize],
) -> ProviderExecutionResult<ProviderResourceEstimate> {
    let live_source = pinned
        .validate_live_source(source)
        .map_err(ProviderExecutionError::from_source)?;
    if !live_source
        || !selected_owner(provider, source)
        || pinned.dims() != dims
        || ![pinned.mean(), pinned.ex2()].iter().all(|handle| {
            selected_owner(provider, handle)
                && handle.shape() == shape
                && handle.physical().storage() == GpuTensorStorage::Real
                && handle.physical().element_type() == NumericElementType::F32
                && !runmat_accelerate_api::handle_is_logical(handle)
                && !runmat_accelerate_api::handle_is_transposed(handle)
                && gpu_class_metadata_matches(handle, Some(ProviderPrecision::F32), None, false)
        })
    {
        return Err(anyhow!("selected warm N-D mean has invalid source or pair").into());
    }
    let requested = checked_sum(&[
        pinned.mean().physical().requested_bytes(),
        pinned.ex2().physical().requested_bytes(),
    ])
    .ok_or_else(|| anyhow!("selected warm N-D mean byte count overflow"))?;
    if requested > pinned.retained_pair_bytes() {
        return Err(anyhow!("selected warm N-D mean understates retained bytes").into());
    }
    let peak = checked_sum(&[source.physical().requested_bytes(), requested])
        .ok_or_else(|| anyhow!("selected warm N-D mean peak overflow"))?;
    Ok(ProviderResourceEstimate {
        transient_bytes: Some(peak),
        output_bytes: Some(pinned.mean().physical().requested_bytes()),
        dispatches: Some(0),
    })
}

pub(super) fn cold_mean(
    provider: &'static dyn AccelProvider,
    source: &GpuTensorHandle,
    selected: &ProviderColdNdMean,
    dims: &[usize],
    shape: &[usize],
) -> ProviderExecutionResult<ProviderResourceEstimate> {
    let live_source = selected
        .validate_live_source(source)
        .map_err(ProviderExecutionError::from_source)?;
    if !live_source
        || !selected_owner(provider, source)
        || selected.dims() != dims
        || selected.output_shape() != shape
    {
        return Err(anyhow!("selected cold N-D mean source or shape changed").into());
    }
    let pair = selected
        .requested_pair_bytes()
        .ok_or_else(|| anyhow!("selected cold N-D mean pair byte overflow"))?;
    let peak = checked_sum(&[
        source.physical().requested_bytes(),
        pair,
        selected.requested_uniform_bytes(),
    ])
    .ok_or_else(|| anyhow!("selected cold N-D mean peak overflow"))?;
    Ok(ProviderResourceEstimate {
        transient_bytes: Some(peak),
        output_bytes: Some(pair / 2),
        dispatches: Some(1),
    })
}

pub(super) fn executed_cold_mean(
    provider: &'static dyn AccelProvider,
    selected: &ProviderResourceEstimate,
    requested_pair_bytes: u64,
    requested_uniform_bytes: u64,
    executed: &runmat_accelerate_api::ProviderExecutedColdNdMean,
    shape: &[usize],
) -> ProviderExecutionResult<()> {
    for handle in [&executed.mean, &executed.ex2] {
        handle
            .allocation_lease()
            .validate_live()
            .map_err(|source| ProviderExecutionError::from_source(source).after_effect())?;
    }
    let requested = checked_sum(&[
        executed.mean.physical().requested_bytes(),
        executed.ex2.physical().requested_bytes(),
    ]);
    if !selected_owner(provider, &executed.mean)
        || !selected_owner(provider, &executed.ex2)
        || executed.mean.shape() != shape
        || executed.ex2.shape() != shape
        || [&executed.mean, &executed.ex2].iter().any(|handle| {
            handle.physical().storage() != GpuTensorStorage::Real
                || handle.physical().element_type() != NumericElementType::F32
                || runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_is_transposed(handle)
                || !gpu_class_metadata_matches(handle, Some(ProviderPrecision::F32), None, false)
        })
        || executed.mean.allocation_identity() == executed.ex2.allocation_identity()
        || selected.output_bytes != Some(executed.mean.physical().requested_bytes())
        || selected.dispatches != Some(executed.selected_submissions)
        || executed.requested_pair_bytes != requested_pair_bytes
        || executed.requested_uniform_bytes != requested_uniform_bytes
        || checked_sum(&[requested_pair_bytes, requested_uniform_bytes])
            .zip(selected.transient_bytes)
            .is_none_or(|(temporary, peak)| temporary > peak)
        || requested != Some(executed.requested_pair_bytes)
        || executed.observed_pair_bytes < executed.requested_pair_bytes
    {
        return Err(ProviderExecutionError::from(anyhow!(
            "selected cold N-D mean completion disagrees with live resources"
        ))
        .after_effect());
    }
    Ok(())
}
