//! Neutral host quote for provider-owned numeric registration metadata.
//! Device buffer bytes and provider-internal work remain separate authorities.

use runmat_accelerate_api::AccelProvider;

use super::{HostAllocationError, HostAllocationPlan, HostRetainedViewPermit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderRegistrationHostQuote {
    pub preparation_bytes: u64,
    pub retained_bytes: u64,
    pub registrations: usize,
}

/// A missing provider bound is unknown, never zero. One quoted registration
/// retains its record, lease and view; the table and issuer have independent
/// lifetimes. Derived views must be counted separately from registrations.
pub fn provider_registration_host_quote(
    provider: &dyn AccelProvider,
    rank: usize,
    registrations: usize,
    derived_views: usize,
) -> Result<Option<ProviderRegistrationHostQuote>, HostAllocationError> {
    let Some(payload) = provider.registered_output_host_metadata(rank) else {
        return Ok(None);
    };
    let Some(table_backing) = provider.output_registration_table_upper_bytes(registrations) else {
        return Ok(None);
    };
    let Some(provider_preparation) = provider.output_registration_preparation_host_bytes() else {
        return Ok(None);
    };
    let record = HostAllocationPlan::retained_metadata_bytes(payload.record_payload_bytes)?;
    let lease = HostAllocationPlan::retained_metadata_bytes(payload.lease_payload_bytes)?;
    let view = HostAllocationPlan::retained_view_bytes(payload.view_payload_bytes)?;
    let registration_count =
        u64::try_from(registrations).map_err(|_| HostAllocationError::Overflow)?;
    let derived_view_count =
        u64::try_from(derived_views).map_err(|_| HostAllocationError::Overflow)?;
    let table_backing = if table_backing == 0 {
        0
    } else {
        HostAllocationPlan::retained_metadata_bytes(table_backing)?
    };
    let retained_bytes = record
        .checked_add(view)
        .and_then(|per| per.checked_add(lease))
        .and_then(|per| per.checked_mul(registration_count))
        .and_then(|base| base.checked_add(view.checked_mul(derived_view_count)?))
        .and_then(|base| base.checked_add(table_backing))
        .and_then(|base| base.checked_add(HostAllocationPlan::retained_view_issuer_bytes().ok()?))
        .ok_or(HostAllocationError::Overflow)?;
    let permit = std::mem::size_of::<HostRetainedViewPermit>()
        .checked_add(std::mem::align_of::<HostRetainedViewPermit>())
        .ok_or(HostAllocationError::Overflow)?;
    let permit = u64::try_from(permit).map_err(|_| HostAllocationError::Overflow)?;
    let preparation_bytes = provider_preparation
        .checked_add(permit)
        .ok_or(HostAllocationError::Overflow)?;
    Ok(Some(ProviderRegistrationHostQuote {
        preparation_bytes,
        retained_bytes,
        registrations,
    }))
}
