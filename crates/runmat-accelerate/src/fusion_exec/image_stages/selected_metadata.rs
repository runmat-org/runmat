//! Finite selected-image provider metadata authority. The source plan quotes
//! exact request-derived registrations before placement; only the chosen GPU
//! route reserves headroom before its first provider effect.

use super::*;
use runmat_runtime::context::host_allocation::{
    provider_registration_host_quote, HostAllocationPlan, HostAllocationReceipt,
};

pub(super) enum PreparedInput {
    Resident(Box<PreparedImageStages>),
    Host(Box<HostSingleImageSource>),
}

#[derive(Clone, Copy)]
pub(super) struct FiniteMetadataQuote {
    pub retained_bytes: u64,
    pub workspace_bytes: u64,
    pub registrations: usize,
}

/// The finite receipt owns headroom through selection and execution; the
/// provider permit may move into the selected stage without dropping it.
pub(super) struct SelectedMetadataAdmission {
    receipt: Option<HostAllocationReceipt>,
    permit: Option<Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>>,
}

impl SelectedMetadataAdmission {
    pub(super) fn permit(
        &self,
    ) -> Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission> {
        self.permit.as_deref()
    }

    pub(super) fn take_permit(
        &mut self,
    ) -> Option<Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>> {
        self.permit.take()
    }

    pub(super) fn into_receipt(self) -> Option<HostAllocationReceipt> {
        self.receipt
    }
}

pub(super) fn quote(
    provider: &dyn AccelProvider,
    prepared: &mut PreparedInput,
) -> ProviderExecutionResult<Option<FiniteMetadataQuote>> {
    let (registrations, derived_views, power_workspace) = match &*prepared {
        PreparedInput::Resident(stages) => (
            stages.finite_metadata_registration_count(),
            stages.finite_metadata_derived_view_count(),
            stages.finite_power_workspace_bytes(),
        ),
        PreparedInput::Host(host) => (
            host.finite_metadata_registration_count(),
            host.finite_metadata_derived_view_count(),
            host.finite_power_workspace_bytes(),
        ),
    };
    let (Some(registrations), Some(derived_views), Some(power_workspace)) =
        (registrations, derived_views, power_workspace)
    else {
        return Ok(None);
    };
    let Some(registration) =
        provider_registration_host_quote(provider, 3, registrations, derived_views)
            .map_err(ProviderExecutionError::from_source)?
    else {
        return Ok(None);
    };
    let workspace_bytes = power_workspace
        .checked_add(registration.preparation_bytes)
        .ok_or_else(|| {
            ProviderExecutionError::from(anyhow!(
                "selected image provider host workspace extent overflow"
            ))
        })?;
    let retained_bytes = registration.retained_bytes;
    match prepared {
        PreparedInput::Resident(stages) => {
            stages.set_provider_metadata_bytes(retained_bytes);
            stages.set_provider_power_workspace_bytes(workspace_bytes);
        }
        PreparedInput::Host(host) => {
            host.set_provider_metadata_bytes(retained_bytes);
            host.set_provider_power_workspace_bytes(workspace_bytes);
        }
    }
    Ok(Some(FiniteMetadataQuote {
        retained_bytes,
        workspace_bytes,
        registrations,
    }))
}

pub(super) fn admit_selected(
    provider: &dyn AccelProvider,
    runtime: &runmat_runtime::context::RuntimeContext,
    quote: Option<FiniteMetadataQuote>,
) -> ProviderExecutionResult<SelectedMetadataAdmission> {
    let Some(quote) = quote else {
        return Ok(SelectedMetadataAdmission {
            receipt: None,
            permit: None,
        });
    };
    let plan = HostAllocationPlan::new(quote.workspace_bytes, quote.retained_bytes)
        .map_err(ProviderExecutionError::from_source)?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(ProviderExecutionError::from_source)?
        .ok_or_else(|| {
            ProviderExecutionError::from(anyhow!(
                "finite selected image lost its metadata allocation owner"
            ))
        })?;
    let permit = receipt
        .retained_view_permit(runtime)
        .map_err(ProviderExecutionError::from_source)?;
    let prepared = provider.prepare_output_host_metadata(Box::new(permit), quote.registrations)?;
    Ok(SelectedMetadataAdmission {
        receipt: Some(receipt),
        permit: Some(prepared),
    })
}

pub(super) fn fund_cold_cache(
    runtime: &runmat_runtime::context::RuntimeContext,
    prepared: &mut PreparedImageStages,
    upload_submissions: u32,
) -> ProviderExecutionResult<Option<HostAllocationReceipt>> {
    let Some(rank) = prepared.session.cold_cache_view_rank() else {
        return Ok(None);
    };
    let after_upload = |failure: ProviderExecutionError| {
        if upload_submissions != 0 {
            failure.after_effect()
        } else {
            failure
        }
    };
    let metadata = prepared
        .session
        .cold_cache_metadata_payload_bytes()
        .ok_or_else(|| {
            after_upload(ProviderExecutionError::from(anyhow!(
                "cold cache metadata unavailable"
            )))
        })?;
    let bytes = host_memory::cold_cache_retained_bytes(rank, metadata)
        .map_err(|error| after_upload(ProviderExecutionError::from_source(error)))?;
    let plan = HostAllocationPlan::new(0, bytes)
        .map_err(|error| after_upload(ProviderExecutionError::from_source(error)))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|error| after_upload(ProviderExecutionError::from_source(error)))?
        .ok_or_else(|| {
            ProviderExecutionError::from(anyhow!(
                "finite image cold cache lost its managed allocation owner"
            ))
        })?;
    let permit = receipt
        .retained_view_permit(runtime)
        .map_err(|error| after_upload(ProviderExecutionError::from_source(error)))?;
    prepared
        .session_mut()
        .fund_cold_cache_views(&permit)
        .map_err(|error| after_upload(ProviderExecutionError::from_source(error)))?;
    Ok(Some(receipt))
}
