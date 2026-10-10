use crate::{
    handle_integer_type, handle_is_logical, handle_precision, provider_for_handle,
    same_provider_instance, AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage,
    PowerResultDomain, ProviderExecutionError, ProviderPrecision,
};

/// Provider-local failure with explicit effect boundary. A caller may use a
/// host route only for an unsupported pre-effect error or `prepare == None`.
#[derive(Debug)]
pub struct PowerServiceError(pub ProviderExecutionError);

impl PowerServiceError {
    pub(crate) fn pre(message: &'static str) -> Self {
        Self(ProviderExecutionError::from(anyhow::anyhow!(message)))
    }
    pub(crate) fn after(error: impl Into<ProviderExecutionError>) -> Self {
        Self(error.into().after_effect())
    }
    pub(crate) fn unproven(error: impl Into<ProviderExecutionError>) -> Self {
        Self(error.into().with_unproven_effects())
    }
    pub fn is_unsupported(&self) -> bool {
        self.0.is_unsupported()
    }
    pub fn into_execution_error(self) -> ProviderExecutionError {
        self.0
    }
}

impl From<anyhow::Error> for PowerServiceError {
    fn from(error: anyhow::Error) -> Self {
        Self(ProviderExecutionError::from(error))
    }
}
impl From<std::num::TryFromIntError> for PowerServiceError {
    fn from(error: std::num::TryFromIntError) -> Self {
        Self::from(anyhow::Error::from(error))
    }
}
impl std::fmt::Display for PowerServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for PowerServiceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

pub(crate) fn valid_output(
    output: &GpuTensorHandle,
    provider: &dyn AccelProvider,
    shape: &[usize],
    precision: ProviderPrecision,
    storage: GpuTensorStorage,
    inputs: &[&GpuTensorHandle],
) -> bool {
    output.shape() == shape
        && output.device_id() == provider.device_id()
        && provider_for_handle(output).is_some_and(|owner| same_provider_instance(owner, provider))
        && output.allocation_lease().validate_live().is_ok()
        && output.physical().storage() == storage
        && output.descriptor().storage == Some(storage)
        && output.descriptor().element_type == Some(output.physical().element_type())
        && handle_precision(output) == Some(precision)
        && crate::handle_class_identity(output)
            == Some(match precision {
                ProviderPrecision::F32 => runmat_types::standard::SINGLE.owned(),
                ProviderPrecision::F64 => runmat_types::standard::DOUBLE.owned(),
            })
        && !handle_is_logical(output)
        && handle_integer_type(output).is_none()
        && inputs
            .iter()
            .all(|input| !output.shares_allocation_with(input))
}

pub(crate) fn reject(output: GpuTensorHandle, detail: &'static str) -> PowerServiceError {
    // release consumes only this result reference. A borrowed input retains
    // its own lease, while a distinct view's residency-cleanup failure is
    // captured here rather than hidden by Drop.
    let cleanup = output.release().err();
    let primary = PowerServiceError::pre(detail)
        .into_execution_error()
        .after_effect();
    match cleanup {
        Some(cleanup) => PowerServiceError(
            ProviderExecutionError::from_source(crate::ProviderExecutionCompletionError::new(
                Some(primary),
                vec![cleanup],
            ))
            .after_effect(),
        ),
        None => PowerServiceError(primary),
    }
}

pub(crate) fn publish_provenance(
    output: GpuTensorHandle,
    resident: &GpuTensorHandle,
) -> Result<GpuTensorHandle, PowerServiceError> {
    let provenance = if crate::handle_is_explicit(resident) {
        GpuHandleProvenance::Explicit
    } else {
        GpuHandleProvenance::Automatic
    };
    match output.with_provenance(provenance) {
        Ok(published) => Ok(published),
        Err(error) => {
            let cleanup = output.release().err();
            let primary = ProviderExecutionError::from(anyhow::Error::from(error)).after_effect();
            Err(match cleanup {
                Some(cleanup) => PowerServiceError(
                    ProviderExecutionError::from_source(
                        crate::ProviderExecutionCompletionError::new(Some(primary), vec![cleanup]),
                    )
                    .after_effect(),
                ),
                None => PowerServiceError(primary),
            })
        }
    }
}

pub(crate) fn storage(domain: PowerResultDomain) -> GpuTensorStorage {
    match domain {
        PowerResultDomain::Real => GpuTensorStorage::Real,
        PowerResultDomain::Complex => GpuTensorStorage::ComplexInterleaved,
    }
}
