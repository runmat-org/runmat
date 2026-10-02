//! One numerical coefficient source for each provider IIR invocation.
use anyhow::{ensure, Result};
use serde::Serialize;

use crate::{
    handle_integer_type, handle_is_logical, handle_precision, handle_storage, provider_for_handle,
    same_provider_instance, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView,
};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ProviderIirCoefficientSource<'a> {
    /// The provider reads and normalizes the resident coefficient buffers.
    Resident {
        numerator: &'a GpuTensorHandle,
        denominator: &'a GpuTensorHandle,
    },
    /// The denominator is the scalar one; the numerator remains resident.
    UnitDenominatorFir { numerator: &'a GpuTensorHandle },
    /// One checked, uploaded normalization reused by subsequent passes.
    Prepared(&'a ProviderPreparedIirCoefficients),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderIirFilterOptions {
    /// Zero-based dimension along which filtering should be applied.
    pub dim: usize,
    /// Optional initial conditions (state vector) residing on the device.
    pub zi: Option<GpuTensorHandle>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderIirFilterResult {
    /// Filtered output tensor, matching the input signal shape.
    pub output: GpuTensorHandle,
    /// Final conditions for the filter state (same shape as the requested `zi` layout).
    pub final_state: Option<GpuTensorHandle>,
}

/// Provider-bound normalized coefficients. The constructor checks the structural
/// normalization contract and owns both uploads. A failed second upload retires
/// the first rather than publishing a partial preparation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderPreparedIirCoefficients {
    numerator: GpuTensorHandle,
    denominator: GpuTensorHandle,
}

impl ProviderPreparedIirCoefficients {
    pub fn upload(
        provider: &'static dyn AccelProvider,
        numerator: &[f64],
        denominator: &[f64],
    ) -> Result<Self> {
        ensure!(
            !numerator.is_empty() && numerator.len() == denominator.len(),
            "iir_filter: prepared coefficient lengths are incompatible"
        );
        ensure!(
            denominator.first() == Some(&1.0),
            "iir_filter: prepared leading denominator must equal one"
        );
        let shape = [numerator.len(), 1];
        let numerator_handle = provider.upload(&HostTensorView {
            data: numerator,
            shape: &shape,
        })?;
        validate_prepared_handle(provider, &numerator_handle, &shape)?;
        let denominator_handle = provider.upload(&HostTensorView {
            data: denominator,
            shape: &shape,
        })?;
        validate_prepared_handle(provider, &denominator_handle, &shape)?;
        ensure!(
            !numerator_handle.shares_allocation_with(&denominator_handle),
            "iir_filter: prepared coefficient allocations must be distinct"
        );
        Ok(Self {
            numerator: numerator_handle,
            denominator: denominator_handle,
        })
    }

    pub fn numerator(&self) -> &GpuTensorHandle {
        &self.numerator
    }

    pub fn denominator(&self) -> &GpuTensorHandle {
        &self.denominator
    }
}

fn validate_prepared_handle(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    shape: &[usize],
) -> Result<()> {
    handle.allocation_lease().validate_live()?;
    ensure!(
        handle.shape() == shape,
        "iir_filter: invalid prepared coefficient shape"
    );
    ensure!(
        handle.device_id() == provider.device_id()
            && provider_for_handle(handle)
                .is_some_and(|owner| same_provider_instance(owner, provider)),
        "iir_filter: prepared coefficient owner mismatch"
    );
    ensure!(
        handle_storage(handle) == GpuTensorStorage::Real
            && !handle_is_logical(handle)
            && handle_integer_type(handle).is_none()
            && handle_precision(handle) == Some(provider.precision()),
        "iir_filter: prepared coefficient class mismatch"
    );
    Ok(())
}
