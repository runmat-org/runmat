use runmat_value::{ExactHostNumericAllocation, ExactHostNumericAllocationError, NumericDType};

use super::HostAllocationError;

/// Checked maximum scratch precharge and independently retained output
/// headroom for one call. A domain must count actual overlapping phases,
/// including storage and shape metadata; the plan is not a quota lease.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostAllocationPlan {
    scratch_peak_bytes: u64,
    retained_output_bytes: u64,
}

impl HostAllocationPlan {
    pub fn new(
        scratch_peak_bytes: u64,
        retained_output_bytes: u64,
    ) -> Result<Self, HostAllocationError> {
        scratch_peak_bytes
            .checked_add(retained_output_bytes)
            .ok_or(HostAllocationError::Overflow)?;
        Ok(Self {
            scratch_peak_bytes,
            retained_output_bytes,
        })
    }

    pub fn scratch_peak_bytes(self) -> u64 {
        self.scratch_peak_bytes
    }

    pub fn retained_output_bytes(self) -> u64 {
        self.retained_output_bytes
    }

    /// One exact numeric allocation, including its retained release/buffer
    /// metadata. Callers must still sum simultaneous owners into phase peak.
    pub fn numeric_bytes(dtype: NumericDType, len: usize) -> Result<u64, HostAllocationError> {
        let bytes = ExactHostNumericAllocation::requested_bytes(dtype, len)
            .map_err(HostAllocationError::Numeric)?;
        u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)
    }

    /// One stage-backed numeric scratch allocation, including the two exact
    /// claim owners created by its permit. This is not retained-output size.
    pub fn numeric_workspace_bytes(
        dtype: NumericDType,
        len: usize,
    ) -> Result<u64, HostAllocationError> {
        let numeric = Self::numeric_bytes(dtype, len)?;
        let claim = u64::try_from(super::temporary::scratch_claim_bytes())
            .map_err(|_| HostAllocationError::Overflow)?;
        numeric
            .checked_add(claim)
            .ok_or(HostAllocationError::Overflow)
    }
}

impl From<ExactHostNumericAllocationError> for HostAllocationError {
    fn from(error: ExactHostNumericAllocationError) -> Self {
        Self::Numeric(error)
    }
}
