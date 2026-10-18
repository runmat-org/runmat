use runmat_value::{
    ExactHostComplexTensorAllocation, ExactHostLogicalTensorAllocation, ExactHostNumericAllocation,
    ExactHostNumericAllocationError, ExactHostTensorAllocation, NumericDType,
};

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
    pub fn retained_view_issuer_bytes() -> Result<u64, HostAllocationError> {
        super::receipt::retained_view_issuer_host_upper_bytes()
    }
    pub fn retained_view_bytes(view_bytes: u64) -> Result<u64, HostAllocationError> {
        view_bytes
            .checked_add(crate::context::host_effect::retained_claim_host_upper_bytes())
            .and_then(|bytes| {
                bytes.checked_add(super::receipt::retained_view_grant_host_upper_bytes())
            })
            .ok_or(HostAllocationError::Overflow)
    }
    /// Variable retained cache-key payload plus the managed claim record.
    pub fn retained_metadata_bytes(payload_bytes: u64) -> Result<u64, HostAllocationError> {
        payload_bytes
            .checked_add(crate::context::host_effect::retained_claim_host_upper_bytes())
            .ok_or(HostAllocationError::Overflow)
    }

    /// Exact-output physical bytes plus the canonical managed permit's
    /// separate claim-control allocation for each retained owner. A tensor
    /// with independently claimed payload and shape supplies two claims.
    pub fn retained_exact_output_bytes(
        physical_bytes: u64,
        retained_claims: usize,
    ) -> Result<u64, HostAllocationError> {
        let per_claim =
            u64::try_from(crate::context::host_effect::owned_output_claim_metadata_bytes())
                .map_err(|_| HostAllocationError::Overflow)?;
        let controls = u64::try_from(retained_claims)
            .ok()
            .and_then(|count| count.checked_mul(per_claim))
            .ok_or(HostAllocationError::Overflow)?;
        physical_bytes
            .checked_add(controls)
            .ok_or(HostAllocationError::Overflow)
    }

    /// One retained exact dense tensor, including its separately claimed
    /// payload and immutable shape. This is a final-output charge, not a
    /// transient computation scratch bound.
    pub fn retained_tensor_bytes(
        dtype: NumericDType,
        shape: &[usize],
    ) -> Result<u64, HostAllocationError> {
        let raw = ExactHostTensorAllocation::requested_bytes(dtype, shape)?;
        Self::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
            ExactHostTensorAllocation::retained_claim_count(),
        )
    }

    pub fn retained_tensor_bytes_for_dimensions(
        dtype: NumericDType,
        count: usize,
        rank: usize,
    ) -> Result<u64, HostAllocationError> {
        let raw = ExactHostTensorAllocation::requested_bytes_for_dimensions(dtype, count, rank)?;
        Self::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
            ExactHostTensorAllocation::retained_claim_count(),
        )
    }

    pub fn retained_complex_tensor_bytes(
        dtype: NumericDType,
        shape: &[usize],
    ) -> Result<u64, HostAllocationError> {
        let raw = ExactHostComplexTensorAllocation::requested_bytes(dtype, shape)?;
        Self::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
            ExactHostComplexTensorAllocation::retained_claim_count(),
        )
    }

    pub fn retained_complex_tensor_bytes_for_dimensions(
        dtype: NumericDType,
        count: usize,
        rank: usize,
    ) -> Result<u64, HostAllocationError> {
        let raw =
            ExactHostComplexTensorAllocation::requested_bytes_for_dimensions(dtype, count, rank)?;
        Self::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
            ExactHostComplexTensorAllocation::retained_claim_count(),
        )
    }

    pub fn retained_logical_tensor_bytes(shape: &[usize]) -> Result<u64, HostAllocationError> {
        let raw = ExactHostLogicalTensorAllocation::requested_bytes(shape)?;
        Self::retained_exact_output_bytes(
            u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
            ExactHostLogicalTensorAllocation::retained_claim_count(),
        )
    }
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

    /// One interleaved complex64 scratch buffer backed by a managed F64
    /// allocation. The checked factor of two is the real/imaginary lane count.
    pub fn complex_workspace_bytes(len: usize) -> Result<u64, HostAllocationError> {
        let lanes = len.checked_mul(2).ok_or(HostAllocationError::Overflow)?;
        Self::numeric_workspace_bytes(NumericDType::F64, lanes)
    }
}

impl From<ExactHostNumericAllocationError> for HostAllocationError {
    fn from(error: ExactHostNumericAllocationError) -> Self {
        Self::Numeric(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_tensor_quote_includes_two_independent_retained_claims() {
        let shape = [2, 3];
        let raw = ExactHostTensorAllocation::requested_bytes(NumericDType::F32, &shape).unwrap();
        let controls =
            2_u64 * crate::context::host_effect::owned_output_claim_metadata_bytes() as u64;
        assert_eq!(
            HostAllocationPlan::retained_tensor_bytes(NumericDType::F32, &shape).unwrap(),
            raw as u64 + controls
        );
    }
}
